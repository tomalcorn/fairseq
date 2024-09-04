# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.functional as F
import numpy as np

from fairseq.data import ConcatDataset, Dictionary
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.audio.audio_utils import get_features_or_waveform
from fairseq.data.audio.data_cfg import S2SDataConfig
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    TextTargetMultitaskData,
    _collate_frames,
)

logger = logging.getLogger(__name__)


@dataclass
class SpeechToSpeechDatasetItem(object):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    target_speaker: Optional[torch.Tensor] = None
    tgt_lang_tag: Optional[int] = None

@dataclass
class DocSpeechToSpeechDatasetItem(SpeechToSpeechDatasetItem):
    source_mask: Optional[torch.tensor] = None
    target_mask: Optional[torch.tensor] = None
    prev_idxs: Optional[list] = None

class SpeechToSpeechDataset(SpeechToTextDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: S2SDataConfig,
        src_audio_paths: List[str],
        src_n_frames: List[int],
        tgt_audio_paths: List[str],
        tgt_n_frames: List[int],
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        target_is_code: bool = False,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
    ):
        tgt_texts = tgt_audio_paths if target_is_code else None
        super().__init__(
            split=split,
            is_train_split=is_train_split,
            cfg=data_cfg,
            audio_paths=src_audio_paths,
            n_frames=src_n_frames,
            ids=ids,
            tgt_dict=tgt_dict,
            tgt_texts=tgt_texts,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            n_frames_per_step=n_frames_per_step,
        )

        self.tgt_audio_paths = tgt_audio_paths
        self.tgt_lens = [t // self.n_frames_per_step for t in tgt_n_frames]

        assert not target_is_code or tgt_dict is not None
        self.target_is_code = target_is_code

        assert len(tgt_audio_paths) == self.n_samples
        assert len(tgt_n_frames) == self.n_samples

        self.tgt_speakers = None
        if self.cfg.target_speaker_embed:
            samples = SpeechToTextDatasetCreator._load_samples_from_tsv(
                self.cfg.target_speaker_embed, split
            )
            spk_emb_dict = {s["id"]: s["speaker_embed"] for s in samples}
            self.tgt_speakers = [spk_emb_dict[id] for id in self.ids]
            assert len(self.tgt_speakers) == self.n_samples

        logger.info(self.__repr__())

    def pack_units(self, input: torch.Tensor) -> torch.Tensor:
        if self.n_frames_per_step <= 1:
            return input

        offset = 4
        vocab_size = (
            len(self.tgt_dict) - offset
        )  # remove offset from <bos>, <pad>, <eos>, <unk>, which is specific to fairseq dictionary

        assert input.dim() == 1
        stacked_input = (
            input[:-1].view(-1, self.n_frames_per_step) - offset
        )  # remove <eos>
        scale = [
            pow(vocab_size, self.n_frames_per_step - 1 - i)
            for i in range(self.n_frames_per_step)
        ]
        scale = torch.LongTensor(scale).squeeze(0)
        res = input.new((len(input) - 1) // self.n_frames_per_step + 1).fill_(input[-1])
        res[:-1] = (stacked_input * scale).sum(dim=1) + offset

        return res

    def __getitem__(self, index: int) -> SpeechToSpeechDatasetItem:
        source = self._get_source_audio(index)

        tgt_lang_tag = None
        if self.cfg.prepend_tgt_lang_tag_as_bos:
            # prepend_tgt_lang_tag_as_bos: put tgt_lang_tag as bos of target
            tgt_lang_tag = self.get_lang_tag_idx(self.tgt_langs[index], self.tgt_dict)

        if not self.target_is_code:
            target = get_features_or_waveform(self.tgt_audio_paths[index])
            target = torch.from_numpy(target).float()
            target = self.pack_frames(target)
        else:
            target = self.tgt_dict.encode_line(
                self.tgt_audio_paths[index],
                add_if_not_exist=False,
                append_eos=True,
            ).long()
            if self.n_frames_per_step > 1:
                n_tgt_frame = target.size(0) - 1  # exclude <eos>
                keep_n_tgt_frame = n_tgt_frame - n_tgt_frame % self.n_frames_per_step
                target = torch.cat(
                    (
                        target[:keep_n_tgt_frame],
                        target.new_full((1,), self.tgt_dict.eos()),
                    ),
                    dim=0,
                )

        if self.tgt_speakers:
            tgt_spk = get_features_or_waveform(self.tgt_speakers[index])
            tgt_spk = torch.from_numpy(tgt_spk).float()
        else:
            tgt_spk = torch.FloatTensor([])

        return SpeechToSpeechDatasetItem(
            index=index,
            source=source,
            target=target,
            target_speaker=tgt_spk,
            tgt_lang_tag=tgt_lang_tag,
        )

    def _collate_target(self, samples: List[SpeechToSpeechDatasetItem]) -> torch.Tensor:
        if self.target_is_code:
            target = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            # convert stacked units to a single id
            pack_targets = [self.pack_units(x.target) for x in samples]
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                pack_targets,
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            target_lengths = torch.tensor(
                [x.size(0) for x in pack_targets], dtype=torch.long
            )
        else:
            target = _collate_frames([x.target for x in samples], is_audio_input=False)
            bsz, _, d = target.size()
            prev_output_tokens = torch.cat(
                (target.new_full((bsz, 1, d), 0.0), target[:, :-1, :]), dim=1
            )
            target_lengths = torch.tensor(
                [x.target.size(0) for x in samples], dtype=torch.long
            )

        return target, prev_output_tokens, target_lengths

    def collater(
        self, samples: List[SpeechToSpeechDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        frames = _collate_frames([x.source for x in samples], self.cfg.use_audio_input)
        # sort samples by descending number of frames
        n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        target, prev_output_tokens, target_lengths = self._collate_target(samples)
        target = target.index_select(0, order)
        target_lengths = target_lengths.index_select(0, order)
        prev_output_tokens = prev_output_tokens.index_select(0, order)
        ntokens = sum(x.target.size(0) for x in samples)

        tgt_speakers = None
        if self.cfg.target_speaker_embed:
            tgt_speakers = _collate_frames(
                [x.target_speaker for x in samples], is_audio_input=True
            ).index_select(0, order)

        net_input = {
            "src_tokens": frames,
            "src_lengths": n_frames,
            "prev_output_tokens": prev_output_tokens,
            "tgt_speaker": tgt_speakers,  # TODO: unify "speaker" and "tgt_speaker"
        }
        if self.tgt_texts is not None and samples[0].tgt_lang_tag is not None:
            for i in range(len(samples)):
                net_input["prev_output_tokens"][i][0] = samples[order[i]].tgt_lang_tag
        out = {
            "id": indices,
            "net_input": net_input,
            "speaker": tgt_speakers,  # to support Tacotron2 loss for speech-to-spectrogram model
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        if return_order:
            out["order"] = order
        return out


class SpeechToSpeechMultitaskDataset(SpeechToSpeechDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multitask_data = {}

    def add_multitask_dataset(self, task_name, task_data):
        self.multitask_data[task_name] = task_data

    def __getitem__(
        self, index: int
    ) -> Tuple[SpeechToSpeechDatasetItem, Dict[str, torch.Tensor]]:
        s2s_data = super().__getitem__(index)

        multitask_target = {}
        sample_id = self.ids[index]
        tgt_lang = self.tgt_langs[index]
        for task_name, task_dataset in self.multitask_data.items():
            multitask_target[task_name] = task_dataset.get(sample_id, tgt_lang)

        return s2s_data, multitask_target

    def collater(
        self, samples: List[Tuple[SpeechToSpeechDatasetItem, Dict[str, torch.Tensor]]]
    ) -> Dict:
        if len(samples) == 0:
            return {}

        out = super().collater([s for s, _ in samples], return_order=True)
        order = out["order"]
        del out["order"]

        for task_name, task_dataset in self.multitask_data.items():
            if "multitask" not in out:
                out["multitask"] = {}
            d = [s[task_name] for _, s in samples]
            task_target = task_dataset.collater(d)
            out["multitask"][task_name] = {
                "target": task_target["target"].index_select(0, order),
                "target_lengths": task_target["target_lengths"].index_select(0, order),
                "ntokens": task_target["ntokens"],
            }
            out["multitask"][task_name]["net_input"] = {
                "prev_output_tokens": task_target["prev_output_tokens"].index_select(
                    0, order
                ),
            }

        return out


class DocSpeechtoSpeechMultitaskDataset(SpeechToSpeechMultitaskDataset):
    def __init__(self, 
                 doc_ids: List[str],
                 doc_pos_idxs: List[int],
                 doc_context_size: int = 1, 
                 scramble_source: bool = False,
                 scramble_target: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.doc_ids = doc_ids
        self.doc_pos_idxs = doc_pos_idxs
        self.doc_context_size = doc_context_size
        self.scramble_source = scramble_source
        self.scramble_target = scramble_target
        
    def get_scrambled_index(self, doc_id, prev_doc_pos_idx, doc_type):
        """Helper function to get a valid previous index, either scrambled or sequentially."""
        max_tries = 10  # Avoid infinite loops; you can adjust this based on your needs
        for _ in range(max_tries):
            if doc_type == 'source':
                scramble = self.scramble_source
            elif doc_type == 'target':
                scramble = self.scramble_target
            else:
                raise ValueError("doc_type must be either 'source' or 'target'")

            if scramble:
                index = np.random.randint(0, len(self.doc_ids))
                scrambled_doc_id = self.doc_ids[index]
            else:
                scrambled_doc_id = doc_id

            try:
                prev_idx = self._find_prev_idx(scrambled_doc_id, prev_doc_pos_idx)
                return prev_idx
            except ValueError:
                continue
        
        # If it fails to find a valid index, return the original one as a fallback
        return self._find_prev_idx(doc_id, prev_doc_pos_idx)


    def __getitem__(self, index: int) -> Tuple[DocSpeechToSpeechDatasetItem, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        s2s_data, multitask_target = super().__getitem__(index)
        
        multi_keys = list(multitask_target.keys())
        mask_info = {}
        
        for multitask in multi_keys:
            new_mask_len = f'original_{multitask}_length'
            mask_info[new_mask_len] = len(multitask_target[multitask])
        
        # Initialize total source - concat of prev context and current source tokens
        total_source = s2s_data.source
        total_target = s2s_data.target
        original_source_length = len(total_source)
        original_target_length = len(total_target)
        
        doc_id = self.doc_ids[index]
        doc_pos_idx = self.doc_pos_idxs[index]
        max_doc_pos_idx = max(self.doc_pos_idxs)
        contexts_catted = 0
        prev_idxs = []
        prev_doc_pos_idx = doc_pos_idx - 1
        
        while contexts_catted < self.doc_context_size and prev_doc_pos_idx >= 0:
            
            if doc_pos_idx < max_doc_pos_idx:
                # Handle source scrambling or sequential retrieval
                prev_source_idx = self.get_scrambled_index(doc_id, prev_doc_pos_idx, 'source')
                
                # Handle target scrambling or sequential retrieval
                prev_target_idx = self.get_scrambled_index(doc_id, prev_doc_pos_idx, 'target')
            else:
                # If no scrambling or already at max_doc_pos_idx, use the current doc_id
                prev_source_idx = prev_doc_pos_idx
                prev_target_idx = prev_doc_pos_idx
            
            prev_idxs.append(prev_target_idx)
            
            prev_source_item, _ = super().__getitem__(prev_source_idx)
            prev_target_item, prev_multitask_target = super().__getitem__(prev_target_idx)
            # Change EOS to BOS
            prev_target_item.target[-1] = self.tgt_dict.bos()
            for multitask in multi_keys:
                prev_multitask_target[multitask][-1] = self.tgt_dict.bos()
                multitask_target[multitask] = torch.cat([prev_multitask_target[multitask], multitask_target[multitask]])
            total_source = torch.cat([prev_source_item.source, total_source])
            total_target = torch.cat([prev_target_item.target, total_target])
            prev_doc_pos_idx -= 1
            contexts_catted += 1
        
        # Create the source and target masks
        source_mask = torch.zeros(total_source.shape[0])
        source_mask[-original_source_length - 1:] = 1  # Set 1s for the sentence input, PLUS BOS TOKEN
        
        target_mask = torch.zeros(total_target.shape)
        target_mask[-original_target_length - 1:] = 1  # Set 1s for the sentence input
        
        mask_info['source_mask'] = source_mask
        mask_info['target_mask'] = target_mask
        
        for multitask in multi_keys:
            mask_name = f'{multitask}_mask'
            mask = torch.zeros(multitask_target[multitask].shape)
            original_length = mask_info.get(f'original_{multitask}_length')
            mask[-original_length - 1:] = 1
            mask_info[mask_name] = mask
        
        # Update s2s_data with the new source and target
        s2s_data.source = total_source
        s2s_data.target = total_target
        s2s_data.prev_idxs = prev_idxs
        
        return s2s_data, multitask_target, mask_info

        
    def _find_prev_idx(self, doc_id: str, prev_doc_pos_idx: int) -> int:
    
        # Convert lists to numpy arrays for efficient comparison
        doc_ids_array = np.array(self.doc_ids)
        doc_pos_idxs_array = np.array(self.doc_pos_idxs)

        # Create a boolean mask for matching doc_ids
        doc_id_mask = (doc_ids_array == doc_id)

        # Create a boolean mask for matching prev_doc_pos_idx
        pos_idx_mask = (doc_pos_idxs_array == prev_doc_pos_idx)

        # Combine the masks
        combined_mask = doc_id_mask & pos_idx_mask

        # Find the indices where the combined mask is True
        matching_indices = np.where(combined_mask)[0]
            
        if len(matching_indices) == 0:
            raise ValueError(f"No matching index found for doc_id {doc_id} and position {prev_doc_pos_idx}")
        
        # Return the first (and should be only) matching index
        return matching_indices[0]
        
        
    def collater(self, samples: List[Tuple[DocSpeechToSpeechDatasetItem, Dict[str, torch.Tensor]]]) -> Dict:
        # Split the samples into s2s_multitask_samples and mask_infos
        s2s_multitask_samples = [s[:2] for s in samples]
        mask_infos = [s[2] for s in samples]

        # Call the parent collater
        out = super().collater(s2s_multitask_samples)

        # Create a mapping from sample id to index in out
        id_to_index = {id.item(): i for i, id in enumerate(out['id'])}

        # Initialize the mask_info dictionary in the output
        out['mask_info'] = {}

        # Get the padded lengths for source and target
        src_lengths = out['net_input']['src_lengths']
        tgt_lengths = out['target_lengths']

        # Create new masks of all zeros based on padded lengths
        batch_size, max_src_len, *_ = out['net_input']['src_tokens'].shape
        batch_size, max_tgt_len = out['target'].shape
        src_masks = torch.zeros((batch_size, max_src_len), dtype=torch.long)
        tgt_masks = torch.zeros((batch_size, max_tgt_len), dtype=torch.long)
        prev_output_masks = torch.zeros((batch_size, max_tgt_len), dtype=torch.long)

        # Initialize multitask_mask_dic
        multitask_mask_dic = {}
        for task in out['multitask']:
            multitask_mask_dic[f'{task}_lengths'] = out['multitask'][task]['target_lengths']
            max_task_len = out['multitask'][task]['target'].shape[1]
            multitask_mask_dic[f'{task}_masks'] = torch.zeros((batch_size, max_task_len), dtype=torch.long)
             
        # Fill in the masks based on original sample masks
        for i, (sample, mask_info) in enumerate(zip(samples, mask_infos)):
            sample_id = sample[0].index
            if sample_id not in id_to_index:
                continue  # Skip this sample if it's not in the output (e.g., filtered out)
            out_index = id_to_index[sample_id]

            src_mask_len = min(len(mask_info['source_mask']), src_lengths[out_index])
            tgt_mask_len = min(len(mask_info['target_mask']), tgt_lengths[out_index])

            src_masks[out_index, :src_mask_len] = self._to_tensor(mask_info['source_mask'][:src_mask_len])
            tgt_masks[out_index, :tgt_mask_len] = self._to_tensor(mask_info['target_mask'][:tgt_mask_len])
            prev_output_masks[out_index, :tgt_mask_len-1] = self._to_tensor(mask_info['target_mask'][:tgt_mask_len-1])
            prev_output_masks[out_index, -1] = self._to_tensor(mask_info['target_mask'][-1])  # EOS token mask
            prev_output_masks[out_index] = torch.cat([prev_output_masks[out_index, -1:], prev_output_masks[out_index, :-1]])  # Move EOS to start

            target_seq = out['target'][out_index]
            target_mask = tgt_masks[out_index]

            # Check the last few tokens and their corresponding mask values
            last_tokens = target_seq[-3:]
            last_mask_values = target_mask[-3:]
            if torch.any(last_tokens != self.tgt_dict.pad()) and torch.all(last_mask_values == 0):
                print(f"Mismatch detected in sample {out_index}:")
                print(f"Last 3 tokens: {last_tokens}")
                print(f"Last 3 mask values: {last_mask_values}")
                print(f"Full target sequence: {target_seq}")
                print(f"Full target mask: {target_mask}")
                print(f"Original mask from mask_info: {mask_info['target_mask']}")

            for task in out['multitask']:
                task_lengths = multitask_mask_dic[f'{task}_lengths']
                task_mask_len = min(len(mask_info[f'{task}_mask']), task_lengths[out_index])
                multitask_mask_dic[f'{task}_masks'][out_index, :task_mask_len] = self._to_tensor(
                    mask_info[f'{task}_mask'][:task_mask_len]
                )

        # Add masks to the output dictionary
        out['mask_info']['src_masks'] = src_masks
        out['mask_info']['tgt_masks'] = tgt_masks
        out['mask_info']['prev_output_masks'] = prev_output_masks

        # Add multitask masks to the output dictionary
        for task in out['multitask']:
            out['mask_info'][f'{task}_masks'] = multitask_mask_dic[f'{task}_masks']

        return out
        

    def _to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.clone().detach()
        else:
            return torch.tensor(data, dtype=torch.long)
    

class SpeechToSpeechDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_SRC_AUDIO, KEY_SRC_N_FRAMES = "id", "src_audio", "src_n_frames"
    KEY_TGT_AUDIO, KEY_TGT_N_FRAMES = "tgt_audio", "tgt_n_frames"
    # optional columns
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_LANG = ""

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        data_cfg: S2SDataConfig,
        target_is_code: bool = False,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
        multitask: Optional[Dict] = None,
    ) -> SpeechToSpeechDataset:
        audio_root = Path(data_cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        src_audio_paths = [
            (audio_root / s[cls.KEY_SRC_AUDIO]).as_posix() for s in samples
        ]
        tgt_audio_paths = [
            s[cls.KEY_TGT_AUDIO]
            if target_is_code
            else (audio_root / s[cls.KEY_TGT_AUDIO]).as_posix()
            for s in samples
        ]
        src_n_frames = [int(s[cls.KEY_SRC_N_FRAMES]) for s in samples]
        tgt_n_frames = [int(s[cls.KEY_TGT_N_FRAMES]) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]

        has_multitask = multitask is not None and len(multitask.keys()) > 0
        dataset_cls = (
            SpeechToSpeechMultitaskDataset if has_multitask else SpeechToSpeechDataset
        )

        ds = dataset_cls(
            split=split_name,
            is_train_split=is_train_split,
            data_cfg=data_cfg,
            src_audio_paths=src_audio_paths,
            src_n_frames=src_n_frames,
            tgt_audio_paths=tgt_audio_paths,
            tgt_n_frames=tgt_n_frames,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            target_is_code=target_is_code,
            tgt_dict=tgt_dict,
            n_frames_per_step=n_frames_per_step,
        )

        if has_multitask:
            for task_name, task_obj in multitask.items():
                task_data = TextTargetMultitaskData(
                    task_obj.args, split_name, task_obj.target_dictionary
                )
                ds.add_multitask_dataset(task_name, task_data)
        return ds

    @classmethod
    def from_tsv(
        cls,
        root: str,
        data_cfg: S2SDataConfig,
        splits: str,
        is_train_split: bool,
        epoch: int,
        seed: int,
        target_is_code: bool = False,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
        multitask: Optional[Dict] = None,
    ) -> SpeechToSpeechDataset:
        datasets = []
        for split in splits.split(","):
            samples = SpeechToTextDatasetCreator._load_samples_from_tsv(root, split)
            ds = cls._from_list(
                split_name=split,
                is_train_split=is_train_split,
                samples=samples,
                data_cfg=data_cfg,
                target_is_code=target_is_code,
                tgt_dict=tgt_dict,
                n_frames_per_step=n_frames_per_step,
                multitask=multitask,
            )
            datasets.append(ds)
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]


class DocSpeechtoSpeechDatasetCreator(SpeechToSpeechDatasetCreator):
    # mandatory columns
    KEY_ID, KEY_SRC_AUDIO, KEY_SRC_N_FRAMES = "id", "src_audio", "src_n_frames"
    KEY_TGT_AUDIO, KEY_TGT_N_FRAMES = "tgt_audio", "tgt_n_frames"
    # optional columns
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_LANG = ""
    
    # doc values
    DOC_ID = "doc_id"
    DOC_POS_IDX = "doc_pos_idx"
    
    @classmethod
    def from_tsv(
        cls,
        root: str,
        data_cfg: S2SDataConfig,
        splits: str,
        is_train_split: bool,
        epoch: int,
        seed: int,
        target_is_code: bool = False,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
        multitask: Optional[Dict] = None,
        doc_context_size: int = 1,
        scramble_source: bool = False,
        scramble_target: bool = False
    ) -> SpeechToSpeechDataset:
        datasets = []
        for split in splits.split(","):
            samples = SpeechToTextDatasetCreator._load_samples_from_tsv(root, split)
            ds = cls._from_list(
                split_name=split,
                is_train_split=is_train_split,
                samples=samples,
                data_cfg=data_cfg,
                target_is_code=target_is_code,
                tgt_dict=tgt_dict,
                n_frames_per_step=n_frames_per_step,
                multitask=multitask,
                doc_context_size=doc_context_size,
                scramble_source=scramble_source,
                scramble_target=scramble_target
            )
            datasets.append(ds)
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    
    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        data_cfg: S2SDataConfig,
        target_is_code: bool = False,
        tgt_dict: Dictionary = None,
        n_frames_per_step: int = 1,
        multitask: Optional[Dict] = None,
        doc_context_size: int = 1,
        scramble_source: bool = False,
        scramble_target: bool = False
    ) -> SpeechToSpeechDataset:
        audio_root = Path(data_cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        
        # MINE
        doc_ids = [s[cls.DOC_ID] for s in samples]
        doc_pos_idxs =[int(s[cls.DOC_POS_IDX]) for s in samples]
        
        src_audio_paths = [
            (audio_root / s[cls.KEY_SRC_AUDIO]).as_posix() for s in samples
        ]
        tgt_audio_paths = [
            s[cls.KEY_TGT_AUDIO]
            if target_is_code
            else (audio_root / s[cls.KEY_TGT_AUDIO]).as_posix()
            for s in samples
        ]
        src_n_frames = [int(s[cls.KEY_SRC_N_FRAMES]) for s in samples]
        tgt_n_frames = [int(s[cls.KEY_TGT_N_FRAMES]) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]

        has_multitask = multitask is not None and len(multitask.keys()) > 0
        
        has_doc = doc_ids is not None
        
        dataset_cls = (
            DocSpeechtoSpeechMultitaskDataset if has_multitask and has_doc else 
            (SpeechToSpeechMultitaskDataset if has_multitask else SpeechToSpeechDataset)
        )
        
        ds = dataset_cls(
            split=split_name,
            is_train_split=is_train_split,
            data_cfg=data_cfg,
            src_audio_paths=src_audio_paths,
            src_n_frames=src_n_frames,
            tgt_audio_paths=tgt_audio_paths,
            tgt_n_frames=tgt_n_frames,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            doc_ids=doc_ids,
            doc_pos_idxs=doc_pos_idxs,
            target_is_code=target_is_code,
            tgt_dict=tgt_dict,
            n_frames_per_step=n_frames_per_step,
            doc_context_size=doc_context_size,
            scramble_source=scramble_source,
            scramble_target=scramble_target
        )

        if has_multitask:
            for task_name, task_obj in multitask.items():
                task_data = TextTargetMultitaskData(
                    task_obj.args, split_name, task_obj.target_dictionary
                )
                ds.add_multitask_dataset(task_name, task_data)
        return ds