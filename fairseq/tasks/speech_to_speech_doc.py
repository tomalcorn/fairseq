import json
import logging
import math
from argparse import Namespace
from pathlib import Path
from typing import List

import numpy as np
import os

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.data import Dictionary
from fairseq.data.audio.data_cfg import MultitaskConfig, S2SDataConfig
from fairseq.data.audio.speech_to_speech_dataset import SpeechToSpeechDatasetCreator, DocSpeechtoSpeechDatasetCreator
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    TextTargetMultitaskData,
)
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.speech_to_text import DummyMultiTask
from fairseq.tasks.text_to_speech import batch_mel_cepstral_distortion
from fairseq.tasks.speech_to_speech import SpeechToSpeechTask

logger = logging.getLogger(__name__)

@register_task("doc_speech_to_speech")
class DocSpeechToSpeechTask(SpeechToSpeechTask):
    def __init__(self, args, tgt_dict, infer_tgt_lang_id=None):
        super().__init__(args, tgt_dict, infer_tgt_lang_id)
        
    
    @classmethod
    def add_args(cls, parser):
        SpeechToSpeechTask.add_args(parser)
        parser.add_argument(
            "--doc-context-size", 
            type=int,
            default=1,
            help="number of concatenated prefix segments")
        parser.add_argument(
            "--imed-gamma",
            type=float,
            default=0.5,
            help="interpolation parameter for sentence level prediction."
        )
        parser.add_argument(
            "--use-imed",
            action='store_true',
            help="Whether to use imed interpolation"
        )
        parser.add_argument(
            "--use-prefix",
            action='store_true',
            help="whether to force ground truth units for prefix segments"
        )
        parser.add_argument(
            "--scramble-source",
            action='store_true',
            help="ablation tool to choose random prefix target and source prefix material"
        )
        parser.add_argument(
            "--scramble-target",
            action='store_true',
            help="ablation tool to choose random prefix target and source prefix material"
        )
        
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.datasets[split] = DocSpeechtoSpeechDatasetCreator.from_tsv(
            root=self.args.data,
            data_cfg=self.data_cfg,
            splits=split,
            is_train_split=split.startswith("train"),
            epoch=epoch,
            seed=self.args.seed,
            target_is_code=self.args.target_is_code,
            tgt_dict=self.target_dictionary,
            n_frames_per_step=self.args.n_frames_per_step,
            multitask=self.multitask_tasks,
            doc_context_size=self.args.doc_context_size,
            scramble_source=getattr(self.args, "scramble_source", False),
            scramble_target=getattr(self.args, "scramble_target", False)
        )
    
    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            target = sample['target']
            batch_size, seq_len = target.shape
            if self.args.use_prefix:
                bos_positions_batch = []
                for i in range(batch_size):
                    bos_positions = (target[i] == self.tgt_dict.bos()).nonzero(as_tuple=True)[0]
                    if len(bos_positions) > 0:
                        bos_positions_batch.append(bos_positions[-1].item())
                    else:
                        bos_positions_batch.append(seq_len - 1)  # If no BOS, use entire sequence
                
                bos_positions_batch = torch.tensor(bos_positions_batch, device=target.device)
                
                
                # Create a mask for valid positions
                mask = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(target.device)
                mask = mask <= bos_positions_batch.unsqueeze(1)
                
                # Apply the mask to get prefixes
                prefixes = target * mask
                
                # Replace padding with the padding index
                prefix_tokens = prefixes.masked_fill(~mask, self.tgt_dict.pad())
            
            if models[0].encoder.conv_version == 'afs':
                models[0].encoder.subsample.need_attn = True
            
            hypos = generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
                bos_token=None,
                use_imed=self.args.use_imed,
                imed_gamma=self.args.imed_gamma
            )

            # Modify hypos to only return the output for the current segment
            for j, hypo in enumerate(hypos):
                for i, beam in enumerate(hypo):
                    tokens = beam['tokens']
                    if i == 0:
                        first_tokens = tokens
                    target = sample['target']
                    # comparison  = torch.cat([tokens, target[j]], dim=1)
                    score = beam['score']
                    
                    # Find the last BOS token
                    bos_indices = (tokens == self.tgt_dict.bos()).nonzero(as_tuple=True)[0]
                    if len(bos_indices) > 0:
                        last_bos_index = bos_indices[-1]
                        current_segment_tokens = tokens[last_bos_index + 1:]
                    else:
                        current_segment_tokens = tokens
                    
                    # Create a new hypothesis for the current segment
                    hypo[i]['tokens'] = current_segment_tokens
                    hypo[i]['score'] = score

                # Save attention scores and mask for top beam
                need_attn = self.args.attn_save_dir
                
                if need_attn:
                    # Make dirs
                    os.makedirs(self.args.attn_save_dir, exist_ok=True)
                    attn_save_dir = os.path.join(self.args.attn_save_dir, "attn")
                    src_mask_save_dir = os.path.join(self.args.attn_save_dir, "src_segments")
                    tgt_mask_save_dir = os.path.join(self.args.attn_save_dir, "tgt_segments")
                    os.makedirs(attn_save_dir, exist_ok=True)
                    os.makedirs(src_mask_save_dir, exist_ok=True)
                    os.makedirs(tgt_mask_save_dir, exist_ok=True)
                    
                    # Get id for filenames
                    sample_id = self.datasets['test'].ids[sample['id'][j]]
                    
                    # Attention masks
                    attn_scores = hypo[0]['attention']
                    attn_filename = os.path.join(attn_save_dir, f"{sample_id}.npy")
                    
                    # source masks
                    src_mask = sample['mask_info']['src_masks'][j]
                    
                    # Reduce source mask if using AFS at inference
                    if models[0].encoder.conv_version == 'afs':
                        sorted_indices = models[0].encoder.subsample.sorted_indices[:, j]
                        src_mask = self.update_source_mask(src_mask, sorted_indices) 
                    
                    src_mask_filename = os.path.join(src_mask_save_dir, f"{sample_id}.npy")
                    
                    # Extract target mask
                    tgt_mask = torch.zeros_like(first_tokens, dtype=torch.bool)
                    bos_indices = (first_tokens == self.tgt_dict.bos()).nonzero(as_tuple=True)[0]
                    if len(bos_indices) > 0:
                        tgt_mask[bos_indices[-1]:] = True
                    tgt_mask_filename = os.path.join(tgt_mask_save_dir, f"{sample_id}.npy")
                    
                    np.save(attn_filename, attn_scores.cpu().numpy())
                    np.save(src_mask_filename, src_mask.cpu().numpy())
                    np.save(tgt_mask_filename, tgt_mask.cpu().numpy())
            
            return hypos
            

    def build_criterion(self, args):
        from fairseq import criterions

        if len(self.multitask_tasks) > 0:
            if not (self.args.target_is_code and (args._name.startswith("speech_to_unit") or args._name.startswith("doc_speech_to_unit"))):
                raise ValueError(
                    "set --criterion speech_to_unit for speech-to-unit loss with multitask"
                )
            elif not self.args.target_is_code and not args._name.startswith(
                "speech_to_spectrogram"
            ):
                raise ValueError(
                    "set --criterion speech_to_spectrogram for speech-to-spectrogram loss with multitask"
                )

        return criterions.build_criterion(args, self)
    
    def update_source_mask(self, source_mask, sorted_indices):
        seq_len = source_mask.shape[0]  # source_mask is 1D
        reduced_seq_len = sorted_indices.shape[0]
        
        # Calculate the scaling factor
        scaling_factor = seq_len / reduced_seq_len
        
        # Scale the sorted indices to match the source mask sequence length
        scaled_indices = (sorted_indices.float() * scaling_factor).long()
        
        # Clip the indices to ensure they are within the bounds of source_mask
        scaled_indices = torch.clamp(scaled_indices, 0, seq_len - 1)
        
        # Use indexing to get the values from the original mask
        new_mask = source_mask[scaled_indices]
        
        return new_mask