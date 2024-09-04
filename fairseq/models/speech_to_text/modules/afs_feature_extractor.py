import logging
import csv
import torch
import os
import torch.nn as nn
import pandas as pd
from argparse import Namespace
from fairseq.models.speech_to_text.s2t_transformer import AdaptiveFeatureSelection, S2TTransformerEncoder
from fairseq import checkpoint_utils

logger = logging.getLogger(__name__)

class AfsFeatureExtractor(nn.Module):
    
    def __init__(self, pretraining_path, args_tsv, default_args, l0_mask_dir=None) -> None:
        super().__init__()

        self.pretraining_path = pretraining_path
        self.default_args = default_args if type(default_args) == Namespace else self.read_default_args_tsv(default_args)
        self.args = self.read_in_args_tsv(args_tsv)
        self.dim_reduction = nn.Linear(self.args.encoder_embed_dim, self.default_args.encoder_embed_dim)
        self.l0_mask_dir = l0_mask_dir
        self.l0_mask = None
        self.need_attn = default_args.attn_save_dir
        self.sorted_indices = None
        
        
        if self.pretraining_path is not None:
            encoder = S2TTransformerEncoder(self.args)
            encoder.training = False
            afs = AdaptiveFeatureSelection(input_dim=self.args.decoder_embed_dim,
            dropout=self.args.dropout,
            enable_afs_t=self.args.enable_afs_t,
            enable_afs_f=self.args.enable_afs_f)
            self.encoder = checkpoint_utils.load_pretrained_component_from_model(
                        component=encoder, checkpoint=pretraining_path, strict=False
                    )
            logger.info(f"loaded feature extractor - encoder from: {pretraining_path}")
            self.afs = checkpoint_utils.load_pretrained_component_from_model(
                        component=afs, checkpoint=pretraining_path, strict=False
                    )
            logger.info(f"loaded feature extractor - afs from: {pretraining_path}")
            
    
    def read_default_args_tsv(self, default_args_tsv):
        args_dict = {}
    
        with open(default_args_tsv, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 1:
                    key = parts[0]
                    value = "True"
                elif len(parts) == 2:
                    key, value = parts
                else:
                    raise ValueError("Each line must contain one or two tab-separated values.")
                args_dict[key] = value

        # Convert string 'True' or 'False' to boolean if present
        for key, value in args_dict.items():
            if isinstance(value, str) and value.lower() == 'true':
                args_dict[key] = True
            elif isinstance(value, str) and value.lower() == 'false':
                args_dict[key] = False
            elif isinstance(value, str) and value.lower() == "none":
                args_dict[key] = None
            elif value[0].isnumeric():
                try:
                    args_dict[key] = int(value)
                except ValueError:
                    try:
                        args_dict[key] = float(value)
                    except ValueError:
                        pass

        # Create a Namespace object from the dictionary
        args_namespace = Namespace(**args_dict)
        
        return args_namespace
    
    def read_in_args_tsv(self, args_tsv):
        S2TTransformerEncoder(self.default_args)
        
        args = {}
        with open(args_tsv, 'r') as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                key, value = row
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            # Keep as string if not bool, int, or float
                            pass
                
                args[key] = value
                
        # Create a new Namespace with default values
        final_args = Namespace(**vars(self.default_args))

        # Update with the values from the TSV file
        for key, value in args.items():
            setattr(final_args, key, value)
        
        return final_args
    
    def forward(self, src_tokens, src_lengths):
        with torch.no_grad():
            encoder_out = self.encoder(src_tokens, src_lengths)
            x, out_seq_lens_tensor, l0_mask, sorted_indices = self.afs.sparsify_inputs(encoder_out)
            
            if self.need_attn:
                self.sorted_indices = sorted_indices
                self.need_attn = False
            
            # To go from feature size 512 to 256
            x = self.dim_reduction(x)

        
        if self.l0_mask_dir is not None:
            os.makedirs(self.l0_mask_dir, exist_ok=True)
            self.l0_mask = l0_mask
            
            
        return x, out_seq_lens_tensor
    
    