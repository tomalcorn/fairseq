import logging
import csv
import torch
import torch.nn as nn
from argparse import Namespace
from fairseq.models.speech_to_text.s2t_transformer import AdaptiveFeatureSelection, S2TTransformerEncoder
from fairseq import checkpoint_utils

logger = logging.getLogger(__name__)

class AfsFeatureExtractor(nn.Module):
    
    def __init__(self, pretraining_path, args_tsv, default_args) -> None:
        super().__init__()

        self.pretraining_path = pretraining_path
        self.default_args = default_args
        self.args = self.read_in_args_tsv(args_tsv)
        self.dim_reduction = nn.Linear(self.args.encoder_embed_dim, self.default_args.encoder_embed_dim)
        
        
        if self.pretraining_path is not None:
            encoder = S2TTransformerEncoder(self.args)
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
            x, out_seq_lens_tensor = self.afs.sparsify_inputs(encoder_out)
            
        # To go from feature size 512 to 256
        x = self.dim_reduction(x)
        return x, out_seq_lens_tensor
    
    