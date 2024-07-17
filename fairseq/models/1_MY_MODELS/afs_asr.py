import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from torch import Tensor

def reshape_pyramidal(inputs, scale=2, mask=None):
        batch_size, max_time, num_units = inputs.size()
        
        if mask is not None:
            inputs = inputs * mask.unsqueeze(-1)
        
        num_pad = int(torch.ceil(torch.tensor(max_time / scale)) * scale) - max_time
        
        # Pad the time dimension
        inputs = F.pad(inputs, (0, 0, 0, num_pad))
        
        if mask is not None:
            mask = F.pad(mask, (0, num_pad))
        
        # Reshape
        concat_inputs = inputs.view(batch_size, -1, num_units * scale)
        
        if mask is not None:
            concat_mask = mask.view(batch_size, -1, scale)
            concat_mask = 1.0 - (concat_mask.sum(-1) < scale).float()
            return concat_inputs, concat_mask
        else:
            return concat_inputs

@register_model("my_asr_model")
class MyASRModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--hidden-size', type=int, metavar='N', default=512,
                            help='hidden size')
        parser.add_argument('--filter-size', type=int, metavar='N', default=2048,
                            help='filter size of FFN')
        parser.add_argument('--num-heads', type=int, metavar='N', default=8,
                            help='number of attention heads')
        parser.add_argument('--num-encoder-layers', type=int, metavar='N', default=6,
                            help='num encoder layers')
        parser.add_argument('--num-decoder-layers', type=int, metavar='N', default=6,
                            help='num decoder layers')
        parser.add_argument('--dropout', type=float, metavar='D', default=0.2,
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D', default=0.1,
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D', default=0.2,
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--label-smoothing', type=float, metavar='D', default=0.1,
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--max-len', type=int, metavar='N', default=2048,
                            help='maximum sequence length')
        parser.add_argument('--ctc-enable', action='store_true',
                            help='enable CTC loss')
        parser.add_argument('--ctc-alpha', type=float, metavar='A', default=0.3,
                            help='CTC loss weight')
        parser.add_argument('--speech-num-feature', type=int, metavar='N',
                            help='number of speech features')
        

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        encoder = MyEncoder(
            args,
            task.source_dictionary,
            speech_num_feature=args.speech_num_feature,
        )
        decoder = MyDecoder(
            args,
            task.target_dictionary,
        )
        return cls(encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out
    

class MyEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, speech_num_feature):
        super().__init__(dictionary)
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.speech_num_feature = speech_num_feature

        # Add a linear layer to map speech features to model dimension
        self.feature_projection = nn.Linear(speech_num_feature, args.hidden_size)

        self.embed_scale = math.sqrt(self.hidden_size)
        self.embed_positions = PositionalEmbedding(
            args.max_len, self.hidden_size, self.padding_idx,
            learned=args.sinusoid_posenc,
        )

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(args) for _ in range(args.num_encoder_layers)
        ])

        self.layer_norm = LayerNorm(self.hidden_size)
        

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        # src_tokens shape: [batch_size, max_time, speech_num_feature]
        x = self.feature_projection(src_tokens)
        
        # Create mask
        mask = ~torch.eq(src_tokens.sum(-1), 0)
        
        # Apply pyramidal reshaping
        x, mask = reshape_pyramidal(x, scale=3, mask=mask)
        
        x = self.embed_scale * x
        x += self.embed_positions(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # encoder layers
        for layer in self.layers:
            x = layer(x, self.padding_mask(mask))

        x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [~mask],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def padding_mask(self, mask):
        return ~mask

class MyDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        self.embed_tokens = nn.Embedding(
            len(dictionary), self.hidden_size, self.padding_idx
        )

        self.embed_scale = math.sqrt(self.hidden_size)
        self.embed_positions = PositionalEmbedding(
            args.max_len, self.hidden_size, self.padding_idx,
            learned=False,
        )

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(args) for _ in range(args.num_decoder_layers)
        ])

        self.layer_norm = LayerNorm(self.hidden_size)

        if self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.hidden_size, len(dictionary), bias=False
            )

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        positions = self.embed_positions(
            prev_output_tokens, incremental_state=incremental_state
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]

        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            x, _ = layer(
                x,
                encoder_out["encoder_out"][0],
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
            )

        x = self.layer_norm(x)

        return x, None

    def output_layer(self, features):
        return self.output_projection(features)

@register_model_architecture("my_asr_model", "my_asr_model_base")
def base_architecture(args):
    args.hidden_size = getattr(args, "hidden_size", 512)
    args.filter_size = getattr(args, "filter_size", 2048)
    args.num_heads = getattr(args, "num_heads", 8)
    args.num_encoder_layers = getattr(args, "num_encoder_layers", 6)
    args.num_decoder_layers = getattr(args, "num_decoder_layers", 6)
    args.dropout = getattr(args, "dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.relu_dropout = getattr(args, "relu_dropout", 0.2)
    args.label_smoothing = getattr(args, "label_smoothing", 0.1)
    args.max_len = getattr(args, "max_len", 2048)
    args.ctc_enable = getattr(args, "ctc_enable", True)
    args.ctc_alpha = getattr(args, "ctc_alpha", 0.3)
    args.speech_num_feature = getattr(args, "speech_num_feature", 120)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.speech_num_feature = getattr(args, "speech_num_feature", 80)  # or whatever your default is