import torch
import torch.nn as nn
from .prenet import PreNet
from .postnet import PostNet

class LargeConceptModel(nn.Module):
    def __init__(self, input_dim, model_dim, output_dim, num_layers, num_heads, scaler):
        super().__init__()
        self.prenet = PreNet(input_dim, model_dim, scaler)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(model_dim, num_heads),
            num_layers
        )
        self.postnet = PostNet(model_dim, output_dim, scaler)
        self.model_dim = model_dim
        self.embedding_scale = model_dim ** 0.5

    def forward(self, src, memory=None, tgt_mask=None):
        src_emb = self.prenet(src) * self.embedding_scale
        src_emb = src_emb.permute(1, 0, 2)

        if memory is not None:
            memory_emb = self.prenet(memory) * self.embedding_scale
            memory_emb = memory_emb.permute(1, 0, 2)
        else:
            memory_emb = None

        tgt = torch.zeros_like(src_emb)
        decoder_output = self.transformer_decoder(
            tgt=tgt,
            memory=memory_emb if memory_emb is not None else src_emb,
            tgt_mask=tgt_mask
        )

        decoder_output = decoder_output.permute(1, 0, 2)
        return self.postnet(decoder_output)