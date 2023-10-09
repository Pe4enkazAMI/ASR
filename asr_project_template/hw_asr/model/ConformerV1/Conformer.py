import torch.nn as nn
import torch
from .encoder import ConformerEncoder
from .decoder import ConformerDecoder


class Conformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = ConformerEncoder(input_dim=kwargs["input_dim"],
                                        d_encoder=kwargs["d_encoder"],
                                        num_layers=kwargs["num_layers"],
                                        num_heads_attention=kwargs["num_heads_attention"],
                                        ffl_exp_factor=kwargs["ffl_exp_factor"],
                                        ffl_dropout=kwargs["ffl_dropout"],
                                        conv_exp_factor=kwargs["conv_exp_factor"],
                                        conv_kernel_size=kwargs["conv_kernel_size"],
                                        conv_dropout=kwargs["conv_dropout"],
                                        attention_dropout=kwargs["attention_dropout"]
                                        )
        
        self.decoder = ConformerDecoder(d_encoder=kwargs["d_encoder"],
                                        num_classes=kwargs["num_classes"],
                                        hidden_size_decoder=kwargs["hidden_size_decoder"], num_layers=1)

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def transform_input_lengths(self, shape):
        return ((shape - 3)//4) 