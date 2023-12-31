import torch.nn as nn
import torch
from .encoder import ConformerEncoder
from .decoder import ConformerDecoder
import numpy as np


class Conformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = ConformerEncoder(input_dim=kwargs["input_dim"],
                                        d_encoder=kwargs["d_encoder"],
                                        num_layers=kwargs["num_encoder_layers"],
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
                                        hidden_size_decoder=kwargs["hidden_size_decoder"], num_layers=kwargs["num_decoder_layers"])

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def transform_input_lengths(self, shape):
        return ((shape - 3)//4) 
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)