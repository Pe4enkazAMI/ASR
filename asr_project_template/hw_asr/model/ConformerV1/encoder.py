import torch.nn as nn 
from torch import Tensor
import torch.nn.functional as F
from .activations import Swish
from .attention import *
from .feed_forward import ConformerFeedForwardLayer
from .activations import GLU 
from .conv import ConformerConvBlock, ConvSubsample
from .resconnect import ResidualConnection

class ConformerBlock(nn.Module):
    def __init__(self,
        d_encoder,
        num_heads_attention,
        ffl_exp_factor,
        conv_exp_factor,
        ffl_dropout,
        attention_dropout,
        conv_dropout,
        conv_kernel_size):
        super().__init__()
    
        self.block = nn.Sequential(
            ResidualConnection(
                module=ConformerFeedForwardLayer(
                    d_model=d_encoder,
                    exp_factor=ffl_exp_factor,
                    dropout=ffl_dropout
                ),
                module_factor=0.5,
                input_factor=1
            ),
            ResidualConnection(
                module=ConformerAttentionBlock(
                    d_model=d_encoder,
                    num_head=num_heads_attention,
                    dropout=attention_dropout
                ),
                module_factor=1,
                input_factor=1
            ),
            ResidualConnection(
                module=ConformerConvBlock(
                    in_chanels=d_encoder,
                    exp_factor=conv_exp_factor,
                    kernel_size=conv_kernel_size,
                    stride=1,
                    padding=15,
                    dropout=conv_dropout
                ),
                module_factor=1,
                input_factor=1
            ),
            ResidualConnection(
                module=ConformerFeedForwardLayer(
                    d_model=d_encoder,
                    exp_factor=ffl_exp_factor,
                    dropout=ffl_dropout
                ),
                module_factor=0.5,
                input_factor=1
            ),
            nn.LayerNorm(d_encoder)
        )
    def forward(self, x):
        return self.block(x)

class ConformerEncoder(nn.Module):
    def __init__(self, 
                input_dim,
                d_encoder, 
                num_layers, 
                num_heads_attention,
                ffl_exp_factor,
                conv_exp_factor,
                conv_kernel_size,
                ffl_dropout,
                conv_dropout,
                attention_dropout):
        super().__init__()
        self.subsample = ConvSubsample(out_chanels=d_encoder)

        in_feat = d_encoder * (((input_dim - 1)//2) - 1)//2
        self.linear_projection = nn.Sequential(
            nn.Linear(in_feat, d_encoder),
            nn.Dropout(0.1)
        )
        self.layers = nn.ModuleList([ConformerBlock(d_encoder,
                                                    num_heads_attention,
                                                    ffl_exp_factor,
                                                    conv_exp_factor,
                                                    ffl_dropout,
                                                    attention_dropout,
                                                    conv_dropout,
                                                    conv_kernel_size) for _ in range(num_layers)])

    def _count_parameters(self):
        return sum(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        x = self.subsample(x.transpose(1,2))
        x = self.linear_projection(x)
        for layer in self.layers:
            x = layer(x)
        return x

