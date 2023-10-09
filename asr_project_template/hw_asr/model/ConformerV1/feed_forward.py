import torch.nn as nn 
from torch import Tensor
import torch.nn.functional as F
from .activations import Swish

class ConformerFeedForwardLayer(nn.Module):
    def __init__(self, d_model, exp_factor, dropout):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model*exp_factor)
        self.silu = Swish()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * exp_factor, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.linear1(x)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.layer_norm2(x)
        return x