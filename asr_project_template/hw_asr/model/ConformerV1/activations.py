import torch.nn as nn
from torch import Tensor

class GLU(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        self.act = nn.GLU(dim=dim)
    def forward(self, x):
        return self.act(x)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(x)
    
