import torch
import torch.nn as nn
from torch import Tensor


class ResidualConnection(nn.Module):
    def __init__(self, module, module_factor, input_factor):
        super().__init__()
        self.module = module
        self.w1 = module_factor
        self.w2 = input_factor

    def forward(self, x):
        return (self.w1 * self.module(x)) + (self.w2 * x)
