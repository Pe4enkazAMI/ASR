import torch.nn as nn 
from torch import Tensor 
from .activations import Swish, GLU


class DepthwiseConv1d(nn.Module):
    def __init__(self, in_chanels, out_chanels, kernel_size, padding, stride):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_chanels,
            out_channels=out_chanels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_chanels,
            bias=False
        )

    def forward(self, x):
        return self.conv(x)

class PointwiseConv1d(nn.Module):
    def __init__(self, in_chanels, out_chanels, padding, stride):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_chanels,
            out_channels=out_chanels,
            kernel_size=1,
            padding=padding,
            stride=stride,
            bias=True,
        )
    def forward(self, x):
        return self.conv(x)

class ConvSubsample(nn.Module):
    def __init__(self, in_chanels, out_chanels):
        super().__init__()
        self.subsampler = nn.Sequential(
            nn.Conv1d(in_channels=in_chanels, out_channels=out_chanels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_chanels, out_channels=out_chanels, kernel_size=3, stride=2),
            nn.ReLU()
        )
    
    def forward(self, x, xshape):
        x = self.subsampler(x)
        return x.permute(0, 2, 1)

class Printer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        print(x.shape)
        return x

class ConformerConvBlock(nn.Module):
    def __init__(self, in_chanels, exp_factor, kernel_size, padding, stride, dropout):
        super().__init__()
        self.p = dropout
        self.norm = nn.LayerNorm(in_chanels)
        self.block = nn.Sequential(
            PointwiseConv1d(in_chanels, exp_factor*in_chanels, 0, 1),
            GLU(dim=1),
            DepthwiseConv1d(in_chanels, in_chanels, kernel_size, padding, stride),
            nn.BatchNorm1d(in_chanels),
            Swish(),
            PointwiseConv1d(in_chanels, in_chanels, 0, 1),
            nn.Dropout(p=self.p)
        )
        
    def forward(self, x):
        return self.block(self.norm(x).transpose(1,2)).transpose(1,2)
        