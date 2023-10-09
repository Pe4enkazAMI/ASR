import torch.nn as nn
import torch
from .encoder import ConformerEncoder


class ConformerDecoder(nn.Module):
    def __init__(self, d_encoder, num_classes, num_layers, hidden_size_decoder):
        super().__init__()
        self.lstm = nn.LSTM(batch_first=True, input_size=d_encoder, hidden_size=hidden_size_decoder, num_layers=1)
        self.logits = nn.Linear(hidden_size_decoder, num_classes)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)

        return self.logits(output)