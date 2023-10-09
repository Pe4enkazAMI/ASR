import torch.nn as nn 
from torch import Tensor
import torch.nn.functional as F
import torch


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0, "d_model should be divisible by num_heads"
        self.attn_dim = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.p = dropout

        self.Q = nn.Linear(self.d_model, self.attn_dim) 
        self.K = nn.Linear(self.d_model, self.attn_dim)
        self.V = nn.Linear(self.d_model, self.attn_dim)
        self.dropout = nn.Dropout(self.p)


    def forward(self, x):
        Q, K, V = self.Q(x), self.K(x), self.V(x)
        attention_scores = F.softmax(torch.bmm(Q, torch.transpose(K, 1, 2)), dim=-1) / torch.sqrt(torch.Tensor([self.d_model]))
        attention = torch.bmm(self.dropout(attention_scores), V)
        return attention, attention_scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_dim = d_model // num_heads
        self.p = dropout
        self.out = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(self.p)
        self.attention_heads = nn.ModuleList([Attention(self.d_model, self.num_heads, self.p)
                                              for i in range(num_heads)])

    def forward(self, x):
        attn_scores = []
        attn_probas = []
        for head in self.attention_heads:
            attn, attn_probs = head(x)
            attn_scores += [attn]
            attn_probas += [attn_probs]

        attn = torch.cat(attn_scores, dim=-1)
        attn_probs = torch.stack(attn_probas, dim=-1)
        attn = self.dropout(self.out(attn))

        return attn, attn_probs

class ConformerAttentionBlock(nn.Module):
    def __init__(self, d_model, num_head, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_head, dropout)
        self.p = dropout
        self.dropout = nn.Dropout(self.p)

    def forward(self, x):
        x = self.layer_norm(x)
        x, probs = self.mha(x)
        x = self.dropout(x)
        return x
