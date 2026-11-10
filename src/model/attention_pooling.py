import torch
from torch import nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

    def forward(self, x): # S x D
        attn_weights = self.attention(x) # S x 1
        attn_weights = F.softmax(attn_weights, dim = 1)
        pooled = torch.sum(attn_weights * x, dim = 1) # S x 1 * S x D
        return pooled