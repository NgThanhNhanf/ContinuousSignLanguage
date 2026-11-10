
import torch
from torch import nn
import math

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, n_frames, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(
            n_frames,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(n_frames, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        T = x.size(1)

        if T > self.pe.size(1):
            raise ValueError(
                f"Input length {T} exceeds "
                f"max positional length {self.pe.size(1)}"
            )

        x = x + self.pe[:, :T, :]

        return self.dropout(x)
