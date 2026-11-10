import torch
from torch import nn
import torchvision 
from torchvision import models

from src.model.attention_pooling import AttentionPooling
from src.model.positional_embedding import PositionalEmbedding
from src.model.cslr.ctc_head import CTCHead

class CNNTransformer(nn.Module):
    def __init__(self, n_frames, num_classes, weights=None):
        super().__init__()

        cnn = models.resnet18(weights=weights)

        self.cnn = nn.Sequential(
            *list(cnn.children())[:-2]
        )

        self.feature_dim = 512

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.positional_embedding = PositionalEmbedding(
            self.feature_dim,
            n_frames,
            dropout=0.1
        )

        layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=8,
            dim_feedforward=self.feature_dim * 4,
            dropout=0.3,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=2
        )

        self.norm = nn.LayerNorm(self.feature_dim)

        self.ctc_head = CTCHead(
            self.feature_dim,
            num_classes
        )

    def forward(self, x, src_key_padding_mask=0.0):

        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)

        x = self.cnn(x)

        x = self.pool(x)

        x = x.flatten(1)

        x = x.reshape(B, T, -1)

        x = self.positional_embedding(x)

        x = self.transformer(
            x,
            src_key_padding_mask=src_key_padding_mask
        )

        x = self.norm(x)

        x = self.ctc_head(x)

        return x
