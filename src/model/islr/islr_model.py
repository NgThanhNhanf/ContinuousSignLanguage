import torch
from torch import nn
import torchvision 
from torchvision import models

from src.model.attention_pooling import AttentionPooling
from src.model.positional_embedding import PositionalEmbedding

class CNNTransformer(nn.Module):
    def __init__(self, n_frames, num_classes, weights):
        super().__init__()
        cnnn = models.resnet18(weights=weights)
        self.cnn = nn.Sequential(*list(cnnn.children())[:-2])
        self.feature_dim = 512
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.positional_embeding = PositionalEmbedding(self.feature_dim, n_frames, dropout=0.1)
        self.attention_pooling = AttentionPooling(self.feature_dim)
        layer = nn.TransformerEncoderLayer(
            d_model = self.feature_dim,
            nhead = 8,
            dim_feedforward = self.feature_dim * 4,
            dropout = 0.3,
            activation = 'gelu',
            batch_first = True,
            norm_first = True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers = 2)
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        out = self.cnn(x)
        out = self.pool(out)
        out = out.view(B, T, -1)
        out = self.positional_embeding(out)
        out = self.transformer(out)
        out = self.attention_pooling(out)
        out = self.fc(out)
        return out