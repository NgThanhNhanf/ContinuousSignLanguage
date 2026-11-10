import torch
from torch import nn  

class SaliencyLoss(nn.Module):
    def __init__(self):
        super.__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        return self.loss(input)

class CompositeLoss(nn.Module):
    def __init__(self):
        super.__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.saliency_loss = SaliencyLoss()

    def forward(self, input):
        return self.ce_loss(input) + self.saliency_loss(input)