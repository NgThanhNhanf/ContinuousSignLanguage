import torch
import torch.nn as nn


class CTCHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            input_dim,
            num_classes
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, D)

        Returns:
            logits: (B, T, num_classes)
            log_probs: (B, T, num_classes)
        """

        x = self.dropout(x)

        logits = self.classifier(x)

        log_probs = logits.log_softmax(dim=-1)

        return logits, log_probs
