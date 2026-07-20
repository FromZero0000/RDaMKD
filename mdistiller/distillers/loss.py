import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy with label smoothing for legacy distiller imports."""

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        one_hot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1
        )
        smoothed = (
            (1 - self.epsilon) * one_hot
            + self.epsilon / self.num_classes
        )
        return (-smoothed * log_probs).mean(0).sum()
