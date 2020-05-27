import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1: torch.tensor, output2: torch.tensor, target: torch.tensor):
        distances = (output1 - output2).pow(2).sum(dim=1)
        loss = 0.5 * target.float() * distances + \
               0.5 * (1. - target.float()) * F.relu(self.margin - distances)
        return loss.mean()