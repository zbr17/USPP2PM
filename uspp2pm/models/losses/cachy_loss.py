import torch.nn as nn
import torch
import torch.nn.functional as F

class CachyLoss(nn.Module):
    is_regre = True
    def __init__(self):
        super().__init__()
    
    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        data = F.normalize(data, dim=-1)
        target = F.normalize(target, dim=-1)
        loss = - torch.sum(data * target)
        return loss
