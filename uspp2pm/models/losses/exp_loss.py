import torch.nn as nn
import torch

class ExpLoss(nn.Module):
    is_regre = True
    def __init__(self, config):
        super().__init__()
        self.scale = config.scale
    
    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.exp( torch.abs(data - target) * self.scale ).mean()
        raise NotImplementedError()
        return loss 
