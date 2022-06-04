import torch.nn as nn
import torch

class ShiftMSE(nn.Module):
    def __init__(self, scale=0.8, bias=0.1):
        super().__init__()
        self.loss_func = nn.MSELoss()
        self.scale = scale
        self.bias = bias
    
    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        shift_target = self.scale * target + self.bias
        loss = self.loss_func(data, shift_target)
        return loss
