import torch.nn as nn
import torch

class PearsonCorr(nn.Module):
    is_regre = True
    def __init__(self):
        super().__init__()
    
    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        v_data = data - torch.mean(data, dim=-1, keepdim=True)
        v_target = target - torch.mean(target, dim=-1, keepdim=True)

        nomenator = - torch.sum(v_data * v_target, dim=-1)
        denominator = (
            torch.sqrt(torch.sum(v_data ** 2, dim=-1) + 1e-8) *
            torch.sqrt(torch.sum(v_target ** 2, dim=-1) + 1e-8)
        )
        loss = nomenator / denominator
        return loss
