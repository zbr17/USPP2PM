import torch.nn as nn
import torch

class MatchLoss(nn.Module):
    is_regre = True
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def compute_mat(self, data: torch.Tensor):
        data = torch.log(data + 1e-8)
        mat = data.unsqueeze(0) - data.unsqueeze(1)
        return mat
    
    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        term1 = self.compute_mat(data)
        term2 = self.compute_mat(target)
        loss = (term1 - term2).pow(2).mean()
        loss = loss + self.criterion(data, target)
        return loss 
