import torch.nn as nn
import torch

class ShiftMSE(nn.Module):
    def __init__(self, scale=0.8, bias=0.1):
        super().__init__()
        self.loss_func = nn.MSELoss()
        self.scale = scale
        self.bias = bias
    
    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.loss_func(data, target)
        return loss

# class ShiftMSE(nn.Module):
#     def __init__(self, scale=0.8, bias=0.1):
#         super().__init__()
#         self.loss_func = nn.MSELoss(reduction="none")
#         self.scale = scale
#         self.bias = bias
    
#     def compute_mse(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         bs = data.size(0)
#         shift_target = self.scale * target + self.bias

#         # id_x, id_y = torch.meshgrid(torch.arange(bs), torch.arange(bs))
#         # id_x, id_y = id_x.flatten(), id_y.flatten()
#         # data = torch.log(data[id_x] + 1e-8) - torch.log(data[id_y] + 1e-8)
#         # shift_target = torch.log(shift_target[id_x] + 1e-8) - torch.log(shift_target[id_y] + 1e-8)

#         loss = self.loss_func(data, shift_target)
#         loss = torch.pow(loss, 2).sum()
#         return loss
    
#     def forward(self, data_tuple, target):
#         sim, sim_inter = data_tuple
#         loss_sim = self.compute_mse(sim, target)
#         loss_sim_inter = self.compute_mse(sim_inter, target)
#         return loss_sim, loss_sim_inter