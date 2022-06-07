from torch.nn import MSELoss
from .pearson_corr import PearsonCorr
from .match_loss import MatchLoss
from .exp_loss import ExpLoss
from .cross_entropy import CrossEntropy
from .cachy_loss import CachyLoss

import torch.nn as nn
import torch

_loss_dict = {
    "mse": MSELoss,
    "pearson": PearsonCorr,
    "match": MatchLoss,
    "exp": ExpLoss,
    "cross_entropy": CrossEntropy,
    "cachy": CachyLoss,
}

class MultiLoss(nn.Module):
    is_regre = True
    def __init__(self, config):
        super().__init__()
        self.loss_list = []
        for loss_name in config.loss_name.split("+"):
            self.loss_list.append(give_criterion(config, loss_name))
    
    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0
        for loss_func in self.loss_list:
            loss += loss_func(data, target).mean()
        return loss

def give_criterion(config, loss_name=None):
    if loss_name is None:
        loss_name = config.loss_name
    if "+" in loss_name:
        loss_func = MultiLoss(config)
        return loss_func
    
    _meta_loss = _loss_dict[loss_name]
    if loss_name in ["exp"]:
        loss_func = _meta_loss(config)
    if loss_name in ["mse"]:
        loss_func = _meta_loss(reduction="none")
    else:
        loss_func = _meta_loss()
    return loss_func