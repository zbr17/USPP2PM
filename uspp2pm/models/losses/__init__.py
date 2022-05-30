from torch.nn import MSELoss
from .shift_mse import ShiftMSE
from .pearson_corr import PearsonCorr
from .match_loss import MatchLoss

_loss_dict = {
    "mse": MSELoss,
    "pearson": PearsonCorr,
    "shift_mse": ShiftMSE,
    "match": MatchLoss
}

def give_criterion(config):
    _meta_loss = _loss_dict[config.loss_name]
    loss_func = _meta_loss()
    return loss_func