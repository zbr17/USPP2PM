from multiprocessing import reduction
from torch.nn import MSELoss
from .shift_mse import ShiftMSE
from .pearson_corr import PearsonCorr
from .match_loss import MatchLoss
from .exp_loss import ExpLoss

_loss_dict = {
    "mse": MSELoss,
    "pearson": PearsonCorr,
    "shift_mse": ShiftMSE,
    "match": MatchLoss,
    "exp": ExpLoss
}

def give_criterion(config):
    _meta_loss = _loss_dict[config.loss_name]
    if config.loss_name in ["exp"]:
        loss_func = _meta_loss(config)
    if config.loss_name in ["mse"]:
        loss_func = _meta_loss(reduction="none")
    else:
        loss_func = _meta_loss()
    return loss_func