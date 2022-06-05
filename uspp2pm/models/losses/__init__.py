from torch.nn import MSELoss
from .pearson_corr import PearsonCorr
from .match_loss import MatchLoss
from .exp_loss import ExpLoss
from .cross_entropy import CrossEntropy
from .cachy_loss import CachyLoss

_loss_dict = {
    "mse": MSELoss,
    "pearson": PearsonCorr,
    "match": MatchLoss,
    "exp": ExpLoss,
    "cross_entropy": CrossEntropy,
    "cachy": CachyLoss,
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