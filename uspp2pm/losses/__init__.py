from torch.nn import MSELoss
from .shift_mse import ShiftMSE

_loss_dict = {
    "mse": MSELoss,
    "shift_mse": ShiftMSE
}

def give_criterion(config):
    _meta_loss = _loss_dict[config.loss_name]
    loss_func = _meta_loss()
    return loss_func