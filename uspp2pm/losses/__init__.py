from torch.nn import MSELoss
# from .shift_mse import ShiftMSE

_loss_dict = {
    "deberta-v3-large": MSELoss,
    "deberta-v3-base": MSELoss
}

def give_criterion(config):
    _meta_loss = _loss_dict[config.pretrain_name]
    loss_func = _meta_loss()
    return loss_func