from .shift_mse import ShiftMSE

_loss_dict = {
    "deberta-v3-large": ShiftMSE
}

def give_criterion(pretrain_name: str):
    loss_func = _loss_dict[pretrain_name]()
    return loss_func