import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.trainer_pt_utils import get_parameter_names

def add_params(model_list, is_include=True):
    """
    include_list or exclude_list be valid.
    """
    params_list = []
    for module in model_list:
        decay_parameters = get_parameter_names(module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        if is_include:
            params_list += [p for n, p in module.named_parameters() if n in decay_parameters]
        else:
            params_list += [p for n, p in module.named_parameters() if n not in decay_parameters]
    return params_list

def give_deberta_split_optimizer(model: nn.Module, config):
    if isinstance(model, DDP):
        model = model.module
    
    optim_args = []
    base_lr = config.lr
    head_lr = config.lr * config.lr_multi
    # fundational params
    optim_args.append({
        "params": add_params(model.base_model_list, is_include=True),
        "lr": base_lr,
        "weight_decay": config.wd
    })

    # fundational params (no wd)
    optim_args.append({
        "params": add_params(model.base_model_list, is_include=False),
        "lr": base_lr,
        "weight_decay": 0.0
    })

    # head params 
    optim_args.append({
        "params": add_params(model.new_model_list, is_include=True),
        "lr": head_lr,
        "weight_decay": config.wd
    })

    # head params (no wd)
    optim_args.append({
        "params": add_params(model.new_model_list, is_include=False),
        "lr": head_lr,
        "weight_decay": 0.0
    })

    optimizer = optim.AdamW(params=optim_args)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.sche_step, gamma=config.sche_decay)

    return optimizer, scheduler