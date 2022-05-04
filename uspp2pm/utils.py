import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from transformers.trainer_pt_utils import get_parameter_names

class LogMeter:
    def __init__(self):
        self.value_list = []
    
    def append(self, v):
        self.value_list.append(v)
    
    def avg(self):
        return np.mean(self.value_list)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.reshape(len(predictions))
    return {
        "pearson": np.corrcoef(predictions, labels)[0][1]
    }

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

def give_optimizer_v0(pretrain_name: str, model: nn.Module, optim_config: dict):
    if pretrain_name == "deberta-v3-large":
        optim_args = []
        base_lr = optim_config["lr"]
        multi_lr = optim_config["lr"] * optim_config["lr_multi"]
        # fundational params
        optim_args.append({
            "params": add_params(model.base_model_list, is_include=True),
            "lr": base_lr,
            "weight_decay": optim_config["wd"]
        })

        # fundational params (no wd)
        optim_args.append({
            "params": add_params(model.base_model_list, is_include=False),
            "lr": base_lr,
            "weight_decay": 0.0
        })

        # initialized params
        optim_args.append({
            "params": add_params(model.new_model_list, is_include=True),
            "lr": multi_lr,
            "weight_decay": optim_config["wd"]
        })

        # initialized params (no wd)
        optim_args.append({
            "params": add_params(model.new_model_list, is_include=False),
            "lr": multi_lr,
            "weight_decay": 0.0
        })

        optimizer = optim.AdamW(params=optim_args)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=optim_config["sche_step"], gamma=optim_config["sche_decay"])
    
    return optimizer, scheduler

def give_optimizer(pretrain_name: str, model: nn.Module, optim_config: dict):
    if pretrain_name == "deberta-v3-large":
        optim_args = []
        base_lr = optim_config["lr"]
        # fundational params
        optim_args.append({
            "params": add_params([model], is_include=True),
            "lr": base_lr,
            "weight_decay": optim_config["wd"]
        })

        # fundational params (no wd)
        optim_args.append({
            "params": add_params([model], is_include=False),
            "lr": base_lr,
            "weight_decay": 0.0
        })

        optimizer = optim.AdamW(params=optim_args)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=optim_config["sche_step"], gamma=optim_config["sche_decay"])
    
    return optimizer, scheduler