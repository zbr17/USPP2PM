import numpy as np
import datetime
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from transformers.trainer_pt_utils import get_parameter_names

def update_config(config):
    # dataset
    config.input_path = (
        "./data/uspp2pm" if not config.is_kaggle
        else "/kaggle/input/us-patent-phrase-to-phrase-matching"
    )
    config.title_path = (
        "./data/cpcs/titles.csv" if not config.is_kaggle
        else "/kaggle/input/uspp2pm/data/cpcs/titles.csv"
    )
    config.train_data_path = os.path.join(config.input_path, "train.csv")
    config.test_data_path = os.path.join(config.input_path, "test.csv")
    # model
    config.model_path_train = (
        f"./pretrains/{config.pretrain_name}" if not config.is_kaggle
        else f"/kaggle/input/uspp2pm/pretrains/{config.pretrain_name}"
    )
    config.model_path_infer = (
        f"./out/{config.infer_name}/" if not config.is_kaggle
        else f"/kaggle/input/uspp2pm/out/{config.infer_name}"
    )
    config.model_path = config.model_path_train if config.is_training else config.model_path_infer
    # log
    config.save_name = f"PRE{config.pretrain_name}-TAG{config.tag}-{datetime.datetime.now().strftime('%Y%m%d')}"
    config.save_path = (
        f"./out/{config.save_name}/" if not config.is_kaggle
        else f"/kaggle/working/"
    )
    return config

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