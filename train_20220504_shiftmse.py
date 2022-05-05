#%%
import os
import sys
import socket
import datetime

import torch
hostname = socket.gethostname()
if hostname != "zebra":
    is_kaggle = True
    sys.path.append("/kaggle/input")
else:
    is_kaggle = False

#%%
import pandas as pd
import numpy as np
import argparse

from uspp2pm import logger
from uspp2pm.utils import compute_metrics
from uspp2pm.datasets import give_rawdata, give_collate_fn, give_dataset
from uspp2pm.models import give_tokenizer, give_model
from uspp2pm.optimizers import give_optim
from uspp2pm.losses import give_criterion
from uspp2pm.engine import train_one_epoch, predict

#%%
class config:
    device = torch.device("cuda:0")
    dataset_name = "combined"
    # losses
    loss_name = "shift_mse" # mse / shift_mse
    # models
    pretrain_name = "deberta-v3-base"
    infer_name = "test"
    # training
    lr = 2e-5
    wd = 0.01
    num_fold = 5
    epochs = 10
    bs = 64
    num_workers = 12
    lr_multi = 10
    sche_step = 5
    sche_decay = 0.5
    # log
    tag = f"{dataset_name}-{loss_name}"

parser = argparse.ArgumentParser("US patent model")
parser.add_argument("--evaluate", action="store_true")

opt = parser.parse_args(args=[])
opt.evaluate = False
config.is_kaggle = is_kaggle
config.is_training = not opt.evaluate
config.is_evaluation = opt.evaluate

#%%
def update_config(config):
    # dataset
    config.input_path = (
        "./data/uspp2pm" if not config.is_kaggle
        else "/kaggle/input/us-patent-phrase-to-phrase-matching"
    )
    config.title_path = (
        "./data/cpcs/titles.csv" if not config.is_kaggle
        else "/kaggle/input/cpccode/titles.csv"
    )
    config.train_data_path = os.path.join(config.input_path, "train.csv")
    config.test_data_path = os.path.join(config.input_path, "test.csv")
    # model
    config.model_path = (
        f"./pretrains/{config.pretrain_name}" if not config.is_kaggle
        else f"/kaggle/input/{config.pretrain_name}"
    )
    config.model_path_infer = (
        f"./out/{config.infer_name}/" if not config.is_kaggle
        else f"/kaggle/input/{config.infer_name}"
    )
    # log
    config.save_name = f"{datetime.datetime.now().strftime('%Y%m%d')}-{config.tag}"
    config.save_path = (
        f"./out/{config.save_name}/" if not config.is_kaggle
        else f"/kaggle/working/"
    )
    return config
config = update_config(config)

#%%
def run(index, train_data, val_data, tokenizer, collate_fn, is_val, config):
    train_set = give_dataset(train_data, True, tokenizer, config)
    val_set = give_dataset(val_data, True, tokenizer, config) if is_val else None

    # get model and criterion
    model = give_model(config).to(config.device)
    criterion = give_criterion(config).to(config.device)

    # get optimizer and scheduler
    optimizer, scheduler = give_optim(model, config)

    for epoch in range(config.epochs):
        logger.info(f"Epoch: {epoch}")
        # Start to train
        logger.info("Start to train...")
        preds, labels = train_one_epoch(model, criterion, collate_fn, train_set, optimizer, scheduler, config)
        sub_acc = compute_metrics((preds, labels))
        logger.info(f"TrainSET - Fold: {index}, Epoch: {epoch}, Acc: {sub_acc}")
        to_save_dict = {
            "model": model.state_dict(),
            "epoch": epoch
        }
        torch.save(to_save_dict, os.path.join(config.save_path, f"model_{index}.ckpt"))

        if is_val:
            # Validate
            logger.info("Start to validate...")
            preds, labels = predict(model, collate_fn, val_set, config)
            sub_acc = compute_metrics((preds, labels))
            logger.info(f"ValSET - Fold: {index}, Epoch: {epoch}, Acc: {sub_acc}")
    
    if is_val:
        return preds, labels
    else:
        return None, None

#%%
# initiate logger
logger.config_logger(output_dir=config.save_path)

# get dataset
train_data, test_data = give_rawdata("train", config), give_rawdata("test", config)
tokenizer = give_tokenizer(config)
collate_fn = give_collate_fn(tokenizer, config)


# training phase
if config.is_training:
    
    if config.num_fold < 2:
        run("all", train_data, None, tokenizer, collate_fn, False, config)
    else:
        preds_all, labels_all = [], []
        for fold in range(config.num_fold):
            sub_train_data = train_data[train_data["fold"] != fold].reset_index(drop=True)
            sub_val_data = train_data[train_data["fold"] == fold].reset_index(drop=True)

            preds, labels = run(fold, sub_train_data, sub_val_data, tokenizer, collate_fn, True, config)
            preds_all.append(preds)
            labels_all.append(labels)
        
        preds_all = np.concatenate(preds_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        final_acc = compute_metrics((preds_all, labels_all))
        logger.info(f"Final acc: {final_acc}")
