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
    loss_name = "pearson" # mse / shift_mse / pearson
    # models
    pretrain_name = "deberta-v3-large" # bert-for-patents / deberta-v3-large
    infer_name = "20220505-PREdeberta-v3-large-DATcombined-LOSSpearson-FOLD1"
    # training
    lr = 2e-5
    wd = 0.01
    num_fold = 1 # 0/1 for training all
    epochs = 10
    bs = 64
    num_workers = 12
    lr_multi = 10
    sche_step = 5
    sche_decay = 0.5
    # log
    tag = f"PRE{pretrain_name[:5]}-DAT{dataset_name[:5]}-LOSS{loss_name[:5]}-FOLD{num_fold[:5]}"

parser = argparse.ArgumentParser("US patent model")
parser.add_argument("--evaluate", action="store_true")

opt = parser.parse_args(args=[])
opt.evaluate = True
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
def run_test(index, test_data, tokenizer, collate_fn, config):
    test_set = give_dataset(test_data, False, tokenizer, config)

    # get model and criterion
    model = give_model(config).to(config.device)

    # load parameters
    params_dict = torch.load(os.path.join(config.model_path_infer, f"model_{index}.ckpt"))
    model.load_state_dict(params_dict["model"])

    # Evaluate
    logger.info("Start to validate...")
    preds = predict(model, collate_fn, test_set, config, is_test=True)

    return preds

#%%
# initiate logger
logger.config_logger(output_dir=config.save_path)

# get dataset
train_data, test_data = give_rawdata("train", config), give_rawdata("test", config)
tokenizer = give_tokenizer(config)
collate_fn = give_collate_fn(tokenizer, config)

# training phase
if config.is_evaluation:
    preds_all = []
    for fold in range(config.num_fold):
        if config.num_fold < 2:
            preds = run_test("all", test_data, tokenizer, collate_fn, config)
        else:
            preds = run_test(fold, test_data, tokenizer, collate_fn, config)
        preds_all.append(preds)
    
    predictions = np.mean(preds_all, axis=0)
    submission = pd.DataFrame(data={
        'id': test_data['id'],
        'score': predictions,
    })
    submission.to_csv(os.path.join(config.save_path, 'submission.csv'), index=False)
    