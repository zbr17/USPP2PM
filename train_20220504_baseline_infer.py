#%%
import os
import sys
import socket

import torch
hostname = socket.gethostname()
if hostname != "zebra":
    is_kaggle = True
    sys.path.append("/kaggle/input/uspp2pm")
else:
    is_kaggle = False

#%%
import pandas as pd
import numpy as np
import argparse

from uspp2pm import logger
from uspp2pm.utils import compute_metrics, update_config
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
    loss_name = "mse" # mse / shift_mse
    # models
    model_config = {
        "deberta-v3-large": {"embed_dim": 512},
        "deberta-v3-base": {"embed_dim": 512}
    }
    pretrain_name = "deberta-v3-base"
    infer_name = "PREdeberta-v3-base-TAGbaseline-20220504"
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
    tag = "baseline"

parser = argparse.ArgumentParser("US patent model")
parser.add_argument("--evaluate", action="store_true")

opt = parser.parse_args(args=[])
opt.evaluate = True
config.is_kaggle = is_kaggle
config.is_training = not opt.evaluate
config.is_evaluation = opt.evaluate
config = update_config(config)

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
        test_set = give_dataset(test_data, False, tokenizer, config)

        # get model and criterion
        model = give_model(config).to(config.device)

        # load parameters
        params_dict = torch.load(os.path.join(config.model_path_infer, f"model_{fold}.ckpt"))
        model.load_state_dict(params_dict["model"])

        # Evaluate
        logger.info("Start to validate...")
        preds = predict(model, collate_fn, test_set, config, is_test=True)
        
        preds_all.append(preds)
    
    predictions = np.mean(preds_all, axis=0)
    submission = pd.DataFrame(data={
        'id': test_data['id'],
        'score': predictions,
    })
    submission.to_csv(os.path.join(config.save_path, 'submission.csv'), index=False)
    