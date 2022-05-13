#%%
import os
import sys
import socket
import datetime
from matplotlib import backends

hostname = socket.gethostname()
if hostname != "zebra":
    is_kaggle = True
    sys.path.append("/kaggle/input")
else:
    is_kaggle = False

import pandas as pd
import numpy as np
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from uspp2pm import logger
from uspp2pm.utils import compute_metrics
from uspp2pm.datasets import give_rawdata, give_collate_fn, give_dataset
from uspp2pm.models import give_tokenizer, give_model
from uspp2pm.optimizers import give_optim
from uspp2pm.losses import give_criterion
from uspp2pm.engine import train_one_epoch, predict

#%%
class CONFIG:
    nproc_per_node = 1
    is_distributed = nproc_per_node > 1
    dataset_name = "combined"
    # losses
    loss_name = None # mse / shift_mse / pearson
    # models
    pretrain_name = None # bert-for-patents / deberta-v3-large
    infer_name = "test"
    # training
    lr = 2e-5
    wd = 0.01
    num_fold = None # 0/1 for training all
    epochs = None
    bs = None
    num_workers = 12
    lr_multi = 10
    sche_step = 5
    sche_decay = 0.5
    # log
    tag = ""
    # general
    seed = 42
    dist_port = 12346

def get_config():
    parser = argparse.ArgumentParser("US patent model")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--num_fold", type=int, default=5)
    parser.add_argument("--pretrain_name", type=str, default="deberta-v3-large")
    parser.add_argument("--loss_name", type=str, default="mse")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)

    opt = parser.parse_args(args=[])
    opt.evaluate = False
    config = CONFIG()
    config.is_kaggle = is_kaggle
    config.is_training = not opt.evaluate
    config.is_evaluation = opt.evaluate
    def update_param(name, config, opt):
        ori_value = getattr(config, name)
        if ori_value is None:
            setattr(config, name, getattr(opt, name))
    update_param("num_fold", config, opt)
    update_param("pretrain_name", config, opt)
    update_param("bs", config, opt)
    update_param("epochs", config, opt)
    update_param("loss_name", config, opt)

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
    config.tag = config.tag + f"{config.pretrain_name}_{config.dataset_name}_{config.loss_name}-N{config.num_fold}"
    config.save_name = f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}_{config.tag}"
    config.save_path = (
        f"./out/{config.save_name}/" if not config.is_kaggle
        else f"/kaggle/working/"
    )
    return config

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

config = get_config()
# initiate logger
logger.config_logger(output_dir=config.save_path)

# get dataset
train_data, test_data = give_rawdata("train", config), give_rawdata("test", config)
tokenizer = give_tokenizer(config)
collate_fn = give_collate_fn(tokenizer, config)

# training phase
if config.is_evaluation:
    preds_all = []
    if config.num_fold < 2:
        preds = run_test("all", test_data, tokenizer, collate_fn, config)
        preds_all.append(preds)
    else:
        for fold in range(config.num_fold):  
            preds = run_test(fold, test_data, tokenizer, collate_fn, config)
            preds_all.append(preds)
    
    predictions = np.mean(preds_all, axis=0)
    submission = pd.DataFrame(data={
        'id': test_data['id'],
        'score': predictions,
    })
    submission.to_csv(os.path.join(config.save_path, 'submission.csv'), index=False)
    