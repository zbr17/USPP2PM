import os
import sys
import socket
import datetime
import yaml

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

from uspp2pm import logger, tbwriter
from uspp2pm.utils import compute_metrics
from uspp2pm.datasets import give_rawdata, give_collate_fn, give_dataset
from uspp2pm.models import give_tokenizer, give_model
from uspp2pm.optimizers import give_optim, give_warming_optim
from uspp2pm.engine import train_one_epoch, predict

def get_config(opt):
    config = CONFIG()
    config.is_kaggle = is_kaggle
    config.is_training = False
    config.is_evaluation = True

    hparam_dict = {}
    def update_param(name, config, opt):
        hparam_dict[name] = getattr(opt, name)
        setattr(config, name, getattr(opt, name))
    for k in dir(opt):
        if not k.startswith("_") and not k.endswith("_"):
            update_param(k, config, opt)

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
        else f"/kaggle/input/{config.pretrain_name}/{config.pretrain_name}"
    )
    config.model_path_infer = (
        f"./out/{config.infer_name}/" if not config.is_kaggle
        else f"/kaggle/input/{config.infer_name}"
    )
    # log
    name = "-".join([k[:5].upper() + str(v) for k, v in hparam_dict.items()])
    config.tag = config.tag + "-" + name
    config.save_name = f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}-{config.tag}"
    config.save_path = (
        f"./out/{config.save_name}/" if not config.is_kaggle
        else f"/kaggle/working/"
    )
    if config.debug:
        config.save_path = f"./out/debug"
    config.is_distributed = config.nproc_per_node > 1
    return config, hparam_dict

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

def load_config(path, config):
    with open(os.path.join(path, "config.yaml"), mode="r", encoding="utf-8") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config_dict.pop("infer_name")
        config_dict.pop("model_path_infer")
        config_dict.pop("input_path")
        config_dict.pop("title_path")
        config_dict.pop("model_path")
        config_dict.pop("save_path")
        config_dict.pop("is_kaggle")
        config_dict.pop("train_data_path")
        config_dict.pop("test_data_path")
    
    for k, v in config_dict.items():
        setattr(config, k, v)

def main(opt):
    config, hparam_dict = get_config(opt)
    # initiate logger
    logger.config_logger(output_dir=config.save_path)
    load_config(config.model_path_infer, config)
    config.device = torch.device("cuda:0")
    for k in dir(config):
        if not k.startswith("_") and not k.endswith("_"):
            print(f"{k}: {getattr(config, k)}")

    # get dataset
    train_data, test_data = give_rawdata("train", config), give_rawdata("test", config)
    tokenizer = give_tokenizer(config)
    collate_fn = give_collate_fn(tokenizer, config)

    # test phase
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

class CONFIG:
    # dataset
    # loss
    # model
    infer_name = "202206021034--bs24-datascombined-debugfalse-dropo0"
    # optimizer
    # scheduler
    sche_step = 5
    sche_decay = 0.5
    # training 
    num_workers = 8
    # general
    seed = 42
    dist_port = 12346

if __name__ == "__main__":
    parser = argparse.ArgumentParser("US patent model")
    # dataset
    parser.add_argument("--dataset_name", type=str, default="split", 
                                            help="split / combined")
    # loss
    parser.add_argument("--loss_name", type=str, default="mse", 
                                            help="mse / shift_mse / pearson")
    ### ExpLoss
    parser.add_argument("--scale", type=float, default=1.0)
    # model
    parser.add_argument("--pretrain_name", type=str, default="deberta-v3-large", 
                                            help="bert-for-patents / deberta-v3-large")
    parser.add_argument("--model_name", type=str, default="combined_baseline",
                                            help="combined_baseline / split_baseline / split_similarity")
    parser.add_argument("--handler_name", type=str, default="hidden_cls_emb",
                                            help="cls_emb / hidden_cls_emb / max_pooling / mean_max_pooling / hidden_cls_emb / hidden_weighted_cls_emb / hidden_lstm_cls_emb / hidden_attention_cls_emb")
    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=768)
    ### combined_hdc
    parser.add_argument("--num_block", type=int, default=1)
    parser.add_argument("--update_rate", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--growth_rate", type=int, default=2)
    # optimizer
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--lr_multi", type=float, default=1)
    # scheduler
    # training
    parser.add_argument("--num_fold", type=int, default=5, 
                                            help="0/1 for training all")
    parser.add_argument("--bs", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    # general
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nproc_per_node", type=int, default=2)

    opt = parser.parse_args(args=[])
    main(opt)