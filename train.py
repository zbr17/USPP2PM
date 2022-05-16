#%%
import logging
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
    debug = False
    nproc_per_node = 2
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
    num_workers = 8
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
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nproc_per_node", type=int, default=2)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--num_fold", type=int, default=5)
    parser.add_argument("--dataset_name", type=str, default="split")
    parser.add_argument("--pretrain_name", type=str, default="deberta-v3-large")
    parser.add_argument("--loss_name", type=str, default="mse")
    parser.add_argument("--bs", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)

    opt = parser.parse_args()
    opt.evaluate = False
    config = CONFIG()
    config.is_kaggle = is_kaggle
    config.is_training = not opt.evaluate
    config.is_evaluation = opt.evaluate
    def update_param(name, config, opt):
        ori_value = getattr(config, name)
        setattr(config, name, getattr(opt, name))
    update_param("debug", config, opt)
    update_param("nproc_per_node", config, opt)
    update_param("num_fold", config, opt)
    update_param("dataset_name", config, opt)
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
    config.tag = config.tag + f"{config.pretrain_name}-{config.dataset_name}-{config.loss_name}-N{config.num_fold}"
    config.save_name = f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}-{config.tag}"
    config.save_path = (
        f"./out/{config.save_name}/" if not config.is_kaggle
        else f"/kaggle/working/"
    )
    if config.debug:
        config.save_path = f"./out/debug"
    config.is_distributed = config.nproc_per_node > 1
    return config

def run(index, train_data, val_data, tokenizer, collate_fn, is_val, config):
    train_set = give_dataset(train_data, True, tokenizer, config)
    val_set = give_dataset(val_data, True, tokenizer, config) if is_val else None

    # get model and criterion
    model = give_model(config).to(config.device)
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.rank])
    criterion = give_criterion(config).to(config.device)

    # get optimizer and scheduler
    optimizer, scheduler = give_optim(model, config)

    best_val_acc, best_epoch = -1, 0
    cur_acc = 0
    for epoch in range(config.epochs):
        config.epoch = epoch
        logger.info(f"Epoch: {epoch}")
        # Start to train
        logger.info("Start to train...")
        preds, labels = train_one_epoch(model, criterion, collate_fn, train_set, optimizer, scheduler, config)
        sub_acc = compute_metrics((preds, labels))["pearson"]
        logger.info(f"TrainSET - Fold: {index}, Epoch: {epoch}, Acc: {sub_acc}")

        if is_val:
            # Validate
            logger.info("Start to validate...")
            preds, labels = predict(model, collate_fn, val_set, config)
            sub_acc = compute_metrics((preds, labels))["pearson"]
            cur_acc = sub_acc
            logger.info(f"ValSET - Fold: {index}, Epoch: {epoch}, Acc: {sub_acc}")
            # detect if collapse
            if sub_acc < 0.5:
                logger.info("Training collapse!!!")
                raise RuntimeError
        
        if config.rank == 0:
            model_params = (
                model.module.state_dict()
                if dist.is_initialized()
                else model.state_dict()
            )
            to_save_dict = {
                "model": model_params,
                "epoch": epoch
            }
            if is_val:
                if cur_acc > best_val_acc:
                    best_val_acc = cur_acc
                    best_epoch = epoch
                    torch.save(to_save_dict, os.path.join(config.save_path, f"model_{index}.ckpt"))
                    logger.info(f"New best checkpoint: Epoch {best_epoch}")
                logger.info(f"ValBEST- Fold: {index}, Epoch: {best_epoch}, Acc: {best_val_acc}")
            else:
                torch.save(to_save_dict, os.path.join(config.save_path, f"model_{index}.ckpt"))
    
    if is_val:
        return preds, labels
    else:
        return None, None

def main_worker(gpu, config):
    # initiate dist
    config.world_size = torch.cuda.device_count()
    config.rank = gpu
    config.device = torch.device(f"cuda:{config.rank}")
    torch.cuda.set_device(config.rank)
    if config.is_distributed:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{config.dist_port}",
            world_size=config.world_size,
            rank=config.rank
        )

    # initiate seed
    np.random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    cudnn.deterministic = True

    # initiate logger
    logger.config_logger(output_dir=config.save_path, dist_rank=config.rank)

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

                is_collapse = True
                while(is_collapse):
                    try:
                        preds, labels = run(fold, sub_train_data, sub_val_data, tokenizer, collate_fn, True, config)
                        is_collapse = False
                    except Exception as e:
                        is_collapse = True
                        print(f"Except: {e}")
                
                preds_all.append(preds)
                labels_all.append(labels)
            
            preds_all = np.concatenate(preds_all, axis=0)
            labels_all = np.concatenate(labels_all, axis=0)
            final_acc = compute_metrics((preds_all, labels_all))
            logger.info(f"Final acc: {final_acc}")

def main():
    config = get_config()
    if config.nproc_per_node > 1:
        mp.spawn(main_worker, nprocs=config.nproc_per_node, args=(config,))
    else:
        main_worker(0, config)

if __name__ == "__main__":
    main()