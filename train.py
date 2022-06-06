#%%
import os
import sys
import socket
import datetime
import yaml

hostname = socket.gethostname()
if hostname != "zebra":
    is_kaggle = True
    sys.path.append("/kaggle/input")
    sys.path.append("/kaggle/input/uspp2pm/dependency/nlpaug")
else:
    is_kaggle = False
    sys.path.append("./dependency/nlpaug")

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

_COLLAPSE_REPEAT = 6
_CUR_REPEAT = 0
_COLLAPSE_THRESH = 0.2

def get_config(opt):
    config = CONFIG()
    config.is_kaggle = is_kaggle
    config.is_training = True
    config.is_evaluation = False

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
        f"./pretrains/{config.pretrain_name}/{config.pretrain_name}" if not config.is_kaggle
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
        f"./out/{config.save_name[:100]}/" if not config.is_kaggle
        else f"/kaggle/working/"
    )
    if config.debug:
        config.save_path = f"./out/debug"
    config.is_distributed = config.nproc_per_node > 1
    return config, hparam_dict

def run(index, train_data, val_data, tokenizer, collate_fn, is_val, config):
    config.fold = index
    train_set = give_dataset(train_data, True, tokenizer, config)
    val_set = give_dataset(val_data, True, tokenizer, config) if is_val else None

    # get model
    model = give_model(config).to(config.device)
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.rank])

    best_val_acc, best_epoch = -1, 0
    cur_acc = 0

    # warming up
    config.epoch = 0
    optimizer, scheduler = give_warming_optim(model, config)
    logger.info(f"Warming epoch")
    logger.info("Start to warm...")
    preds, labels = train_one_epoch(model, collate_fn, train_set, optimizer, scheduler, config)
    sub_acc = compute_metrics((preds, labels))["pearson"]
    logger.info(f"TrainSET - Fold: {index}, Acc: {sub_acc}")
    tbwriter.add_scalar(f"fold{index}/train/acc", sub_acc)
    # detect if collapse
    if sub_acc < _COLLAPSE_THRESH:
        logger.info("Training collapse!!!")
        return -1, -1

    # get optimizer and scheduler
    optimizer, scheduler = give_optim(model, config)

    for epoch in range(1, config.epochs+1):
        config.epoch = epoch
        logger.info(f"Epoch: {epoch}")
        # Start to train
        logger.info("Start to train...")
        preds, labels = train_one_epoch(model, collate_fn, train_set, optimizer, scheduler, config)
        sub_acc = compute_metrics((preds, labels))["pearson"]
        logger.info(f"TrainSET - Fold: {index}, Epoch: {epoch}, Acc: {sub_acc}")
        tbwriter.add_scalar(f"fold{index}/train/acc", sub_acc)
        # detect if collapse
        if sub_acc < _COLLAPSE_THRESH:
            logger.info("Training collapse!!!")
            return -1, -1

        if is_val:
            # Validate
            logger.info("Start to validate...")
            preds, labels = predict(model, collate_fn, val_set, config)
            sub_acc = compute_metrics((preds, labels))["pearson"]
            cur_acc = sub_acc
            logger.info(f"ValSET - Fold: {index}, Epoch: {epoch}, Acc: {sub_acc}")
            tbwriter.add_scalar(f"fold{index}/val/acc", sub_acc)
            # detect if collapse
            if sub_acc < _COLLAPSE_THRESH:
                logger.info("Training collapse!!!")
                return -1, -1
        
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

def save_config(path, config):
    dict_to_save = {}
    for k in dir(config):
        if not k.startswith("_") and not k.endswith("_"):
            v = getattr(config, k)
            if isinstance(v, (str, float, int, list)):
                dict_to_save[k] = getattr(config, k)
    with open(os.path.join(path, "config.yaml"), mode="w", encoding="utf-8") as f:
        yaml.dump(dict_to_save, f)

def main_worker(gpu, config, hparam_dict):
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

    # initiate recorder
    logger.config_logger(output_dir=config.save_path, dist_rank=config.rank)
    tbwriter.config(output_dir=config.save_path, dist_rank=config.rank)
    if config.rank == 0:
        save_config(config.save_path, config)
    for item in dir(config):
        if not item.startswith("__") and not item.endswith("__"):
            logger.info(f"{item}: {getattr(config, item)}")
    
    tbwriter.add_hparams(
        hparam_dict=hparam_dict,
        metric_dict={"hparam/train_acc": 0},
        progress=-1
    )

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
                    global _CUR_REPEAT
                    preds, labels = run(fold, sub_train_data, sub_val_data, tokenizer, collate_fn, True, config)
                    if isinstance(preds, int) and preds == -1:
                        is_collapse = True
                        _CUR_REPEAT += 1
                        if _CUR_REPEAT > _COLLAPSE_REPEAT:
                            tbwriter.add_hparams(
                                hparam_dict=hparam_dict,
                                metric_dict={"hparam/train_acc": -1},
                                progress=-2
                            )
                            exit(1)
                    else:
                        is_collapse = False
                        _CUR_REPEAT = 0
                
                preds_all.append(preds)
                labels_all.append(labels)

                tbwriter.add_hparams(
                    hparam_dict=hparam_dict,
                    metric_dict={"hparam/train_acc": 0},
                    progress=float(fold) / float(config.num_fold)
                )
            
            preds_all = np.concatenate(preds_all, axis=0)
            labels_all = np.concatenate(labels_all, axis=0)
            final_acc = compute_metrics((preds_all, labels_all))["pearson"]
            logger.info(f"Final acc: {final_acc}")
            tbwriter.add_scalar(f"final/train/acc", final_acc)
            tbwriter.add_hparams(
                hparam_dict=hparam_dict,
                metric_dict={"hparam/train_acc": final_acc},
                progress=1,
            )


def main(opt):
    config, hparam_dict = get_config(opt)
    if config.nproc_per_node > 1:
        mp.spawn(main_worker, nprocs=config.nproc_per_node, args=(config, hparam_dict))
    else:
        main_worker(0, config, hparam_dict)

class CONFIG:
    # dataset
    # loss
    # model
    infer_name = "test"
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
                                            help="mse / shift_mse / pearson / cross_entropy")
    ### ExpLoss
    parser.add_argument("--scale", type=float, default=1.0)
    # model
    parser.add_argument("--pretrain_name", type=str, default="deberta-v3-large", 
                                            help="bert-for-patents / deberta-v3-large")
    parser.add_argument("--model_name", type=str, default="combined_baseline",
                                            help="combined_baseline / split_baseline / split_similarity")
    parser.add_argument("--handler_name", type=str, default="hidden_cls_emb",
                                            help="cls_emb / hidden_cls_emb / max_pooling / mean_max_pooling / hidden_cls_emb / hidden_weighted_cls_emb / hidden_lstm_cls_emb / hidden_attention_cls_emb /hidden_branch_mean_max_pooling")
    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=768)
    parser.add_argument("--adjust", action="store_true")
    ### combined_hdc
    parser.add_argument("--num_block", type=int, default=1)
    parser.add_argument("--update_rate", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--growth_rate", type=int, default=2)
    parser.add_argument("--ensemble_name", type=str, default="hard", 
                                            help="hard / adaboostr2")
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
    parser.add_argument("--seed", type=int, default=42)

    opt = parser.parse_args()
    main(opt)