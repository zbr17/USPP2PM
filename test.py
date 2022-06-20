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

def get_config(opt):
    config = CONFIG()
    config.is_kaggle = is_kaggle

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
        "./data/cpcs/cpc_texts.pth" if not config.is_kaggle
        else "/kaggle/input/cpccode/cpc_texts.pth"
    )
    config.train_data_path = os.path.join(config.input_path, "train.csv")
    config.test_data_path = os.path.join(config.input_path, "test.csv")
    # model
    config.model_path = (
        f"./pretrains/{config.pretrain_name}/{config.pretrain_name}" if not config.is_kaggle
        else f"/kaggle/input/{config.pretrain_name}/{config.pretrain_name}"
    )
    if not os.path.exists(config.model_path) and config.is_kaggle:
        config.model_path = f"/kaggle/input/{config.pretrain_name}"
    
    config.model_path_infer = (
        f"./out/{config.infer_name}/" if not config.is_kaggle
        else f"/kaggle/input/{config.infer_name}"
    )
    # log
    name = "-".join([k[:2].upper() + str(v)[:5] for k, v in hparam_dict.items()])
    config.tag = name
    config.save_name = f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}--infer-{config.tag}".replace(".", "-").replace("_", "-")
    config.save_path = (
        f"./out/{config.save_name[:100]}/" if not config.is_kaggle
        else f"/kaggle/working/"
    )
    if config.debug:
        config.save_path = f"./out/debug"
    config.is_distributed = config.nproc_per_node > 1
    return config, hparam_dict


def run_test(index, test_data, tokenizer, collate_fn, config):
    test_set = give_dataset(test_data, False, tokenizer, config)

    # get model and criterion
    model = give_model(config).to(config.device)

    # load parameters
    model_path = os.path.join(config.model_path_infer, f"model_{index}.ckpt")
    logger.info(f"Loading: {model_path}")
    params_dict = torch.load(model_path)
    model.load_state_dict(params_dict["model"])

    # Evaluate
    logger.info("Start to validate...")
    preds = predict(model, collate_fn, test_set, config, is_test=True)

    return preds

def load_config(opt):
    config, _ = get_config(opt)
    path = config.model_path_infer
    with open(os.path.join(path, "config.yaml"), mode="r", encoding="utf-8") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config_dict.pop("infer_name")
    
    for k in dir(opt):
        if not k.startswith("_") and not k.endswith("_"):
            v = config_dict.get(k, None)
            if v is not None:
                setattr(opt, k, v)

def main(opt):
    config, hparam_dict = get_config(opt)
    # initiate logger
    logger.config_logger(output_dir=config.save_path)
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
    return test_data["id"], predictions, config.save_path

class CONFIG:
    # dataset
    # loss
    # model
    infer_name = "test"
    # optimizer
    # scheduler
    sche_step = 5
    sche_decay = 0.5
    sche_T = 5
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
    parser.add_argument("--has_targets", action="store_true")
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
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nproc_per_node", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    opt = parser.parse_args(args=[])
    
    infer_name_dict = {
        # "202206070929--adjustrue-bs96-datascombined-debugfa": {},
        # "202206071019--adjustrue-bs96-datascombined-debugfa": {},
        # "202206040003--bs24-datascombined-debugfalse-dropo0": {},
        # "202206021034--bs24-datascombined-debugfalse-dropo0": {},
        # "202206041610--bs24-datascombined-debugfalse-dropo0": {},
        # "202206091050--ADTrue-BS24-DAcombi-DEFalse-DR0-5-ENrand-EP10-GR2-HAhidde-LOmse-LR2e-05-LR10-0-MOcombi": {},
        "202206141149--ADTrue-BS24-DAcombi-DEFalse-DR0-5-ENrand-EP10-GR2-HAhidde-HAFalse-LOmse-LR2e-05-LR10-0": {},
        "202206141812--ADTrue-BS24-DAcombi-DEFalse-DR0-5-ENrand-EP10-GR2-HAhidde-HAFalse-LOmse-LR2e-05-LR10-0": {},
        "202206151246--ADTrue-BS24-DAcombi-DEFalse-DR0-5-ENrand-EP10-GR2-HAhidde-HAFalse-LOmse-LR2e-05-LR1-0-": {},
        "202206160055--ADTrue-BS24-DAcombi-DEFalse-DR0-5-ENrand-EP10-GR2-HAhidde-HAFalse-LOmse-LR2e-05-LR1-0-": {},
        "202206161901--ADTrue-BS36-DAcombi-DEFalse-DR0-5-ENrand-EP10-GR2-HAhidde-HAFalse-LOmse-LR2e-05-LR1-0-": {},
        "202206170456--ADTrue-BS24-DAcombi-DEFalse-DR0-5-ENrand-EP10-GR2-HAhidde-HAFalse-LOmse-LR2e-05-LR1-0-": {},
        "202206171651--ADTrue-BS24-DAcombi-DEFalse-DR0-5-ENrand-EP10-GR2-HAhidde-HAFalse-LOmse-LR2e-05-LR1-0-": {},
        "202206180359--ADTrue-BS36-DAcombi-DEFalse-DR0-5-ENrand-EP10-GR2-HAhidde-HAFalse-LOmse-LR2e-05-LR1-0-": {},
        "202206181010--ADTrue-BS36-DAcombi-DEFalse-DR0-5-ENrand-EP10-GR2-HAhidde-HAFalse-LOmse-LR2e-05-LR1-0-": {}
    }
    ids_list = []
    preds_list = []
    path = None
    for k, v in infer_name_dict.items():
        opt.infer_name = (
            k[:50].lower() if is_kaggle
            else k
        )
        load_config(opt)
        for subk, subv in v.items():
            setattr(opt, subk, subv)
        ids, preds, path = main(opt)
        ids_list.append(ids)
        preds_list.append(preds)
    
    # average
    ttl_preds = np.stack(preds_list)
    ttl_preds = np.mean(ttl_preds, axis=0)
    submission = pd.DataFrame(data={
        'id': ids_list[0],
        'score': ttl_preds,
    })
    submission.to_csv(os.path.join(path, 'submission.csv'), index=False)