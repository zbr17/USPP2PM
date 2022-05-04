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
import datetime
import argparse

from uspp2pm.datasets import PatentDatasetCombined, load_split_data_combined
import uspp2pm.logger as logger
from uspp2pm.utils import compute_metrics, give_optimizer
from uspp2pm.models import give_tokenizer, give_model
from uspp2pm.losses import give_criterion
from uspp2pm.engine import train_one_epoch, predict

#%%
class config:
    device = torch.device("cuda:0")
    # dataset
    input_path = (
        "./data/uspp2pm" if not is_kaggle
        else "/kaggle/input/us-patent-phrase-to-phrase-matching"
    )
    title_path = (
        "./data/cpcs/titles.csv" if not is_kaggle
        else "/kaggle/input/uspp2pm/data/cpcs/titles.csv"
    )
    train_data_path = os.path.join(input_path, "train.csv")
    test_data_path = os.path.join(input_path, "test.csv")

    # models
    # model_config = {
    #     "deberta-v3-large": {"embed_dim": 512},
    #     "deberta-v3-base": {"embed_dim": 512}
    # }
    ## training model
    pretrain_name = "deberta-v3-large"
    model_path_train = (
        f"./pretrains/{pretrain_name}" if not is_kaggle
        else f"/kaggle/input/uspp2pm/pretrains/{pretrain_name}"
    )
    ## test model
    infer_name = "test"
    model_path_infer = (
        f"./out/{infer_name}/" if not is_kaggle
        else f"/kaggle/input/uspp2pm/out/{infer_name}"
    )

    # training
    lr = 2e-5 # 2e-5
    wd = 0.01
    num_fold = 4
    epochs = 10
    bs = 16
    num_workers = 12
    lr_multi = 10
    sche_step = 5
    sche_decay = 0.5

    # log
    tag = "v1"
    save_name = f"PRE{pretrain_name}-TAG{tag}-{datetime.datetime.now().strftime('%Y%m%d')}"
    save_path = (
        f"./out/{save_name}/" if not is_kaggle
        else f"/kaggle/working/"
    )

parser = argparse.ArgumentParser("US patent model")
parser.add_argument("--evaluate", action="store_true")

opt = parser.parse_args(args=[])
opt.evaluate = False
config.is_training = not opt.evaluate
config.is_evaluation = opt.evaluate
config.model_path = config.model_path_train if config.is_training else config.model_path_infer

#%%
# initiate logger
logger.config_logger(output_dir=config.save_path)

# get dataset
train_data = load_split_data_combined(data_path=config.train_data_path, title_path=config.title_path, num_fold=config.num_fold)
test_data = load_split_data_combined(data_path=config.test_data_path, title_path=config.title_path)

# training phase
if config.is_training:
    tokenizer = give_tokenizer(config.pretrain_name, config.model_path)
    config.tokenizer = tokenizer
    preds_all, labels_all = [], []
    for fold in range(config.num_fold):
        sub_train_data = train_data[train_data["fold"] != fold].reset_index(drop=True)
        sub_val_data = train_data[train_data["fold"] == fold].reset_index(drop=True)
        sub_train_set = PatentDatasetCombined(data=sub_train_data, is_training=True)
        sub_val_set = PatentDatasetCombined(data=sub_val_data, is_training=True)
        sub_train_set.set_tokenizer(tokenizer)
        sub_val_set.set_tokenizer(tokenizer)

        # get model
        model_config = config.model_config[config.pretrain_name]
        model_config["pretrained"] = config.model_path
        model = give_model(config.pretrain_name, model_config)

        # get criterion
        criterion = give_criterion(config.pretrain_name)

        # get optimizer and scheduler
        optim_config = {
            "lr": config.lr, "wd": config.wd, "lr_multi": config.lr_multi,
            "sche_step": config.sche_step, "sche_decay": config.sche_decay
        }
        optimizer, scheduler = give_optimizer(config.pretrain_name, model, optim_config)

        # model-to-device
        model = model.to(config.device)
        criterion = criterion.to(config.device)

        for epoch in range(config.epochs):
            logger.info(f"Epoch: {epoch}")
            # Start to train
            logger.info("Start to train...")
            preds, labels = train_one_epoch(model, criterion, sub_train_set, optimizer, scheduler, config)
            sub_acc = compute_metrics((preds, labels))
            logger.info(f"TrainSET - Fold: {fold}, Epoch: {epoch}, Acc: {sub_acc}")
            to_save_dict = {
                "model": model,
                "criterion": criterion,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "epoch": epoch
            }
            torch.save(to_save_dict, os.path.join(config.save_path, f"model_{fold}.ckpt"))

            # Validate
            logger.info("Start to validate...")
            preds, labels = predict(model, sub_val_set, config)
            sub_acc = compute_metrics((preds, labels))
            logger.info(f"ValSET - Fold: {fold}, Epoch: {epoch}, Acc: {sub_acc}")
        
        preds_all.append(preds)
        labels_all.append(labels)
    
    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    final_acc = compute_metrics((preds_all, labels_all))
    logger.info(f"Final acc: {final_acc}")
    final_result = pd.DataFrame()


# inference phase
# if config.is_evaluation:
#     predictions = []

#     for fold in range(config.num_fold):
#         model_path = os.path.join(config.model_path, f"uspp2pm_{fold}")
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         test_set = PatentDataset(data=test_data, is_training=False)
#         test_set.set_tokenizer(tokenizer)
        
#         model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
#         trainer = Trainer(
#                 model,
#                 tokenizer=tokenizer
#             )

#         outputs = trainer.predict(test_set)
#         prediction = outputs.predictions.reshape(-1)
#         predictions.append(prediction)
    
#     predictions = np.mean(predictions, axis=0)
#     submission = pd.DataFrame(data={
#         'id': test_data['id'],
#         'score': predictions,
#     })

#     submission.to_csv(os.path.join(config.save_path, 'submission.csv'), index=False)