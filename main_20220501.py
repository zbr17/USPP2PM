#%%
import os
import sys
import socket
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import argparse

from uspp2pm.dataset import PatentDataset, load_split_data
import uspp2pm.logger as logger
from uspp2pm.utils import compute_metrics

os.environ["WANDB_DISABLED"] = "true"

#%%
class config:
    outprefix = "." if hostname == "zebra" else "/kaggle/working"
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
    lr = 2e-5
    wd = 0.01
    num_fold = 5
    epochs = 5
    bs = 16

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
train_data = load_split_data(data_path=config.train_data_path, title_path=config.title_path, num_fold=config.num_fold)
test_data = load_split_data(data_path=config.test_data_path, title_path=config.title_path)

# training phase
if config.is_training:
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    oof_df = pd.DataFrame()
    for fold in range(config.num_fold):
        sub_train_data = train_data[train_data["fold"] != fold].reset_index(drop=True)
        sub_val_data = train_data[train_data["fold"] == fold].reset_index(drop=True)
        sub_train_set = PatentDataset(data=sub_train_data, is_training=True)
        sub_val_set = PatentDataset(data=sub_val_data, is_training=True)
        sub_train_set.set_tokenizer(tokenizer)
        sub_val_set.set_tokenizer(tokenizer)

        args = TrainingArguments(
            output_dir=config.save_path,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=config.lr,
            per_device_train_batch_size=config.bs,
            per_device_eval_batch_size=config.bs,
            num_train_epochs=config.epochs,
            weight_decay=config.wd,
            metric_for_best_model="pearson",
            load_best_model_at_end=True,
        )

        model = AutoModelForSequenceClassification.from_pretrained(config.model_path, num_labels=1)
        trainer = Trainer(
            model,
            args,
            train_dataset=sub_train_set,
            eval_dataset=sub_val_set,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.save_model(f"uspp2pm_{fold}")

        outputs = trainer.predict(sub_val_set)
        predictions = outputs.predictions.reshape(-1)
        sub_val_data["preds"] = predictions
        oof_df = pd.concat([oof_df, sub_val_data])
    
    # save data
    predictions = oof_df["preds"].values
    labels = oof_df["score"].values
    eval_pred = predictions, labels
    logger.info(str(compute_metrics(eval_pred)))

    oof_df.to_csv(os.path.join(config.save_path, "oof_df.csv"))


# inference phase
if config.is_evaluation:
    predictions = []

    for fold in range(config.num_fold):
        model_path = os.path.join(config.model_path, f"uspp2pm_{fold}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        test_set = PatentDataset(data=test_data, is_training=False)
        test_set.set_tokenizer(tokenizer)
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
        trainer = Trainer(
                model,
                tokenizer=tokenizer
            )

        outputs = trainer.predict(test_set)
        prediction = outputs.predictions.reshape(-1)
        predictions.append(prediction)
    
    predictions = np.mean(predictions, axis=0)
    submission = pd.DataFrame(data={
        'id': test_data['id'],
        'score': predictions,
    })

    submission.to_csv(os.path.join(config.save_path, 'submission.csv'), index=False)