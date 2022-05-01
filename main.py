import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import argparse

from uspp2pm.dataset import PatentDataset, load_split_data
import uspp2pm.logger as logger
from uspp2pm.utils import compute_metrics

os.environ["WANDB_DISABLED"] = "true"

class config:
    # dataset
    input_path = "./data/uspp2pm"
    title_path = "./data/cpcs/titles.csv"
    model_path = "./pretrains/deberta-v3-large"
    train_data_path = os.path.join(input_path, "train.csv")
    test_data_path = os.path.join(input_path, "test.csv")
    # training
    lr = 2e-5
    wd = 0.01
    num_fold = 5
    epochs = 5
    bs = 16
    # log
    save_path = "./out/test/"

parser = argparse.ArgumentParser("US patent model")
parser.add_argument("--evaluate", action="store_true")

opt = parser.parse_args()
config.is_training = not opt.evaluate
config.is_evaluation = opt.evaluate

# initiate logger
logger.config_logger(output_dir=config.save_path)

# get dataset
train_data = load_split_data(data_path=config.train_data_path, title_path=config.title_path, num_fold=config.num_fold)
test_data = load_split_data(data_path=config.test_data_path, title_path=config.title_path)

# get model
tokenizer = AutoTokenizer.from_pretrained(config.model_path)
print(tokenizer)

# training phase
if config.is_training:
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


# # inference phase
# if config.is_evaluation:
#     predictions = []

#     for fold in range(config.num_fold):
#         test_set = PatentDataset(data=test_data, is_training=False)
#         model = AutoModelForSequenceClassification.from_pretrained(config, num_labels=1)
#         trainer = Trainer(
#                 model,
#                 tokenizer=tokenizer
#             )

#         outputs = trainer.predict(test_set)
#         prediction = outputs.predictions.reshape(-1)
#         predictions.append(prediction)
    
#     predictions = np.mean(predictions, axis=0)
    # submission = datasets.Dataset.from_dict({
    #     'id': test_set['id'],
    #     'score': predictions,
    # })

    # submission.to_csv('submission.csv', index=False)