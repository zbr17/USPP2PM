from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np 
from uspp2pm import logger
import random

import sys
sys.path.append("./dependency/nlpaug")
try:
    import nlpaug.augmenter.word as naw
except:
    print("No nlpaug")

class PatentDatasetCombined(Dataset):
    def __init__(self, data: pd.DataFrame, is_training: bool = True, tokenizer = None):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

        self.is_training = is_training
        self.inputs = data["input"].values.astype(str)
        if self.is_training:
            self.labels = data["score"].values
            self.aug = Augs()
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx) -> dict:
        if self.is_training: 
            input_data = self.inputs[idx].split("[SEP]")
            input_data = [self.aug(item) for item in input_data]
            input_data = "[SEP]".join(input_data)
        
        output_dict = {
            "inputs": self.tokenizer(input_data),
        }
        
        if self.is_training:
            output_dict["labels"] = self.labels[idx]

        return output_dict

class SynonymAug:
    def __init__(self, p):
        self.p = p
        self.aug = naw.SynonymAug(aug_src="wordnet")
    
    def __call__(self, x):
        if random.random() < self.p:
            x = self.aug.augment(x)
        else:
            x = x
        return x

class WordEmbsAug:
    def __init__(self, p):
        self.p = p
        self.aug = naw.WordEmbsAug(model_type="word2vec", action="insert")
    
    def __call__(self, x):
        if random.random() < self.p:
            x = self.aug.augment(x)
        else:
            x = x
        return x

class Augs:
    def __init__(self):
        self.aug_list = [
            # SynonymAug(p=0.3),
            # WordEmbsAug(p=0.3),
        ]

    def __call__(self, x):
        for aug in self.aug_list:
            x = aug(x)
        return x

def load_split_data_combined(data_path: str, title_path: str, num_fold: int = 0) -> pd.DataFrame:
    # Load data
    data = pd.read_csv(data_path)
    mapping = torch.load(title_path)

    data["context_text"] = data["context"].map(mapping)
    data["input"] = (
        data["anchor"].apply(str.lower) + "[SEP]" 
        + data["target"].apply(str.lower) + "[SEP]" 
        + data["context_text"].apply(str.lower)
    )
    data.reset_index(inplace=True)

    # Split data
    if num_fold < 2:
        logger.info("Not create KFold")
    else:
        logger.info("Create {} Folds".format(num_fold))
        data = create_folds(data, num_fold)
    
    return data


def create_folds(data: pd.DataFrame, num_splits: int) -> pd.DataFrame:
    """
    Code from: https://www.kaggle.com/code/abhishek/phrase-matching-folds
    """
    # we create a new column called kfold and fill it with -1
    data["fold"] = -1
    
    # the next step is to randomize the rows of the data
    # data = data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    # num_bins = int(np.floor(1 + np.log2(len(data))))
    
    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["score"], bins=5, labels=False
    )
    
    # initiate the kfold class from model_selection module
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'fold'] = f
    
    # drop the bins column
    data = data.drop("bins", axis=1)

    # return dataframe with folds
    return data

if __name__ == "__main__":
    aug = Augs()
    test_src = "The quick brown fox jumps over the lazy dog."
    out = aug(test_src)
    pass