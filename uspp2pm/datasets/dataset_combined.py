from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np 
import uspp2pm.logger as logger

class PatentDatasetCombined(Dataset):
    def __init__(self, data: pd.DataFrame, is_training: bool = True):
        super().__init__()
        self.data = data

        self.is_training = is_training
        self.inputs = data["input"].values.astype(str)
        if self.is_training:
            self.labels = data["score"].values
    
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx) -> dict:
        output_dict = {
            "inputs": self.inputs[idx],
        }
        
        if self.is_training:
            output_dict["labels"] = self.labels[idx]

        return output_dict

def load_split_data_combined(data_path: str, title_path: str, num_fold: int = 0) -> pd.DataFrame:
    # Load data
    data = pd.read_csv(data_path)
    title = pd.read_csv(title_path)
    data = data.merge(title, left_on="context", right_on="code")
    data["input"] = data["title"] + "[SEP]" + data["anchor"] + "[SEP]" + data["target"]

    # Split data
    if num_fold == 0:
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