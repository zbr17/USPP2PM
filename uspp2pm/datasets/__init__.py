from .dataset_combined import PatentDatasetCombined, load_split_data_combined
from .dataset_split import PatentDatasetSplit, load_split_data_split
from .collate_fn import DataCollatorWithPaddingCombined, DataCollatorWithPaddingSplit

_dataset_dict = {
    "combined": PatentDatasetCombined,
    "split": PatentDatasetSplit
}

_collate_dict = {
    "combined": PatentDatasetCombined,
    "split": PatentDatasetSplit
}

_rawdata_dict = {
    "combined": DataCollatorWithPaddingCombined,
    "split": DataCollatorWithPaddingSplit
}

def give_rawdata(config):
    _meta_rawdata_class = _rawdata_dict[config.dataset_name]
    rawdata = _meta_rawdata_class(data_path=config.train_data_path, title_path=config.title_path, num_fold=config.num_fold)
    return rawdata

def give_collate_fn(dataset_name):
    raise NotImplementedError # FIXME