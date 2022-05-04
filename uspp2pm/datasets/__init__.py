from .dataset_combined import PatentDatasetCombined, load_split_data_combined
from .dataset_split import PatentDatasetSplit, load_split_data_split
from .collate_fn import DataCollatorCombined

_dataset_dict = {
    "combined": PatentDatasetCombined,
    "split": PatentDatasetSplit
}

_collate_dict = {
    "combined": DataCollatorCombined,
    "split": None # FIXME
}

_rawdata_dict = {
    "combined": load_split_data_combined,
    "split": load_split_data_split
}

def give_rawdata(flag, config):
    _meta_rawdata_class = _rawdata_dict[config.dataset_name]
    if flag == "train":
        rawdata = _meta_rawdata_class(data_path=config.train_data_path, title_path=config.title_path, num_fold=config.num_fold)
    else:
        rawdata = _meta_rawdata_class(data_path=config.train_data_path, title_path=config.title_path, num_fold=0)
    return rawdata

def give_collate_fn(tokenizer, config):
    _meta_collate_fn = _collate_dict[config.dataset_name]
    collate_fn = _meta_collate_fn(tokenizer)
    return collate_fn

def give_dataset(raw_data, is_training, config):
    _meta_dataset = _dataset_dict[config.dataset_name]
    dataset = _meta_dataset(data=raw_data, is_training=is_training)
    return dataset