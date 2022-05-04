from .dataset_combined import PatentDatasetCombined, load_split_data_combined
from .dataset_split import PatentDatasetSplit, load_split_data_split
from .collate_fn import DataCollatorWithPadding

_dataset_dict = {
    "combined": PatentDatasetCombined,
    "split": PatentDatasetSplit
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
    collate_fn = DataCollatorWithPadding(tokenizer)
    return collate_fn

def give_dataset(raw_data, is_training, tokenizer, config):
    _meta_dataset = _dataset_dict[config.dataset_name]
    dataset = _meta_dataset(data=raw_data, is_training=is_training, tokenizer=tokenizer)
    return dataset