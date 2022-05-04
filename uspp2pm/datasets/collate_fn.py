import torch
import collections

class DataCollatorWithPaddingSplit:
    def __init___(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        elem = features[0]
        keys_list = list(elem.keys())
        batch_dict = {}
        for key in keys_list:
            if key != "labels":
                sub_features = [item[key] for item in features]
                batch_dict[key] = self.tokenizer.pad(
                    sub_features,
                    padding=self.padding,
                    max_length=self.max_length,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                )
        sub_features = [item["labels"] for item in features]
        batch_dict["labels"] = torch.tensor(sub_features)
        raise NotImplementedError

class DataCollatorCombined:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        elem = batch[0]
        assert isinstance(elem, collections.abc.Mapping)
        keys_list = list(elem.keys())
        batch_dict = {}
        for key in keys_list:
            if key != "labels":
                sub_elem = elem[key]
                if isinstance(sub_elem, str):
                    sub_batch = [item[key] for item in batch]
                    _max_len = max([len(item) for item in sub_batch]) + 2
                    batch_dict[key] = self.tokenizer(sub_batch, _max_len)
                else:
                    batch_dict[key] = torch.stack([item[key] for item in batch], dim=0)
        batch_dict["labels"] = torch.tensor([item["labels"] for item in batch]).float()

        return batch_dict