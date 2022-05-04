import torch
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

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
        if "labels" in keys_list:
            batch_dict["labels"] = torch.tensor([item["labels"] for item in features]).float()
        return batch_dict