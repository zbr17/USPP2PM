from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

class BertPatentCombined(nn.Module):
    """
    Without dropout in the final layer.
    """
    def __init__(
        self, 
        criterion,
        config
    ):
        super().__init__()
        # get args
        cache_dir = config.model_path
        # initialize
        self.criterion = criterion
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=cache_dir,
            num_labels=1
        )
        raise NotImplementedError()
    
    def forward(
        self,
        data_info: dict,
        labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        input_ids = data_info["inputs"]["input_ids"]
        attention_mask = data_info["inputs"]["attention_mask"]
        token_type_ids = data_info["inputs"]["token_type_ids"]

        outputs = self.model.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs[1]
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        logits = torch.sigmoid(logits).squeeze()

        if self.training:
            # compute losses
            loss = self.criterion(logits, labels)
            return logits, loss
        else:
            return logits

class Mlp(nn.Module):
    def __init__(self, size_list=[7,64,2]):
        super().__init__()
        self.size_list = size_list
        self.num_layer = len(self.size_list) - 1

        self.layer_list = []
        for i in range(self.num_layer):
            self.layer_list.append(nn.Linear(self.size_list[i], self.size_list[i+1]))
            if i != self.num_layer-1:
                # self.layer_list.append(nn.BatchNorm1d(self.size_list[i+1]))
                self.layer_list.append(nn.LeakyReLU(inplace=True))
        self.layer_list = nn.ModuleList(self.layer_list)
    
    def forward(self, x):
        for module in self.layer_list:
            x = module(x)
        return x
