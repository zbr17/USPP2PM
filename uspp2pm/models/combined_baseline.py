from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

class Mlp(nn.Module):
    def __init__(self, size_list=[7,64,2]):
        super().__init__()
        self.size_list = size_list
        self.num_layer = len(self.size_list) - 1

        self.layer_list = []
        for i in range(self.num_layer):
            self.layer_list.append(nn.Linear(self.size_list[i], self.size_list[i+1]))
            if i != self.num_layer-1:
                self.layer_list.append(nn.LeakyReLU(inplace=True))
        self.layer_list = nn.ModuleList(self.layer_list)
    
    def forward(self, x):
        for module in self.layer_list:
            x = module(x)
        return x

class CombinedBaseline(nn.Module):
    """
    Without dropout in the final layer.
    """
    def __init__(
        self, 
        criterion,
        pretrain,
        config
    ):
        super().__init__()
        # get args
        cache_dir = config.model_path
        num_layer = config.num_layer
        # initialize
        self.criterion = criterion
        self.model = pretrain.from_pretrained(
            pretrained_model_name_or_path=cache_dir,
            num_labels=1
        )
        self.output_dim = self.model.pooler.output_dim
        self.model.classifier = nn.Identity()
        if num_layer == 1:
            self.classifier = nn.Linear(self.output_dim, 1)
        else:
            size_list = [self.output_dim] * num_layer + [1]
            self.classifier = Mlp(size_list=size_list)
        
        self.base_model_list = [self.model]
        self.new_model_list = [self.classifier]
        self.init_weights()
    
    def init_weights(self):
        def _init_weights(module: nn.Module):
            for sub_module in module.modules():
                if isinstance(sub_module, nn.Linear):
                    init.kaiming_normal_(sub_module.weight, a=math.sqrt(5))
                    init.zeros_(sub_module.bias)
        to_init_list = self.new_model_list
        for module in to_init_list:
            _init_weights(module)
    
    def forward(
        self,
        data_info: dict,
        labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        input_ids = data_info["inputs"]["input_ids"]
        attention_mask = data_info["inputs"]["attention_mask"]
        token_type_ids = data_info["inputs"]["token_type_ids"]

        outputs = self.model.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        encoder_layer = outputs[0]
        pooled_output = self.model.pooler(encoder_layer)
        pooled_output = self.model.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = torch.sigmoid(logits).squeeze()

        if self.training:
            # compute losses
            loss = self.criterion(logits, labels).mean()
            return logits, loss
        else:
            return logits
