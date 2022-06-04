from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

from .handler import give_handler

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
        self.output_dim = config.output_dim
        # initialize
        self.criterion = criterion
        self.is_regre = getattr(self.criterion, "is_regre", True)
        self.model = pretrain.from_pretrained(
            pretrained_model_name_or_path=cache_dir,
            num_labels=1
        )
        meta_handler = give_handler(config)
        self.model.config.output_hidden_states = meta_handler.output_hidden_states
        self.output_dim = int(self.output_dim * meta_handler.multiply)
        config.num_hidden_layers = self.model.config.num_hidden_layers
        self.handler = meta_handler(config)

        if num_layer == 1:
            if self.is_regre:
                self.classifier = nn.Linear(self.output_dim, 1)
            else:
                self.classifier = nn.Linear(self.output_dim, 5)
        else:
            if self.is_regre:
                size_list = [self.output_dim] * num_layer + [1]
            else:
                size_list = [self.output_dim] * num_layer + [5]
            self.classifier = Mlp(size_list=size_list)
        
        self.base_model_list = [self.model]
        self.new_model_list = [self.classifier]
        if self.handler.trainable:
            self.new_model_list.append(self.handler)
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
        outputs = self.model(**data_info["inputs"])
        embeddings = self.handler(outputs, data_info)

        logits = self.classifier(embeddings)
        if self.is_regre:
            logits = 1.25 * torch.sigmoid(logits).squeeze() - 0.125

        if self.training:
            # compute losses
            loss = self.criterion(logits, labels).mean()
            return logits, loss
        else:
            return logits

