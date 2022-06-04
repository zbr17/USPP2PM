from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

from uspp2pm import tbwriter
from .handler import give_handler
from .ensembler import give_ensembler

class Mlp(nn.Module):
    def __init__(self, size_list=[7,64,2], dropout=0.):
        super().__init__()
        self.size_list = size_list
        self.num_layer = len(self.size_list) - 1

        self.layer_list = []
        for i in range(self.num_layer):
            self.layer_list.append(nn.Linear(self.size_list[i], self.size_list[i+1]))
            self.layer_list.append(nn.LeakyReLU(inplace=True))
            self.layer_list.append(nn.Dropout(p=dropout))
        self.layer_list = nn.ModuleList(self.layer_list)
    
    def forward(self, x):
        for module in self.layer_list:
            x = module(x)
        return x

class CombinedHDCV2(nn.Module):
    """
    Hard-aware deeply cascaded.
    """
    def __init__(
        self, 
        criterion,
        pretrain,
        config
    ):
        super().__init__()
        # get args
        self.config = config
        cache_dir = config.model_path
        num_layer = config.num_layer
        dropout = config.dropout
        growth_rate = config.growth_rate
        self.num_block = config.num_block
        self.output_dim = config.output_dim
        # initialize
        self.criterion = criterion
        self.model = pretrain.from_pretrained(
            pretrained_model_name_or_path=cache_dir,
            num_labels=1
        )
        meta_handler = give_handler(config)
        self.model.config.output_hidden_states = meta_handler.output_hidden_states
        self.output_dim = int(self.output_dim * meta_handler.multiply)
        config.num_hidden_layers = self.model.config.num_hidden_layers
        self.handler = meta_handler(config)

        self.base_model_list = [self.model]
        self.new_model_list = []

        # get ensemble function
        self.init_ensemble()

        # initiate block ensemble
        for i in range(self.num_block):
            sub_embedder = nn.Linear(self.output_dim, 1)
            self.new_model_list.append(sub_embedder)
            setattr(self, f"embedder{i}", sub_embedder)
        
        self.init_weights()
    
    def init_ensemble(self):
        if isinstance(self.criterion, nn.MSELoss):
            self.ensemble = give_ensembler(self.criterion, self.config)
            if isinstance(self.ensemble, nn.Module):
                self.new_model_list.append(self.ensemble)
        else:
            raise TypeError(f"Invalid loss type: {type(self.criterion)}")
    
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
        feature = self.handler(outputs, data_info)
        assert len(feature) == self.num_block

        # hard-aware deeply cascaded module
        loss_list = []
        out_list = []
        self.ensemble.fresh()
        for i in range(self.num_block):
            sub_embedder = getattr(self, f"embedder{i}")
            out = 1.25 * torch.sigmoid(sub_embedder(feature[i])).squeeze() - 0.125
            out_list.append(out)
            
            if self.training:
                loss = self.ensemble(i, labels, out)
                loss_list.append(loss)
                tbwriter.add_scalar(f"loss/{i}", loss)

        out = self.ensemble.predict(out_list)
        
        if self.training:
            loss_list.append(self.criterion(out, labels).mean())
            return out, torch.stack(loss_list).sum()
        else:
            return out

