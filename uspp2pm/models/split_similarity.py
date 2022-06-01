from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
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

class SplitSimilarity(nn.Module):
    """
    Split with similarity modules
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
        
        hidden_dim = 2 * self.output_dim
        if num_layer == 1:
            self.ac_embedder = nn.Linear(hidden_dim, self.output_dim)
            self.tc_embedder = nn.Linear(hidden_dim, self.output_dim)
        else:
            size_list = [hidden_dim] * num_layer + [self.output_dim]
            self.ac_embedder = Mlp(size_list=size_list)
            self.tc_embedder = Mlp(size_list=size_list)

        self.base_model_list = [self.model]
        self.new_model_list = [
            self.ac_embedder, self.tc_embedder
        ]
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
        input_a = data_info["anchors"]
        input_t = data_info["targets"]
        input_c = data_info["contexts"]

        # Compute embeddings
        out_a = self.model.deberta(**input_a)[0]
        out_t = self.model.deberta(**input_t)[0]
        out_c = self.model.deberta(**input_c)[0]

        pooled_a = self.model.pooler(out_a)
        pooled_t = self.model.pooler(out_t)
        pooled_c = self.model.pooler(out_c)

        pooled_a = self.model.dropout(pooled_a)
        pooled_t = self.model.dropout(pooled_t)
        pooled_c = self.model.dropout(pooled_c)

        # Fuse context
        ac_emb = torch.cat([pooled_a, pooled_c], dim=-1)
        tc_emb = torch.cat([pooled_t, pooled_c], dim=-1)
        ac_emb = self.ac_embedder(ac_emb)
        tc_emb = self.tc_embedder(tc_emb)

        logits = F.cosine_similarity(ac_emb, tc_emb, dim=-1)
        logits = (logits + 1) * 0.5

        if self.training:
            # compute losses
            loss = self.criterion(logits, labels).mean()
            return logits, loss
        else:
            return logits
