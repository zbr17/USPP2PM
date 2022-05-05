from typing import Optional
from matplotlib.pyplot import isinteractive
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from transformers.models.deberta_v2 import DebertaV2ForSequenceClassification

class DeBertaPP2PCombined(nn.Module):
    """
    Without dropout in the final layer.
    """
    def __init__(
        self, 
        config
    ):
        super().__init__()
        # get args
        cache_dir = config.model_path
        # initialize
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=cache_dir,
            num_labels=1
        )
    
    def forward(
        self,
        data_info: dict,
    ) -> torch.Tensor:
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
        logits = self.model.classifier(pooled_output)
        logits = torch.sigmoid(logits)
        return logits.squeeze()

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

class DeBertaPP2PSplit(nn.Module):
    """
    Without dropout in the final layer.
    """
    def __init__(
        self, 
        config
    ):
        super().__init__()
        # get args
        cache_dir = config.model_path
        self.embed_dim = config.embed_dim
        # initialize
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=cache_dir,
            num_labels=1
        )
        self.output_dim = self.model.pooler.output_dim
        self.classifier = nn.Linear(3 * self.output_dim, 1)

        # self.embedder_a = nn.Linear(self.output_dim, self.embed_dim)
        # self.embedder_t = nn.Linear(self.output_dim, self.embed_dim)
        # self.embedder_c = nn.Linear(self.output_dim, self.embed_dim)
        # self.mlp_ac = Mlp(size_list=[2*self.embed_dim, self.embed_dim, self.embed_dim])
        # self.mlp_tc = Mlp(size_list=[2*self.embed_dim, self.embed_dim, self.embed_dim])

        self.base_model_list = [self.model]
        # self.new_model_list = [
        #     self.embedder_a, self.embedder_t, self.embedder_c,
        #     self.mlp_ac, self.mlp_tc
        # ]
        self.new_model_list = [
            self.classifier
        ]
        # self.init_weights()
    
    # def init_weights(self):
    #     def _init_weights(module: nn.Module):
    #         for sub_module in module.modules():
    #             if isinstance(sub_module, nn.Linear):
    #                 init.kaiming_normal_(sub_module.weight, a=math.sqrt(5))
    #                 init.zeros_(sub_module.bias)
    #     to_init_list = [
    #         self.embedder_a, self.embedder_t, self.embedder_c,
    #         self.mlp_ac, self.mlp_tc
    #     ]
    #     for module in to_init_list:
    #         _init_weights(module)
    
    def forward(
        self,
        data_info: dict,
    ) -> torch.Tensor:
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

        embed_a = self.embedder_a(pooled_a)
        embed_t = self.embedder_t(pooled_t)
        embed_c = self.embedder_c(pooled_c)

        # Fuse context
        embed_ac = torch.cat([embed_a, embed_c], dim=-1)
        embed_tc = torch.cat([embed_t, embed_c], dim=-1)
        embed_ac = self.mlp_ac(embed_ac)
        embed_tc = self.mlp_tc(embed_tc)

        # Compute similarity
        sim = F.cosine_similarity(embed_ac, embed_tc, dim=-1)
        sim = 0.5 * (sim + 1) # Rescale to [0,1]
        return sim