import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from transformers.models.deberta_v2 import DebertaV2ForSequenceClassification
# from DeBERTa import deberta

# 'base' 'large' 'base-mnli' 'large-mnli' 'xlarge' 'xlarge-mnli' 'xlarge-v2' 'xxlarge-v2'
# _pre_trained = {
#     "deberta-v3-large": "large",
#     "deberta-v3-base": "base"
# }

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
        self.deberta = DebertaV2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=cache_dir,
            num_labels=1
        )

        self.deberta.apply_state(cache_dir=cache_dir)

        # self.init_model(pretrained)
        # self.embedder = nn.Linear(self.output_dim, 1)
    
    def forward(
        self,
        input
    ) -> torch.Tensor:
        embed = self.deberta(**input)["hidden_states"][-1]
        return embed

# class Mlp(nn.Module):
#     def __init__(self, size_list=[7,64,2]):
#         super().__init__()
#         self.size_list = size_list
#         self.num_layer = len(self.size_list) - 1

#         self.layer_list = []
#         for i in range(self.num_layer):
#             self.layer_list.append(nn.Linear(self.size_list[i], self.size_list[i+1]))
#             if i != self.num_layer-1:
#                 # self.layer_list.append(nn.BatchNorm1d(self.size_list[i+1]))
#                 self.layer_list.append(nn.LeakyReLU(inplace=True))
#         self.layer_list = nn.ModuleList(self.layer_list)
    
#     def forward(self, x):
#         for module in self.layer_list:
#             x = module(x)
#         return x

# class DeBertaPP2P(nn.Module):
#     """
#     Without dropout in the final layer.
#     """
#     def __init__(
#         self, 
#         pretrained: str = None, 
#         embed_dim: int = 512
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.init_model(pretrained)
#         self.embedder_a = nn.Linear(self.output_dim, embed_dim)
#         self.embedder_t = nn.Linear(self.output_dim, embed_dim)
#         self.embedder_c = nn.Linear(self.output_dim, embed_dim)
#         self.mlp_ac = Mlp(size_list=[2*embed_dim, 4*embed_dim, 4*embed_dim, embed_dim])
#         self.mlp_tc = Mlp(size_list=[2*embed_dim, 4*embed_dim, 4*embed_dim, embed_dim])

#         self.base_model_list = [self.deberta, self.pooler, self.dropout]
#         self.new_model_list = [self.embedder_a, self.embedder_t, self.embedder_c,
#         self.mlp_ac, self.mlp_tc]
    
#     def init_weights(self):
#         def _init_weights(module: nn.Module):
#             for sub_module in module.modules():
#                 init.kaiming_normal_(sub_module.weight, a=math.sqrt(5))
#                 init.zeros_(sub_module.bias)
#         to_init_list = [
#             self.embedder_a, self.embedder_t, self.embedder_c,
#             self.mlp_ac, self.mlp_tc
#         ]
#         for module in to_init_list:
#             _init_weights(module)
    
#     def init_model(self, pretrained):
#         model = AutoModelForSequenceClassification.from_pretrained(pretrained, num_labels=1)
#         self.deberta = model.deberta
#         self.pooler = model.pooler
#         self.dropout = model.dropout # NOTE: Not used!
#         self.output_dim = model.pooler.output_dim
    
#     def forward(
#         self,
#         input_a: dict,
#         input_t: dict,
#         input_c: dict
#     ) -> torch.Tensor:
#         # Compute embeddings
#         out_a = self.deberta(**input_a)[0]
#         out_t = self.deberta(**input_t)[0]
#         out_c = self.deberta(**input_c)[0]

#         pooled_a = self.pooler(out_a)
#         pooled_t = self.pooler(out_t)
#         pooled_c = self.pooler(out_c)

#         pooled_a = self.dropout(pooled_a)
#         pooled_t = self.dropout(pooled_t)
#         pooled_c = self.dropout(pooled_c)

#         embed_a = self.embedder_a(pooled_a)
#         embed_t = self.embedder_t(pooled_t)
#         embed_c = self.embedder_c(pooled_c)

#         # Fuse context
#         embed_ac = torch.cat([embed_a, embed_c], dim=-1)
#         embed_tc = torch.cat([embed_t, embed_c], dim=-1)
#         embed_ac = self.mlp_ac(embed_ac)
#         embed_tc = self.mlp_tc(embed_tc)

#         if self.training:
#             sim_inter = F.cosine_similarity(embed_a, embed_t, dim=-1)
#             sim_inter = 0.5 * (sim_inter + 1) # Rescale to [0,1]
#             # Compute similarity
#             sim = F.cosine_similarity(embed_ac, embed_tc, dim=-1)
#             sim = 0.5 * (sim + 1) # Rescale to [0,1]
#             return sim, sim_inter
#         else:
#             # Compute similarity
#             sim = F.cosine_similarity(embed_ac, embed_tc, dim=-1)
#             sim = 0.5 * (sim + 1) # Rescale to [0,1]
#             return sim