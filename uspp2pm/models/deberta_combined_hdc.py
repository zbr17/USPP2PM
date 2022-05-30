from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

from transformers.models.deberta_v2 import DebertaV2ForSequenceClassification

from uspp2pm import tbwriter

class Mlp(nn.Module):
    def __init__(self, size_list=[7,64,2]):
        super().__init__()
        self.size_list = size_list
        self.num_layer = len(self.size_list) - 1

        self.layer_list = []
        for i in range(self.num_layer):
            self.layer_list.append(nn.Linear(self.size_list[i], self.size_list[i+1]))
            # self.layer_list.append(nn.BatchNorm1d(self.size_list[i+1]))
            self.layer_list.append(nn.LeakyReLU(inplace=True))
        self.layer_list = nn.ModuleList(self.layer_list)
    
    def forward(self, x):
        for module in self.layer_list:
            x = module(x)
        return x

class DeBertaCombinedHDC(nn.Module):
    """
    Hard-aware deeply cascaded.
    """
    def __init__(
        self, 
        criterion,
        config
    ):
        super().__init__()
        # get args
        cache_dir = config.model_path
        num_layer = config.num_layer
        self.num_block = config.num_block
        self.rate = config.update_rate
        # initialize
        self.criterion = criterion
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=cache_dir,
            num_labels=1
        )
        self.output_dim = self.model.pooler.output_dim
        self.model.classifier = nn.Identity()

        self.base_model_list = [self.model]
        self.new_model_list = []

        # get ensemble function
        self.init_ensemble()

        # initiate block ensemble
        for i in range(self.num_block):
            size_list = [self.output_dim] * (num_layer + 1)
            sub_block = Mlp(size_list=size_list)
            self.new_model_list.append(sub_block)
            setattr(self, f"block{i}", sub_block)

            sub_embedder = nn.Linear(self.output_dim, 1)
            self.new_model_list.append(sub_embedder)
            setattr(self, f"embedder{i}", sub_embedder)
        self.beta = [0.5] * self.num_block
        
        self.init_weights()
    
    def init_ensemble(self):
        if isinstance(self.criterion, nn.MSELoss):
            self.ensemble = MSEEnsemble()
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
        feature = self.model.dropout(pooled_output)

        # hard-aware deeply cascaded module
        loss_list = []
        out_list = []
        cum_out: Tensor = None
        for i in range(self.num_block):
            sub_block = getattr(self, f"block{i}")
            sub_embedder = getattr(self, f"embedder{i}")
            feature = sub_block(feature)
            cur_out = torch.sigmoid(sub_embedder(feature)).squeeze()
            
            if self.training:
                cur_beta, cum_out = self.ensemble(labels, cur_out, cum_out)
                self.beta[i] = (1-self.rate) * self.beta[i] + self.rate * cur_beta
                tbwriter.add_scalar(f"cur-beta/{i}", cur_beta)
                tbwriter.add_scalar(f"cum-beta/{i}", self.beta[i])

                cur_loss = self.criterion(cur_out, labels)
                loss_list.append(cur_loss)
                tbwriter.add_scalar(f"loss/{i}", cur_loss)
            
            out_list.append(cur_out)

        out = self.ensemble.predict(self.beta, out_list)

        if self.training:
            return out, torch.stack(loss_list).sum()
        else:
            return out

class MSEEnsemble:
    def __call__(self, labels, cur_out, cum_out=None):
        # get max error
        e_max = torch.max(torch.abs(cur_out - labels))
        # get relative error
        e_rel = (cur_out - labels).pow(2) / (e_max.pow(2) + 1e-8)
        # get regressor total error
        if cum_out is None:
            cum_out = torch.ones_like(e_rel) / len(e_rel)
        e_ttl = torch.sum(cum_out * e_rel)
        # get regressor weight
        beta = (e_ttl / (1 - e_ttl + 1e-8))
        # update cum-out
        cum_ttl = torch.sum(cum_out * torch.pow(beta, 1 - e_rel)) + 1e-8
        cum_out = cum_out * torch.pow(beta, 1 - e_rel) / cum_ttl
        return beta.item(), cum_out.detach()
    
    def predict(self, beta_list, out_list):
        out_stack = torch.stack(out_list, dim=-1)
        beta_list = torch.tensor(beta_list).to(out_stack.device)

        out = torch.median(out_stack, dim=-1)[0]
        coef = torch.sum(torch.log(1 / beta_list))

        return coef * out