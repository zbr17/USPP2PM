from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor

from uspp2pm import tbwriter

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

class CombinedHDC(nn.Module):
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
        # initialize
        self.criterion = criterion
        self.model = pretrain.from_pretrained(
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
        in_dim = self.output_dim
        for i in range(self.num_block):
            hid_dim = in_dim * growth_rate
            out_dim = in_dim
            size_list = [in_dim] + [hid_dim] * (num_layer-1) + [out_dim]
            sub_block = Mlp(size_list=size_list, dropout=dropout)
            self.new_model_list.append(sub_block)
            setattr(self, f"block{i}", sub_block)

            sub_embedder = nn.Linear(out_dim, 1)
            self.new_model_list.append(sub_embedder)
            setattr(self, f"embedder{i}", sub_embedder)
            in_dim = out_dim
        
        self.init_weights()
    
    def init_ensemble(self):
        if isinstance(self.criterion, nn.MSELoss):
            self.ensemble = HardEnsemble(self.config, self.criterion)
            # self.ensemble = MSEEnsemble(self.config, self.criterion)
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
        self.ensemble.fresh()
        for i in range(self.num_block):
            sub_block = getattr(self, f"block{i}")
            sub_embedder = getattr(self, f"embedder{i}")
            feature = sub_block(feature)
            out = torch.sigmoid(sub_embedder(feature)).squeeze()
            out_list.append(out)
            
            if self.training:
                loss = self.ensemble(i, labels, out)
                loss_list.append(loss)
                tbwriter.add_scalar(f"loss/{i}", loss)

        out = self.ensemble.predict(out_list)

        if self.training:
            return out, torch.stack(loss_list).sum()
        else:
            return out

class HardEnsemble:
    def __init__(self, config, criterion):
        self.criterion = criterion
    
    def __call__(self, i, labels, out):
        bs = labels.size(0)
        # generate idx
        if self.info is None:
            sample_i = torch.arange(bs)
        else:
            e_error = (self.info - labels).pow(2)
            # sampling probability
            sample_p = e_error / torch.sum(e_error)
            sample_i = torch.multinomial(
                input=sample_p.float(),
                num_samples=bs,
                replacement=True
            )
        # compute loss
        loss = torch.mean(self.criterion(out[sample_i], labels[sample_i]))
        # update info
        self.info = out
        return loss
    
    def fresh(self):
        self.info = None
    
    def predict(self, out_list):
        out_stack = torch.stack(out_list, dim=-1)
        out = torch.mean(out_stack, dim=-1)
        return out

class MSEEnsemble:
    def __init__(self, config, criterion):
        self.num_block = config.num_block
        self.rate = config.update_rate
        self.criterion = criterion
        self.beta = [0.5] * self.num_block

    def __call__(self, i, labels, out):
        # get max error
        e_max = torch.max(torch.abs(out - labels))
        # get relative error
        e_rel = (out - labels).pow(2) / (e_max.pow(2) + 1e-8)
        # get regressor total error
        if self.info is None:
            self.info = torch.ones_like(e_rel) / len(e_rel)
        e_ttl = torch.sum(self.info * e_rel)
        # get regressor weight
        beta = (e_ttl / (1 - e_ttl + 1e-8))
        # update information
        cum_ttl = torch.sum(self.info * torch.pow(beta, 1 - e_rel)) + 1e-8
        self.info = self.info * torch.pow(beta, 1 - e_rel) / cum_ttl
        self.info = self.info.detach()
        # update beta
        self.beta[i] = (1-self.rate) * self.beta[i] + self.rate * beta.item()
        tbwriter.add_scalar(f"cur-beta/{i}", beta)
        tbwriter.add_scalar(f"cum-beta/{i}", self.beta[i])
        # compute loss
        loss = torch.mean(self.criterion(out, labels))
        return loss
    
    def fresh(self):
        self.info = None
    
    def predict(self, out_list):
        out_stack = torch.stack(out_list, dim=-1)
        out = torch.mean(out_stack, dim=-1)
        return out
