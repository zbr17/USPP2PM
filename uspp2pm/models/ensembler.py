from ftplib import error_reply
import torch
import torch.nn as nn
from uspp2pm import tbwriter

class DefaultEnsemble(nn.Module):
    def __init__(self, config, criterion):
        super().__init__()
        self.criterion = criterion
    
    def forward(self, i, labels, out):
        # compute loss
        loss = torch.mean(self.criterion(out, labels))
        return loss
    
    def fresh(self):
        self.info = None
    
    def predict(self, out_list):
        out_stack = torch.stack(out_list, dim=-1).detach()
        out = torch.mean(out_stack, dim=-1)
        return out
    
class WeightEnsemble(nn.Module):
    def __init__(self, config, criterion):
        super().__init__()
        self.criterion = criterion
        self.bias = nn.Parameter(torch.ones(config.num_block) / config.num_block)
    
    def forward(self, i, labels, out):
        # compute loss
        loss = torch.mean(self.criterion(out, labels))
        return loss
    
    def fresh(self):
        self.info = None
    
    def predict(self, out_list):
        out_stack = torch.stack(out_list, dim=-1).detach()
        # out = torch.mean(out_stack, dim=-1)
        out = out_stack @ self.bias
        return out

class RandEnsemble(nn.Module):
    def __init__(self, config, criterion):
        super().__init__()
        self.criterion = criterion
        self.bias = nn.Parameter(torch.ones(config.num_block) / config.num_block)
    
    def forward(self, i, labels, out):
        bs = int(labels.size(0))
        # sampling probability
        sample_p = torch.ones(bs) / bs
        sample_i = torch.multinomial(
            input=sample_p.float(),
            num_samples=bs,
            replacement=True
        )

        # compute loss
        loss = torch.mean(self.criterion(out[sample_i], labels[sample_i]))
        return loss
    
    def fresh(self):
        self.info = None
    
    def predict(self, out_list):
        out_stack = torch.stack(out_list, dim=-1).detach()
        # out = torch.mean(out_stack, dim=-1)
        out = out_stack @ self.bias
        return out

class HardEnsemble(nn.Module):
    def __init__(self, config, criterion):
        super().__init__()
        self.criterion = criterion
        self.bias = nn.Parameter(torch.ones(config.num_block) / config.num_block)
    
    def forward(self, i, labels, out):
        bs = int(labels.size(0))
        # generate idx
        if self.info is None:
            loss = torch.mean(self.criterion(out, labels))
        else:
            e_error = (self.info - labels).pow(2)
            # sampling probability
            sort_idx = torch.sort(e_error, descending=False)[1]
            sample_p = sort_idx + 1
            sample_p = sample_p / torch.sum(sample_p)
            sample_i = torch.multinomial(
                input=sample_p.float(),
                num_samples=bs,
                replacement=True
            )
            # compute loss
            loss = torch.mean(self.criterion(
                out[sort_idx][sample_i], 
                labels[sort_idx][sample_i])
            )
        # update info
        self.info = out
        return loss
    
    def fresh(self):
        self.info = None
    
    def predict(self, out_list):
        out_stack = torch.stack(out_list, dim=-1).detach()
        # out = torch.mean(out_stack, dim=-1)
        out = out_stack @ self.bias
        return out

class AdaboostR2Ensemble:
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

_meta_ensembler = {
    "default": DefaultEnsemble,
    "weight": WeightEnsemble,
    "rand": RandEnsemble,
    "hard": HardEnsemble,
    "adaboostr2": AdaboostR2Ensemble
}

def give_ensembler(criterion, config):
    meta_class = _meta_ensembler[config.ensemble_name]
    return meta_class(config, criterion)
