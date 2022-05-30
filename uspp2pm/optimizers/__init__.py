# from ._default_optim import give_default_optimizer
from .deberta_split_optim import give_deberta_split_optimizer

_optim_dict = {}

def give_optim(model, config):
    _meta_optim = _optim_dict.get(config.model_name, give_deberta_split_optimizer)
    optimizer, scheduler = _meta_optim(model, config)
    return optimizer, scheduler