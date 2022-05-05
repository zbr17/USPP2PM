from ._default_optim import give_default_optimizer
from .deberta_split_optim import give_deberta_split_optimizer



_optim_dict = {
    "combined": {
        "deberta-v3-large": give_default_optimizer,
        "deberta-v3-base": give_default_optimizer,
        "bert-for-patents": give_default_optimizer
    },
    "split": {
        "deberta-v3-large": give_deberta_split_optimizer,
        "deberta-v3-base": give_deberta_split_optimizer
    }
}

def give_optim(model, config):
    _meta_optim = _optim_dict[config.dataset_name][config.pretrain_name]
    optimizer, scheduler = _meta_optim(model, config)
    return optimizer, scheduler