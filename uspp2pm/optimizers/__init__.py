# from ._default_optim import give_default_optimizer
from .deberta_split_optim import give_deberta_split_optimizer



_optim_dict = {
    "combined_baseline": give_deberta_split_optimizer,
    "split_baseline": give_deberta_split_optimizer,
    "split_similarity": give_deberta_split_optimizer
}

def give_optim(model, config):
    _meta_optim = _optim_dict[config.model_name]
    optimizer, scheduler = _meta_optim(model, config)
    return optimizer, scheduler