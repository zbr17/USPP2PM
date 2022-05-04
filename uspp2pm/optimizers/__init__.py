from ._default_optim import give_default_optimizer

_optim_dict = {
    "deberta-v3-large": give_default_optimizer,
    "deberta-v3-base": give_default_optimizer
}

def give_optim(model, config):
    _meta_optim = _optim_dict[config.pretrain_name]
    optimizer, scheduler = _meta_optim(model, config)
    return optimizer, scheduler