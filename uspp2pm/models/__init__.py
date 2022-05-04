from .deberta_pp2p import DeBertaPP2P

from .deberta_tokenizer import DebertaTokenizer

_model_dict = {
    "deberta-v3-large": DeBertaPP2P
}

_tokenizer_dict = {
    "deberta-v3-large": DebertaTokenizer
}

def give_tokenizer(config):
    _meta_tokenizer_class = _tokenizer_dict[config.pretrain_name]
    tokenizer = _meta_tokenizer_class(config)
    return tokenizer

def give_model(pretrain_name: str, model_config: dict):
    model = _model_dict[pretrain_name](**model_config)
    raise NotImplementedError
    return model