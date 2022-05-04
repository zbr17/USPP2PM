from .deberta_pp2p import DeBertaPP2PCombined

from .deberta_tokenizer import DebertaTokenizer

_model_dict = {
    "combined": {
        "deberta-v3-large": DeBertaPP2PCombined,
        "deberta-v3-base": DeBertaPP2PCombined,
    }
}



_tokenizer_dict = {
    "deberta-v3-large": DebertaTokenizer,
    "deberta-v3-base": DebertaTokenizer
}

def give_tokenizer(config):
    _meta_tokenizer_class = _tokenizer_dict[config.pretrain_name]
    tokenizer = _meta_tokenizer_class(config)
    return tokenizer

def give_model(config):
    # get args
    dataset_name = config.dataset_name
    pretrain_name = config.pretrain_name
    # initialize
    _meta_model = _model_dict[dataset_name][pretrain_name]
    model = _meta_model(config)
    return model