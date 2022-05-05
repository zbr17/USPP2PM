from .deberta_pp2p import DeBertaPP2PCombined, DeBertaPP2PSplit
from .bert_patents import BertPatentCombined

from .deberta_tokenizer import give_deberta_tokenizer
from .bert_tokenizer import give_bert_tokenizer

_model_dict = {
    "combined": {
        "deberta-v3-large": DeBertaPP2PCombined,
        "deberta-v3-base": DeBertaPP2PCombined,
        "bert-for-patents": BertPatentCombined
    },
    "split": {
        "deberta-v3-large": DeBertaPP2PSplit,
        "deberta-v3-base": DeBertaPP2PSplit,
    }
}

_tokenizer_dict = {
    "deberta-v3-large": give_deberta_tokenizer,
    "deberta-v3-base": give_deberta_tokenizer,
    "bert-for-patents": give_bert_tokenizer
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