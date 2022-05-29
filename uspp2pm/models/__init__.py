from .deberta_combined_baseline import DeBertaCombinedBaseline
from .deberta_split_baseline import DeBertaSplitBaseline
from .deberta_split_similarity import DeBertaSplitSimilarity
# from .bert_patents import BertPatentCombined

from .deberta_tokenizer import give_deberta_tokenizer
# from .bert_tokenizer import give_bert_tokenizer

from .losses import give_criterion

_model_dict = {
    "combined_baseline": {
        "deberta-v3-large": DeBertaCombinedBaseline,
        "deberta-v3-base": DeBertaCombinedBaseline,
    },
    "split_baseline": {
        "deberta-v3-large": DeBertaSplitBaseline,
        "deberta-v3-base": DeBertaSplitBaseline,
    },
    "split_similarity": {
        "deberta-v3-large": DeBertaSplitSimilarity,
        "deberta-v3-base": DeBertaSplitSimilarity,
    }
}

_tokenizer_dict = {
    "deberta-v3-large": give_deberta_tokenizer,
    "deberta-v3-base": give_deberta_tokenizer,
}

def give_tokenizer(config):
    _meta_tokenizer_class = _tokenizer_dict[config.pretrain_name]
    tokenizer = _meta_tokenizer_class(config)
    return tokenizer

def give_model(config):
    # get args
    dataset_name = config.dataset_name
    model_name = config.model_name
    pretrain_name = config.pretrain_name
    assert dataset_name in model_name
    # get criterion
    criterion = give_criterion(config)
    # initialize
    _meta_model = _model_dict[model_name][pretrain_name]
    model = _meta_model(criterion, config)
    return model