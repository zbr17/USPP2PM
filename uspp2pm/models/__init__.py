from transformers.models.deberta_v2 import DebertaV2Model
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.bert.modeling_bert import BertForSequenceClassification

from transformers.models.deberta_v2 import DebertaV2Tokenizer
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer

from .combined_baseline import CombinedBaseline
from .split_baseline import SplitBaseline
from .split_similarity import SplitSimilarity
from .combined_hdc import CombinedHDC

from .tokenizer import _init_tokenizer
from .losses import give_criterion

_model_dict = {
    "combined_baseline": CombinedBaseline,
    "split_baseline": SplitBaseline,
    "split_similarity": SplitSimilarity,
    "combined_hdc": CombinedHDC
}

_pretrain_dict = {
    "deberta-v3-base": DebertaV2Model,
    "deberta-v3-large": DebertaV2Model,
    "roberta-base": RobertaModel,
    "bert-for-patents": BertForSequenceClassification
}

_tokenizer_dict = {
    "deberta-v3-large": DebertaV2Tokenizer,
    "deberta-v3-base": DebertaV2Tokenizer,
    "roberta-base": RobertaTokenizer,
    "bert-for-patents": BertTokenizer,
}

def give_tokenizer(config):
    _meta_tokenizer_class = _tokenizer_dict[config.pretrain_name]
    tokenizer = _init_tokenizer(_meta_tokenizer_class, config)
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
    _pretrain = _pretrain_dict[pretrain_name]
    _meta_model = _model_dict[model_name]
    model = _meta_model(criterion, _pretrain, config)
    return model