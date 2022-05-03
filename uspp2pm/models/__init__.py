from .deberta_pp2p import DeBertaPP2P

from transformers import AutoTokenizer

_model_dict = {
    "deberta-v3-large": DeBertaPP2P
}

_tokenizer_dict = {
    "deberta-v3-large": AutoTokenizer
}

def give_model(pretrain_name: str, model_config: dict):
    model = _model_dict[pretrain_name](**model_config)
    return model

def give_tokenizer(pretrain_name: str, model_path: str):
    tokenizer = _tokenizer_dict[pretrain_name].from_pretrained(model_path)
    return tokenizer