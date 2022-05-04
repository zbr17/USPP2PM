import torch
from transformers.models.deberta_v2 import DebertaV2Tokenizer

def give_deberta_tokenizer(config):
    cache_dir = config.model_path
    tokenizer = DebertaV2Tokenizer.from_pretrained(cache_dir)
    return tokenizer
