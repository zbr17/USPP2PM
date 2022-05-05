import torch
from transformers.models.bert.tokenization_bert import BertTokenizer

def give_bert_tokenizer(config):
    cache_dir = config.model_path
    tokenizer = BertTokenizer.from_pretrained(cache_dir)
    return tokenizer
