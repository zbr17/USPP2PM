import torch

def _init_tokenizer(meta_class, config):
    cache_dir = config.model_path
    tokenizer = meta_class.from_pretrained(cache_dir)
    return tokenizer
