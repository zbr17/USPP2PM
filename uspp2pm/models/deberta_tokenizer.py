from DeBERTa import deberta
import torch

# 'base' 'large' 'base-mnli' 'large-mnli' 'xlarge' 'xlarge-mnli' 'xlarge-v2' 'xxlarge-v2'
_pretrained_id_dict = {
    "deberta-v3-large": "large",
    "deberta-v3-base": "base"
}

class DebertaTokenizer:
    """
    Apply the same schema of special tokens as BERT, e.g. [CLS], [SEP], [MASK]
    """
    def __init__(self, config) -> None:
        # get config
        pretrained_id = _pretrained_id_dict[config.pretrain_name]
        cache_dir = config.model_path
        # get tokenizer
        vocab_path, vocab_type = deberta.load_vocab(pretrained_id=pretrained_id, cache_dir=cache_dir)
        self.tokenizer = deberta.tokenizers[vocab_type](vocab_path) 
        self.max_len = 512   

    def compute_tokens(self, data):  
        tokens = self.tokenizer.tokenize(data)
        # Truncate long sequence
        tokens = tokens[:self.max_len - 2]
        # Add special tokens to the `tokens`
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # padding
        padding = self.max_len - len(input_ids)
        input_ids = input_ids + [0] * padding
        input_mask = input_mask + [0] * padding

        return (
            torch.tensor(input_ids, dtype=torch.int), 
            torch.tensor(input_mask, dtype=torch.int)
        )

    def __call__(self, data, max_len=512):
        self.max_len = max_len
        output = {"input_ids": [], "attention_mask": []}
        for item in data:
            input_ids, input_mask = self.compute_tokens(item)
            output["input_ids"].append(input_ids)
            output["attention_mask"].append(input_mask)
        for k, v in output.items():
            output[k] = torch.stack(v, dim=0)
        return output
