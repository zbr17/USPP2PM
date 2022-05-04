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
    _max_seq_len = 512
    def __init__(self, config) -> None:
        # get config
        pretrained_id = _pretrained_id_dict[config.pretrain_name]
        # get tokenizer
        vocab_path, vocab_type = deberta.load_vocab(pretrained_id=pretrained_id)
        self.tokenizer = deberta.tokenizers[vocab_type](vocab_path)      

    def __call__(self, data):
        tokens = self.tokenizer.tokenize(data)
        # Truncate long sequence
        tokens = tokens[:self._max_seq_len - 2]
        # Add special tokens to the `tokens`
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # padding
        padding = self._max_seq_len - len(input_ids)
        input_ids = input_ids + [0] * padding
        input_mask = input_mask + [0] * padding
        features = {
            "input_ids": torch.tensor(input_ids, dtype=torch.int),
            "input_mask": torch.tensor(input_mask, dtype=torch.int)
        }
        return features
