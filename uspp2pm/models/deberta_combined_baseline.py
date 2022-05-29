from typing import Optional
from matplotlib.pyplot import isinteractive
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from transformers.models.deberta_v2 import DebertaV2ForSequenceClassification

class DeBertaCombinedBaseline(nn.Module):
    """
    Without dropout in the final layer.
    """
    def __init__(
        self, 
        config
    ):
        super().__init__()
        # get args
        cache_dir = config.model_path
        # initialize
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=cache_dir,
            num_labels=1
        )
    
    def forward(
        self,
        data_info: dict,
    ) -> torch.Tensor:
        input_ids = data_info["inputs"]["input_ids"]
        attention_mask = data_info["inputs"]["attention_mask"]
        token_type_ids = data_info["inputs"]["token_type_ids"]

        outputs = self.model.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        encoder_layer = outputs[0]
        pooled_output = self.model.pooler(encoder_layer)
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        logits = torch.sigmoid(logits)
        return logits.squeeze()
