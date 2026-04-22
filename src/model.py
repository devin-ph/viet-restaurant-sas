import torch
import torch.nn as nn
from transformers import AutoModel

class PhoBERTABSA(nn.Module):
    def __init__(self, model_name, num_sentiments, num_aspects, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        hidden_size = self.bert.config.hidden_size

        self.sentiment_head = nn.Linear(hidden_size, num_sentiments)
        self.aspect_head = nn.Linear(hidden_size, num_aspects)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = output.last_hidden_state[:, 0, :]
        cls_token = self.dropout(cls_token)

        sentiment_logits = self.sentiment_head(cls_token)
        aspect_logits = self.aspect_head(cls_token)

        return sentiment_logits, aspect_logits