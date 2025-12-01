import torch.nn as nn
from transformers import AutoModel


class BertModel(nn.Module):
    def __init__(self, bert_model):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_model)
        self.head = nn.Linear(256, 1)

    def forward(self, x, attention_mask=None):
        x = self.bert(x, attention_mask=attention_mask)["last_hidden_state"][:, 0, :]
        return self.head(x)
