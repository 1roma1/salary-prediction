import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from src.base.registries import ModelRegistry


@ModelRegistry.register("bert")
class BertModel(nn.Module):
    def __init__(self, bert_model, hid_size):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_model)
        self.feat_enc = nn.Linear(181, 64)
        self.head = nn.Linear(hid_size + 64, 1)

    def forward(self, descriptions, features, attention_mask=None):
        bert_out = self.bert(descriptions, attention_mask=attention_mask)[
            "last_hidden_state"
        ][:, 0, :]
        feat_enc_out = F.relu(self.feat_enc(features))
        return self.head(torch.concat((bert_out, feat_enc_out), dim=-1))
