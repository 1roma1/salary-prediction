import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from src.base.registries import ModelRegistry


@ModelRegistry.register("bert")
class BertModel(nn.Module):
    def __init__(
        self, bert_model, bert_hid_size, cat_size, lin_hid_size, dropout=0
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_model)
        self.feat_enc = nn.Linear(cat_size, lin_hid_size)
        self.head = nn.Linear(bert_hid_size + lin_hid_size, 1)

        self.dropout = nn.Dropout(p=dropout)
        self.feat_bn = nn.BatchNorm1d(lin_hid_size)

    def forward(
        self,
        descriptions,
        features,
        mask=None,
    ):
        description_emb = self.bert(descriptions, attention_mask=mask)[
            "last_hidden_state"
        ][:, 0, :]
        feat_enc_out = self.dropout(
            F.relu(self.feat_bn(self.feat_enc(features)))
        )

        return self.head(torch.concat((description_emb, feat_enc_out), dim=-1))
