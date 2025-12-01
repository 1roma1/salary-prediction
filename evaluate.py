import torch
import torch.nn as nn
import pandas as pd

from transformers import AutoTokenizer
from src.models.bert import BertModel

bert_model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

model = BertModel()
model.load_state_dict(torch.load("chpts/bert-adaptable-sloth-250.pth", weights_only=True))
model.eval()


str = "В связи с увеличением объема оказываемых услуг наша компания приглашает к себе в команду ПРОГРАММИСТА C ASP.NET"
tokenized = tokenizer(
    str,
    return_tensors="pt",
    padding="max_length",
    max_length=512,
    truncation=True,
)

y = model(tokenized["input_ids"], tokenized["attention_mask"])

print(y)
print(torch.expm1(y))
