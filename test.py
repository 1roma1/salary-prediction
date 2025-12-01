import torch
import torch.nn as nn
import pandas as pd

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader


class SalaryDataset(Dataset):
    def __init__(self, df):
        self.descriptions = df.description.tolist()
        self.salaries = df.salary.tolist()

        bert_model_name = "distilbert-base-multilingual-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        tokenized_description = self.tokenizer(
            self.descriptions[idx],
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True,
        )
        salary = torch.tensor(self.salaries[idx])

        return (
            tokenized_description["input_ids"][0],
            torch.log1p(salary),
            tokenized_description["attention_mask"][0],
        )


class BertModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
        self.head = nn.Linear(768, 1)

    def forward(self, x, attention_mask=None):
        x = self.bert(x, attention_mask=attention_mask)["last_hidden_state"][:, 0, :]
        return self.head(x)


if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed/train_vacancies.csv")

    dataset = SalaryDataset(df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    model = BertModel().to("cuda")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for texts, targets, attention_masks in dataloader:
        texts, targets, attention_masks = (
            texts.to("cuda"),
            targets.to("cuda"),
            attention_masks.to("cuda"),
        )
        model.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = model(texts, attention_mask=attention_masks)
            loss = loss_fn(y.view(-1), targets.view(-1))
        loss.backward()
        optimizer.step()

        print(loss.item() / 64)
