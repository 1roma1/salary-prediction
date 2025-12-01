import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SalaryDataset(Dataset):
    def __init__(self, df, bert_model_name):
        self.descriptions = df.description.tolist()
        self.salaries = df.salary.tolist()

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        tokenized_description = self.tokenizer(
            self.descriptions[idx],
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True,
        )
        salary = torch.tensor(self.salaries[idx])

        return (
            tokenized_description["input_ids"][0],
            torch.log1p(salary),
            tokenized_description["attention_mask"][0],
        )
