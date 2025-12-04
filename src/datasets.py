import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SalaryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: dict):
        self.descriptions = df.description.tolist()
        self.salaries = df.salary.tolist()

        self.log_target = config.get("log_target")
        self.max_length = config.get("max_length")
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer"))

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        tokenized_description = self.tokenizer(
            self.descriptions[idx],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        salary = torch.tensor(self.salaries[idx])

        return (
            tokenized_description["input_ids"][0],
            torch.log1p(salary) if self.log_target else salary,
            tokenized_description["attention_mask"][0],
        )
