import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SalaryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: dict):
        self.experiences = df.experience.tolist()
        self.employments = df.employment.tolist()
        self.roles = df.role.tolist()
        self.descriptions = df.description.tolist()
        self.titles = df.title.tolist()
        self.salaries = df.salary.tolist()

        self.max_experience = 4
        self.max_employment = 3
        self.max_role = 174

        self.log_target = config.get("log_target")
        self.max_length = config.get("max_length")
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer"))

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        tokenized_text = self.tokenizer(
            self.titles[idx] + " " + self.descriptions[idx],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        salary = torch.tensor(self.salaries[idx])

        features = torch.concat(
            (
                F.one_hot(
                    torch.tensor(self.experiences[idx]),
                    num_classes=self.max_experience,
                ),
                F.one_hot(
                    torch.tensor(self.employments[idx]),
                    num_classes=self.max_employment,
                ),
                F.one_hot(
                    torch.tensor(self.roles[idx]),
                    num_classes=self.max_role,
                ),
            ),
        ).type(torch.float32)

        return (
            tokenized_text["input_ids"][0],
            features,
            torch.log1p(salary) if self.log_target else salary,
            tokenized_text["attention_mask"][0],
        )
