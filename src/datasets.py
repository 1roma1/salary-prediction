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
        self.salaries = df.salary.tolist()

        self.max_experience = len(self.roles)
        self.max_employment = len(self.employments)
        self.max_role = len(self.roles)

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
            tokenized_description["input_ids"][0],
            features,
            torch.log1p(salary) if self.log_target else salary,
            tokenized_description["attention_mask"][0],
        )
