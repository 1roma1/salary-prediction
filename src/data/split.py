import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.base.utils import load_yaml


def split(
    all_data_path: str,
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    val_test_size: int,
    test_size: int,
    random_state: int,
):
    df = pd.read_csv(all_data_path)
    df["bin_salary"] = pd.qcut(df.salary, q=10)

    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_size,
        random_state=random_state,
        stratify=df.bin_salary,
    )

    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_size,
        random_state=random_state,
        stratify=val_test_df.bin_salary,
    )
    train_df = train_df.drop(labels=["bin_salary"], axis=1)
    val_df = val_df.drop(labels=["bin_salary"], axis=1)
    test_df = test_df.drop(labels=["bin_salary"], axis=1)

    train_df.to_csv(train_data_path, index=False)
    val_df.to_csv(val_data_path, index=False)
    test_df.to_csv(test_data_path, index=False)
    os.remove(all_data_path)


if __name__ == "__main__":
    config = load_yaml("configs/config.yaml")
    data_path = Path(config["preprocessed_data_dir"])

    split(
        data_path / config["all_data"],
        data_path / config["train_data"],
        data_path / config["val_data"],
        data_path / config["test_data"],
        config["val_test_size"],
        config["test_size"],
        config["random_state"],
    )
