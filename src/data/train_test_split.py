import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_configuration


def split(
    raw_data_path: str,
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    val_size: int,
    test_size: int,
    random_state: int,
):
    df = pd.read_csv(raw_data_path)

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
    )

    train_df, test_df = train_test_split(
        train_df,
        test_size=test_size,
        random_state=random_state,
    )

    train_df.to_csv(train_data_path, index=False)
    val_df.to_csv(val_data_path, index=False)
    test_df.to_csv(test_data_path, index=False)
    os.remove(raw_data_path)


if __name__ == "__main__":
    config = load_configuration("configs/config.yaml")
    data_path = Path(config["raw_data_dir"])

    split(
        data_path / config["raw_data"],
        data_path / config["train_data"],
        data_path / config["val_data"],
        data_path / config["test_data"],
        config["val_size"],
        config["test_size"],
        config["random_state"],
    )
