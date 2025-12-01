import pandas as pd

from pathlib import Path

from pydantic import BaseModel
from pandantic import Pandantic

from src.utils import load_configuration


class DataSchema(BaseModel):
    title: str
    description: str
    salary: float


def validate_data(df: pd.DataFrame):
    validator = Pandantic(schema=DataSchema)
    validator.validate(df, errors="raise")


if __name__ == "__main__":
    config = load_configuration("configs/config.yaml")

    train_data_path = Path(config["preprocessed_data_dir"]) / config["train_data"]
    val_data_path = Path(config["preprocessed_data_dir"]) / config["val_data"]
    test_data_path = Path(config["preprocessed_data_dir"]) / config["test_data"]

    train = pd.read_csv(train_data_path)
    val = pd.read_csv(val_data_path)
    test = pd.read_csv(test_data_path)

    for path, df in zip((train_data_path, val_data_path, test_data_path), (train, val, test)):
        try:
            validate_data(df)
            print(f"{path} is successfully validated")
        except ValueError as e:
            print(f"Validation error: {e}")
