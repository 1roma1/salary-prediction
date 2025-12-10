import os
import mlflow
import argparse
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

from src.base.trainer import Trainer
from src.datasets import SalaryDataset
from src.utils import load_yaml


def get_argv() -> dict:
    parser = argparse.ArgumentParser(prog="Model Training")
    parser.add_argument(
        "-e", "--experiment", type=str, help="MLflow experiment name"
    )
    parser.add_argument(
        "--train-config",
        type=str,
        help="Train config file",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        help="Data config file",
    )

    return vars(parser.parse_args())


def train() -> None:
    argv = get_argv()
    config = load_yaml("configs/config.yaml")
    train_config = load_yaml(argv["train_config"])
    data_config = load_yaml(argv["data_config"])

    train_data_file = Path(
        config["preprocessed_data_dir"], config["train_data"]
    )
    val_data_file = Path(config["preprocessed_data_dir"], config["val_data"])

    train_df = pd.read_csv(train_data_file)
    val_df = pd.read_csv(val_data_file)

    train_dataset = SalaryDataset(train_df, data_config)
    val_dataset = SalaryDataset(val_df, data_config)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))
    trainer = Trainer(train_dataset, val_dataset)
    trainer.train(
        argv["experiment"],
        checkpoints_dir=config["chpts_dir"],
        train_config=train_config,
        data_config=data_config,
    )


if __name__ == "__main__":
    load_dotenv()
    train()
