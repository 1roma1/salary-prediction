import mlflow
import argparse
import pandas as pd

from src.trainer import Trainer
from src.datasets import SalaryDataset
from src.utils import load_configuration


def get_argv() -> dict:
    parser = argparse.ArgumentParser(prog="Model Training")
    parser.add_argument("-e", "--experiment", type=str, help="MLflow experiment name")
    parser.add_argument(
        "--train-config",
        type=str,
        help="Train config file",
    )

    return vars(parser.parse_args())


def train() -> None:
    argv = get_argv()
    config = load_configuration("configs/config.yaml")
    train_config = load_configuration(argv["train_config"])

    train_df = pd.read_csv("data/preprocessed/train_vacancies.csv")
    val_df = pd.read_csv("data/preprocessed/val_vacancies.csv")

    train_dataset = SalaryDataset(train_df, train_config["model_params"]["bert_model"])
    val_dataset = SalaryDataset(val_df, train_config["model_params"]["bert_model"])

    mlflow.set_tracking_uri(config["mlflow_uri"])
    trainer = Trainer(train_dataset, val_dataset)
    trainer.train(
        argv["experiment"], checkpoints_dir=config["chpts_dir"], train_config=train_config
    )


if __name__ == "__main__":
    train()
