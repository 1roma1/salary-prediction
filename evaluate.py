import os
import mlflow
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from src.base.evaluator import Evaluator
from src.datasets import SalaryDataset
from src.base.utils import load_yaml


def get_argv() -> dict:
    parser = argparse.ArgumentParser(prog="Model Training")
    parser.add_argument(
        "-e", "--experiment", type=str, help="MLflow experiment name"
    )
    parser.add_argument(
        "-m", "--model", type=str, help="Model checkpoint file path"
    )
    parser.add_argument(
        "--train-config",
        type=str,
        help="Train config file path",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        help="Data config file path",
    )
    return vars(parser.parse_args())


def evaluate() -> None:
    argv = get_argv()
    config = load_yaml("configs/config.yaml")
    train_config = load_yaml(argv["train_config"])
    data_config = load_yaml(argv["data_config"])

    test_data_file = Path(config["preprocessed_data_dir"], config["test_data"])
    test_df = pd.read_csv(test_data_file)
    test_dataset = SalaryDataset(test_df, data_config)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))
    evaluator = Evaluator(test_dataset)
    evaluator.evaluate(
        argv["experiment"], argv["model"], train_config, data_config
    )


if __name__ == "__main__":
    load_dotenv()
    evaluate()
