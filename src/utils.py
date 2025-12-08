import json
import yaml
import pandas as pd

from pathlib import Path
from typing import Dict


def load_configuration(config_file: str) -> Dict:
    """Load configuration from yaml file"""

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_json(filename: str):
    """Load json data from file"""

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_schema(dictionary):
    result = {}
    for value in dictionary.values():
        result.update(value)
    return result


def get_X_y(df: pd.DataFrame, target: str):
    return df.drop(labels=target, axis=1), df[target]
