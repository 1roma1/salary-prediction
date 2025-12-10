import json
import yaml

from typing import Dict


def load_yaml(config_file: str) -> Dict:
    """Load configuration from yaml file"""

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_json(filename: str):
    """Load json data from file"""

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
