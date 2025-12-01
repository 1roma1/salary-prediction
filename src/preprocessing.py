import os
import re
import pandas as pd

from nltk.corpus import stopwords
from pathlib import Path

from src.utils import load_configuration

STOP_WORDS = stopwords.words("russian") + stopwords.words("english")


def remove_html_tags(text: str) -> str:
    """
    Removes HTML tags from a string.
    """
    return re.sub(r"<.*?>", "", text)


def remove_urls(text: str) -> str:
    """
    Removes URLs from a string.
    """
    return re.sub(r"http[s]?://\S+", "", text)


def remove_non_alphanumeric(text: str) -> str:
    """
    Removes all non-alphanumeric characters from a string.
    """
    return re.sub(r"[^a-zA-Z0-9а-яА-Я]", " ", text)


def remove_spaces(text: str) -> str:
    """
    Removes all extra spaces from a string.
    """
    return re.sub(" +", " ", text.strip())


def remove_stop_words(text: str) -> str:
    """
    Removes stop words from a string
    """
    return " ".join([word for word in text.split() if word not in STOP_WORDS])


class Preprocessor:
    def __init__(self, config=None):
        self.config = config

    def _normalize_salary(self, salary: str) -> float:
        currency_map = {"BYR": 1, "USD": 2.9743, "EUR": 3.4473, "RUR": 0.036639, "KZT": 0.00057104}

        salary, currency = salary.split()

        if "-" in salary:
            salary_from, salary_to = salary.split("-")
            return ((float(salary_from) + float(salary_to)) / 2) * currency_map[currency]
        else:
            return float(salary) * currency_map[currency]

    def _normalize_text(self, text: str) -> str:
        # text = text.lower()
        for func in (
            remove_html_tags,
            remove_urls,
            remove_non_alphanumeric,
            lambda x: x.replace("quot", ""),
            remove_spaces,
            # remove_stop_words,
        ):
            text = func(text)
        return text

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[["description", "salary"]]

        df = df.assign(
            salary=df.salary.apply(lambda x: self._normalize_salary(x)),
            # title=df.title.apply(lambda x: self._normalize_text(x)),
            description=df.description.apply(lambda x: self._normalize_text(x)),
        )

        return df


if __name__ == "__main__":
    config = load_configuration("configs/config.yaml")
    preprocessing_config = load_configuration("configs/preprocessing_config.yaml")

    raw_train = pd.read_csv(Path(config["raw_data_dir"]) / config["train_data"])
    raw_val = pd.read_csv(Path(config["raw_data_dir"]) / config["val_data"])
    raw_test = pd.read_csv(Path(config["raw_data_dir"]) / config["test_data"])

    print(
        f"Raw data shape: Train - {raw_train.shape} Val - {raw_val.shape} Test - {raw_test.shape}"
    )

    preprocessor = Preprocessor(preprocessing_config)
    preprocessed_train = preprocessor.preprocess(raw_train)
    preprocessed_val = preprocessor.preprocess(raw_val)
    preprocessed_test = preprocessor.preprocess(raw_test)

    print(
        f"Preprocessed data shape: Train - {preprocessed_train.shape} Val - {preprocessed_val.shape} Test - {preprocessed_test.shape}"
    )

    os.makedirs(config["preprocessed_data_dir"], exist_ok=True)
    preprocessed_train.to_csv(
        Path(config["preprocessed_data_dir"]) / config["train_data"],
        index=False,
    )
    preprocessed_val.to_csv(
        Path(config["preprocessed_data_dir"]) / config["val_data"],
        index=False,
    )
    preprocessed_test.to_csv(
        Path(config["preprocessed_data_dir"]) / config["test_data"],
        index=False,
    )
