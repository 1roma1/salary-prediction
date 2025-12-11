import os
import re
import pandas as pd

from nltk.corpus import stopwords
from pathlib import Path

from src.base.utils import load_yaml, load_json

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
    def __init__(self, features, config=None):
        self.features = features
        self.config = config

        self.experience = {
            feature: i for i, feature in enumerate(self.features["experience"])
        }
        self.employment = {
            feature: i for i, feature in enumerate(self.features["employment"])
        }
        self.role = {
            feature: i for i, feature in enumerate(self.features["role"])
        }

    def _normalize_salary(self, salary: str) -> float:
        currency_map = {
            "BYR": 1,
            "USD": 2.95,
            "EUR": 3.4473,
            "RUR": 0.036639,
            "KZT": 0.00057104,
        }

        salary, currency = salary.split()

        if "-" in salary:
            salary_from, salary_to = salary.split("-")
            return (
                (float(salary_from) + float(salary_to)) / 2
            ) * currency_map[currency]
        else:
            return float(salary) * currency_map[currency]

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        for func in (
            remove_html_tags,
            remove_urls,
            remove_non_alphanumeric,
            lambda x: x.replace("quot", ""),
            remove_spaces,
            remove_stop_words,
        ):
            text = func(text)
        return text

    def _map_experience(self, string: str):
        return self.experience.get(string)

    def _map_employment(self, string: str):
        return self.employment.get(string)

    def _map_role(self, string: str):
        return self.role.get(string)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[
            [
                "description",
                "title",
                "experience",
                "employment",
                "role",
                "salary",
            ]
        ]

        df = df.assign(
            salary=df.salary.apply(lambda x: self._normalize_salary(x)),
            description=df.description.apply(
                lambda x: self._normalize_text(x)
            ),
            title=df.title.apply(lambda x: self._normalize_text(x)),
            experience=df.experience.apply(lambda x: self._map_experience(x)),
            employment=df.employment.apply(lambda x: self._map_employment(x)),
            role=df.role.apply(lambda x: self._map_role(x)),
        )

        df = df[
            (df.salary > self.config["min_salary"])
            & (df.salary < self.config["max_salary"])
        ]

        return df


if __name__ == "__main__":
    config = load_yaml("configs/config.yaml")
    preprocessing_config = load_yaml("configs/preprocessing_config.yaml")
    features = load_json("data/features.json")

    raw_data = pd.read_csv(Path(config["raw_data_dir"]) / config["all_data"])
    print(f"Raw data shape: {raw_data.shape}")

    preprocessor = Preprocessor(features, preprocessing_config)
    preprocessed_data = preprocessor.preprocess(raw_data)
    print(f"Preprocessed data shape: {preprocessed_data.shape}")

    os.makedirs(config["preprocessed_data_dir"], exist_ok=True)
    preprocessed_data.to_csv(
        Path(config["preprocessed_data_dir"]) / config["all_data"],
        index=False,
    )
