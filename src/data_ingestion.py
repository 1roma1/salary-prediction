import os
import sqlite3
import argparse
import pandas as pd

from pathlib import Path

from src.utils import load_configuration


def fetch_data_from_db(date: str) -> None:
    config = load_configuration("configs/config.yaml")

    vacancy_sql_stmt = """
    SELECT id, date(substr(published_at, 0, 11)) as date, title, description, company, employment, experience, salary
    FROM vacancy 
    WHERE date(substr(published_at, 0, 11)) < date("{date}") and salary NOT NULL
    """.format(date=date)

    with sqlite3.connect(config["data_source"]) as conn:
        data = pd.read_sql(vacancy_sql_stmt, conn)
    os.makedirs(config["raw_data_dir"], exist_ok=True)
    data.to_csv(
        Path(config["raw_data_dir"]) / config["raw_data"],
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Data Ingestion")
    parser.add_argument("--date", type=str)

    fetch_data_from_db(vars(parser.parse_args())["date"])
