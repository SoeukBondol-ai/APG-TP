from __future__ import annotations

import functools
import time
from pathlib import Path

import pandas as pd
from exercise1_csv_reader import CSVReader
from exercise2_missing_values import DataCleaner, FillMean

DATA_DIR = Path(__file__).resolve().parent / "Data"
SAMPLE_CSV = DATA_DIR / "sample_data (2).csv"


def log_action(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[start] {func.__qualname__}")
        result = func(*args, **kwargs)
        print(f"[done]  {func.__qualname__}")
        return result

    return wrapper


def log_time(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        started_at = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_seconds = time.perf_counter() - started_at
        print(f"[time]  {func.__qualname__}: {elapsed_seconds:.3f}s")
        return result

    return wrapper


class LoggedCSVReader(CSVReader):
    @log_action
    @log_time
    def read(self) -> pd.DataFrame:
        return super().read()


class LoggedDataCleaner(DataCleaner):
    @log_action
    @log_time
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return super().clean(df)


def main() -> None:
    reader = LoggedCSVReader(SAMPLE_CSV)
    cleaner = LoggedDataCleaner(FillMean())

    data = reader.read()
    cleaned = cleaner.clean(data)

    print(
        f"\nMissing values: {int(data.isna().sum().sum())} "
        f"-> {int(cleaned.isna().sum().sum())}"
    )


if __name__ == "__main__":
    main()
