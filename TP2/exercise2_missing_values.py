from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from exercise1_csv_reader import CSVReader

DATA_DIR = Path(__file__).resolve().parent / "Data"
SAMPLE_CSV = DATA_DIR / "sample_data (2).csv"


class MissingValueStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with the missing values handled."""


class DropMissing(MissingValueStrategy):
    """Remove every row that contains a missing value."""

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()


class FillMean(MissingValueStrategy):
    """Fill numeric gaps with the column mean.

    Text columns are left untouched because a mean does not exist for them.
    """

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        column_means = df.mean(numeric_only=True)
        return df.fillna(column_means)


class FillMode(MissingValueStrategy):
    """Fill each column's gaps with that column's most frequent value."""

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        filled = df.copy()
        for column in filled.columns:
            most_common_values = filled[column].mode()
            if not most_common_values.empty:
                filled[column] = filled[column].fillna(most_common_values.iloc[0])
        return filled


class DataCleaner:
    def __init__(self, strategy: MissingValueStrategy) -> None:
        self.strategy = strategy

    def set_strategy(self, strategy: MissingValueStrategy) -> None:
        self.strategy = strategy

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.strategy.handle(df)


def main() -> None:
    data = CSVReader(SAMPLE_CSV).read()
    missing_before = int(data.isna().sum().sum())
    print(f"Missing values before: {missing_before}\n")

    cleaner = DataCleaner(DropMissing())
    for strategy in (DropMissing(), FillMean(), FillMode()):
        cleaner.set_strategy(strategy)
        cleaned = cleaner.clean(data)
        missing_after = int(cleaned.isna().sum().sum())
        print(
            f"{type(strategy).__name__:<12} "
            f"-> {len(cleaned)} rows, {missing_after} missing left"
        )


if __name__ == "__main__":
    main()
