from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

if __package__:
    from .exercise1_csv_reader import CSVReader
    from .exercise2_missing_values import (
        DataCleaner,
        DropMissing,
        FillMean,
        FillMode,
        MissingValueStrategy,
    )
    from .exercise3_decorators import log_action, log_time
    from .exercise4_transformations import DataTransform, TransformFactory
else:
    from exercise1_csv_reader import CSVReader
    from exercise2_missing_values import (
        DataCleaner,
        DropMissing,
        FillMean,
        FillMode,
        MissingValueStrategy,
    )
    from exercise3_decorators import log_action, log_time
    from exercise4_transformations import DataTransform, TransformFactory

DATA_DIR = Path(__file__).resolve().parent / "Data"
SAMPLE_CSV = DATA_DIR / "sample_data (2).csv"
CLEANED_CSV = DATA_DIR / "my_data_clean.csv"


class DataPipeline(ABC):
    def run(self) -> pd.DataFrame:
        """Execute load -> clean -> transform -> save."""
        data = self.load()
        data = self.clean(data)
        data = self.transform(data)
        self.save(data)
        return data

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Read the raw data."""

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured transformations."""

    @abstractmethod
    def save(self, df: pd.DataFrame) -> None:
        """Persist the cleaned data."""


class CSVDataPipeline(DataPipeline):
    _STRATEGIES: dict[str, type[MissingValueStrategy]] = {
        "drop": DropMissing,
        "mean": FillMean,
        "mode": FillMode,
    }

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
        missing_strategy: str = "mean",
        transforms: tuple[str, ...] = ("standardize", "dedup"),
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.reader = CSVReader(self.input_path)
        self.cleaner = DataCleaner(self._create_strategy(missing_strategy))
        self.transforms: list[DataTransform] = [
            TransformFactory.get(name) for name in transforms
        ]

    @classmethod
    def _create_strategy(cls, name: str) -> MissingValueStrategy:
        try:
            return cls._STRATEGIES[name.strip().lower()]()
        except KeyError:
            valid_names = ", ".join(sorted(cls._STRATEGIES))
            raise ValueError(
                f"Unknown missing-value strategy '{name}'. Choose from: {valid_names}"
            ) from None

    @log_action
    @log_time
    def load(self) -> pd.DataFrame:
        return self.reader.read()

    @log_action
    @log_time
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.cleaner.clean(df)

    @log_action
    @log_time
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for transform in self.transforms:
            df = transform.apply(df)
        return df

    @log_action
    def save(self, df: pd.DataFrame) -> None:
        df.to_csv(self.output_path, index=False)
        print(f"Saved {len(df)} rows to {self.output_path}")


def main() -> None:
    pipeline = CSVDataPipeline(
        input_path=SAMPLE_CSV,
        output_path=CLEANED_CSV,
        missing_strategy="mean",
        transforms=("standardize", "dedup", "normalize"),
    )
    cleaned = pipeline.run()

    print("\nCleaned data:")
    print(cleaned)


if __name__ == "__main__":
    main()
