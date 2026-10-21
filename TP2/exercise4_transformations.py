from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from exercise1_csv_reader import CSVReader

DATA_DIR = Path(__file__).resolve().parent / "Data"
SAMPLE_CSV = DATA_DIR / "sample_data (2).csv"


class DataTransform(ABC):
    """One transformation that can be applied to a whole DataFrame."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the transformed DataFrame."""


class NormalizeColumns(DataTransform):
    IDENTIFIER_COLUMNS = ("id",)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric_columns = df.select_dtypes(include="number").columns
        value_columns = numeric_columns.drop(
            list(self.IDENTIFIER_COLUMNS), errors="ignore"
        )

        if len(value_columns) == 0:
            return df  # nothing to normalize

        means = df[value_columns].mean()
        stds = df[value_columns].std()

        # Convert scalar (float) → Series when there is only 1 column
        if not isinstance(stds, pd.Series):
            stds = pd.Series(stds, index=value_columns)

        # Avoid division by zero
        standard_deviations = stds.replace(0, 1)

        df[value_columns] = (df[value_columns] - means) / standard_deviations
        return df


class RemoveDuplicates(DataTransform):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()


class StandardizeText(DataTransform):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # pandas 3 stores text as the "string" dtype
        for column in df.select_dtypes(include="string"):
            df[column] = df[column].str.strip().str.title()
        return df


class TransformFactory:
    _REGISTRY: dict[str, type[DataTransform]] = {
        "normalize": NormalizeColumns,
        "dedup": RemoveDuplicates,
        "standardize": StandardizeText,
    }

    @staticmethod
    def get(name: str) -> DataTransform:
        try:
            return TransformFactory._REGISTRY[name.strip().lower()]()
        except KeyError:
            valid_names = ", ".join(sorted(TransformFactory._REGISTRY))
            raise ValueError(
                f"Unknown transform '{name}'. Choose from: {valid_names}"
            ) from None


def main() -> None:
    data = CSVReader(SAMPLE_CSV).read()
    print(f"Before: {len(data)} rows\n")

    transformed = data
    for name in ("standardize", "dedup", "normalize"):
        transform = TransformFactory.get(name)
        transformed = transform.apply(transformed)
        print(f"After {name:<12}: {len(transformed)} rows")

    print("\nResult:")
    print(transformed)


if __name__ == "__main__":
    main()
