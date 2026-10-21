from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "Data"
SAMPLE_CSV = DATA_DIR / "sample_data (2).csv"


class CSVReader:
    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)

    @property
    def file_path(self) -> Path:
        return self._file_path

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self._file_path)

    def preview(self, n: int = 5) -> None:
        print(self.read().head(n))


def main() -> None:
    reader = CSVReader(SAMPLE_CSV)
    print(f"Reading: {reader.file_path}")
    reader.preview(3)

    data = reader.read()
    row_count, column_count = data.shape
    print(f"\nShape: {row_count} rows x {column_count} columns")


if __name__ == "__main__":
    main()
