import argparse
import logging
from pathlib import Path
import re
from typing import Union

import pandas as pd
import polars as pl

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class CSVAggregator:
    """
    Aggregates multiple CSV files in a directory into a single Polars DataFrame.
    Ensures consistent column structure, optionally casts strings to floats,
    and adds metadata columns.
    """

    FLOAT_LIKE_PATTERN = r"^\s*[-+]?(?:\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$"

    def __init__(self, root_dir: Union[str, Path], csv_name: str = "*.csv"):
        """
        Initialize the aggregator with a directory and optional filename pattern.

        Args:
            root_dir (str | Path): Root directory to search for CSV files.
            csv_name (str): Glob pattern for CSV filenames. Defaults to "*.csv".
        """
        self.root_dir = Path(root_dir)
        self.csv_name = csv_name
        self.df = self._load_all()

    def _load_all(self) -> pl.DataFrame:
        """
        Load and combine all matching CSV files into a single DataFrame.

        Returns:
            pl.DataFrame: Aggregated and harmonized DataFrame.
        """
        all_columns = self._collect_all_columns()
        dataframes = []

        for file in self.root_dir.rglob(self.csv_name):
            try:
                df = pl.read_csv(file)
                df = self._harmonize_dataframe(df, all_columns, file)
                dataframes.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {file}: {e}")

        if not dataframes:
            logger.warning("No valid CSV files were found.")
            return pl.DataFrame()

        return pl.concat(dataframes, how="vertical_relaxed")

    def _collect_all_columns(self) -> list[str]:
        """
        Scan all CSVs to determine the complete set of column names.

        Returns:
            list[str]: Union of all column names found in the CSV files.
        """
        all_columns = set()
        for file in self.root_dir.rglob(self.csv_name):
            try:
                df = pl.read_csv(file, n_rows=1)
                all_columns.update(df.columns)
            except Exception as e:
                logger.warning(f"Skipping {file} due to error: {e}")
        return list(all_columns)

    def _harmonize_dataframe(
        self, df: pl.DataFrame, all_columns: list[str], file: Path
    ) -> pl.DataFrame:
        """
        Ensure the DataFrame matches the structure of the aggregate.

        Args:
            df (pl.DataFrame): Input DataFrame from a single CSV.
            all_columns (list[str]): Full list of expected columns.
            file (Path): Path to the source file.

        Returns:
            pl.DataFrame: Harmonized DataFrame.
        """
        df = self._cast_string_columns_to_float(df)
        df = self._add_missing_columns(df, all_columns)
        df = self._add_metadata_column(df, file)
        df = self._reorder_columns(df, all_columns)
        return df

    def _cast_string_columns_to_float(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Attempt to cast string columns to float if they appear numeric.

        Args:
            df (pl.DataFrame): The DataFrame to cast.

        Returns:
            pl.DataFrame: Updated DataFrame with casted columns.
        """
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                df = df.with_columns(
                    df[col]
                    .str.strip_chars()
                    .cast(pl.Float64, strict=False)
                    .fill_null(0.0)
                    .alias(col)
                )
        return df

    def _add_missing_columns(
        self, df: pl.DataFrame, all_columns: list[str]
    ) -> pl.DataFrame:
        """
        Add any columns that are missing from this DataFrame, filled with 0.0.

        Args:
            df (pl.DataFrame): The DataFrame to update.
            all_columns (list[str]): Complete list of expected columns.

        Returns:
            pl.DataFrame: DataFrame with missing columns added.
        """
        missing = set(all_columns) - set(df.columns)
        for col in missing:
            df = df.with_columns(pl.lit(0.0).alias(col))
        return df

    def _add_metadata_column(self, df: pl.DataFrame, file: Path) -> pl.DataFrame:
        """
        Add a column with the name of the parent directory for tracking.

        Args:
            df (pl.DataFrame): The DataFrame to update.
            file (Path): Path to the source file.

        Returns:
            pl.DataFrame: Updated DataFrame with 'wandb_name' column.
        """
        return df.with_columns(pl.lit(str(file.parent.name)).alias("wandb_name"))

    def _reorder_columns(
        self, df: pl.DataFrame, all_columns: list[str]
    ) -> pl.DataFrame:
        """
        Reorder columns for consistency and better readability.

        Args:
            df (pl.DataFrame): DataFrame with all expected columns.
            all_columns (list[str]): List of all known column names.

        Returns:
            pl.DataFrame: Reordered DataFrame.
        """
        op_cols = [c for c in all_columns if re.search(r"op_(\d+)", c)]
        other_cols = [c for c in all_columns if c not in op_cols and c != "wandb_name"]

        op_cols_sorted = sorted(
            op_cols,
            key=lambda c: (
                self._is_ope_coeff(c),
                self._extract_spin_number(c) or float("inf"),
                self._extract_op_number(c) or float("inf"),
            ),
        )

        ordered = (["wandb_name"] if "wandb_name" in df.columns else []) + op_cols_sorted + other_cols
        return df.select(ordered)

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the internal Polars DataFrame to a Pandas DataFrame.

        Returns:
            pd.DataFrame: The converted DataFrame.
        """
        return self.df.to_pandas()

    def save_csv(self, csv_name: Path) -> None:
        """
        Save the aggregated DataFrame to a CSV file.

        Args:
            csv_name (Path): Output path for the CSV file.
        """
        try:
            self.df.write_csv(csv_name)
            logger.info(f"Saved CSV to {csv_name.resolve()}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    @staticmethod
    def _extract_op_number(colname: str) -> int | None:
        """
        Extract numeric part from 'op_*' in the column name.

        Args:
            colname (str): Column name to parse.

        Returns:
            int | None: Extracted number or None.
        """
        match = re.search(r"op_(\d+)", colname)
        return int(match.group(1)) if match else None

    @staticmethod
    def _extract_spin_number(colname: str) -> int | None:
        """
        Extract numeric part from 'spin*' in the column name.

        Args:
            colname (str): Column name to parse.

        Returns:
            int | None: Extracted number or None.
        """
        match = re.search(r"spin(\d+)", colname)
        return int(match.group(1)) if match else None

    @staticmethod
    def _is_ope_coeff(colname: str) -> int:
        """
        Determine whether a column name indicates an OPE coefficient.

        Args:
            colname (str): Column name to check.

        Returns:
            int: 0 if it's an OPE coefficient, 1 otherwise.
        """
        return 0 if "ope_coeff_" in colname else 1


def main():
    """
    Command-line entry point for CSV aggregation.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate multiple CSVs into one unified file."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Path to the root directory containing CSVs.",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        required=True,
        help="Path to save the aggregated CSV.",
    )
    args = parser.parse_args()

    aggregator = CSVAggregator(root_dir=args.input_dir)
    aggregator.save_csv(args.output_file)


if __name__ == "__main__":
    main()
