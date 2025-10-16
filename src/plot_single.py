"""
Script to display a single time series CSV file.

Usage:
    python src/plot_single.py data/CASASOLA.csv --start 2010-01-01 --end 2020-12-31 --value reserva

Arguments:
    data_source (positional): path to the CSV inside `data/` or an absolute path.
    --start (optional): start date (e.g. 2010-01-01). If omitted, the beginning of the data is used.
    --end (optional): end date (e.g. 2020-12-31). If omitted, the end of the data is used.
    --value (optional): which field to plot: 'reserva' (total water) or 'porcentaje' (percentage). Default: 'reserva'.
    --output (optional): path to save the figure (png). If omitted the figure is shown on screen.

Behavior:
    - Attempts to automatically detect/parse the date column (e.g. 'timestamp').
    - Plots the selected column (reserva or porcentaje). If the selected column is not present
      an informative error is raised.
"""

import argparse
import os
import sys
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt


def load_time_series(path: str) -> pd.DataFrame:
    """Load a CSV and return a DataFrame indexed by datetime.

    Notes:
    - InfluxDB CSV exports often include a first line with datatype metadata
      (e.g. "#datatype measurement,tag,tag,field,field,timestamp") and the
      second line contains the actual column names. This function detects that
      pattern and will read the correct header row (header=1) when necessary.
    - Expected column names (case-insensitive): measurement,codigo,nombre,reserva,porcentaje,timestamp
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    expected = {"measurement", "codigo", "nombre", "reserva", "porcentaje", "timestamp"}

    # Try reading with header=0 first
    try:
        df = pd.read_csv(path, header=0)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")

    cols_lower = {str(c).lower() for c in df.columns}

    # If the first header row is a datatype line or expected columns are missing,
    # try reading with header=1 (use second row as header)
    if any(str(c).lower().startswith('#datatype') for c in df.columns) or not expected.issubset(cols_lower):
        try:
            df = pd.read_csv(path, header=1)
        except Exception:
            # If this also fails, keep the original df and attempt to recover below
            pass

    # At this point, try to detect the timestamp column (case-insensitive)
    ts_col = None
    for c in df.columns:
        if str(c).lower() == 'timestamp':
            ts_col = c
            break

    if ts_col is not None:
        # parse timestamp column and set as index
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        df = df.set_index(ts_col)
    else:
        # If no explicit timestamp column, try to parse the first column as datetime
        first_col = df.columns[0]
        parsed = pd.to_datetime(df[first_col], errors='coerce')
        if parsed.notna().sum() > 0:
            df[first_col] = parsed
            df = df.set_index(first_col)
        else:
            # As a last resort, try re-reading with index_col=0 and parsing the index
            try:
                df = pd.read_csv(path, header=1, index_col=0)
                df.index = pd.to_datetime(df.index, errors='coerce')
            except Exception:
                pass

    # drop rows with NaT index and sort
    if pd.api.types.is_datetime64_any_dtype(df.index):
        df = df[~df.index.isna()]
        df = df.sort_index()
    else:
        raise ValueError("Could not detect or parse a date/time column in the CSV. Make sure the file has a 'timestamp' column (second row header for InfluxDB exports).")

    return df


def slice_by_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        start_ts = pd.to_datetime(start)
    else:
        start_ts = df.index.min()
    if end:
        end_ts = pd.to_datetime(end)
    else:
        end_ts = df.index.max()
    return df.loc[start_ts:end_ts]


def plot_series(df: pd.DataFrame, value_col: str = "reserva", title: str = None, output: str | None = None) -> None:
    plt.figure(figsize=(12, 6))

    # If the requested column exists, use it (attempt to coerce to numeric)
    if value_col in df.columns:
        series = pd.to_numeric(df[value_col], errors="coerce")
        series.index = df.index
        if series.dropna().empty:
            raise ValueError(f"Column '{value_col}' contains no numeric data to plot.")
        plt.plot(series.index, series.values, label=value_col)
    else:
        # Fallback: select any numeric columns (previous behavior)
        numeric = df.select_dtypes(include=["number"]) 
        if numeric.shape[1] == 0:
            available = ", ".join(map(str, df.columns.tolist()))
            raise ValueError(f"Requested column '{value_col}' not found and no numeric columns available. Available columns: {available}")
        if isinstance(numeric, pd.Series):
            plt.plot(numeric.index, numeric.values, label=numeric.name)
        else:
            for col in numeric.columns:
                plt.plot(numeric.index, numeric[col], label=col)

    plt.xlabel("Date")
    plt.ylabel("Value")
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        print(f"Figure saved to: {output}")
    else:
        plt.show()


def main(argv: list[str]):
    parser = argparse.ArgumentParser(description="Display a time series from a single CSV file.")
    parser.add_argument("data_source", help="Path to the CSV file (e.g. data/CASASOLA.csv)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)", default=None)
    parser.add_argument("--end", help="End date (YYYY-MM-DD)", default=None)
    parser.add_argument("--value", help="Field to plot: 'reserva' or 'porcentaje'", choices=["reserva", "porcentaje"], default="reserva")
    parser.add_argument("--output", help="Path to save the figure (png)", default=None)

    args = parser.parse_args(argv)

    try:
        df = load_time_series(args.data_source)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    try:
        sub = slice_by_dates(df, args.start, args.end)
    except Exception as e:
        print(f"Error applying date range: {e}")
        sys.exit(1)

    if sub.shape[0] == 0:
        print("No data in the requested date range.")
        sys.exit(0)

    title = f"{os.path.basename(args.data_source)}  ({sub.index.min().date()} — {sub.index.max().date()})"
    plot_series(sub, value_col=args.value, title=title, output=args.output)


if __name__ == "__main__":
    main(sys.argv[1:])
