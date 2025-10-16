"""
Plot all CSV files from a data directory on the same figure.

Usage:
    python src/plot_all.py --data-dir data --value reserva --start 2002-01-01 --end 2002-03-01 --output all.png
"""

import argparse
import os
import sys
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt

from plot_single import load_time_series, slice_by_dates


def plot_all(data_dir: str, value_col: str = "reserva", start: str | None = None, end: str | None = None, output: str | None = None):
    files = sorted(glob(os.path.join(data_dir, "*.csv")))
    if not files:
        print(f"No CSV files found in: {data_dir}")
        sys.exit(1)

    plt.figure(figsize=(14, 7))
    for f in files:
        try:
            df = load_time_series(f)
        except Exception as e:
            print(f"Skipping {f}: failed to load ({e})")
            continue
        df = slice_by_dates(df, start, end)
        if df.empty:
            continue
        if value_col in df.columns:
            series = pd.to_numeric(df[value_col], errors="coerce")
            series.index = df.index
            if series.dropna().empty:
                continue
            plt.plot(series.index, series.values, label=os.path.basename(f))
        else:
            numeric = df.select_dtypes(include=["number"]) 
            if numeric.shape[1] == 0:
                continue
            # plot first numeric column as fallback
            col = numeric.columns[0]
            plt.plot(numeric.index, numeric[col], label=f"{os.path.basename(f)}:{col}")

    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.title(f"All reservoirs — {value_col}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        print(f"Saved figure to: {output}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot all CSV files from a directory on the same figure.")
    parser.add_argument("--data-dir", default="data", help="Path to data directory containing CSV files")
    parser.add_argument("--value", choices=["reserva", "porcentaje"], default="reserva", help="Field to plot")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default=None, help="Path to save the figure (png)")
    args = parser.parse_args()

    plot_all(args.data_dir, value_col=args.value, start=args.start, end=args.end, output=args.output)
