"""
Exploratory analysis tools for reservoir time series.

Features:
- Trend/seasonal decomposition (STL)
- Anomaly detection via robust z-score on residuals
- Annual and seasonal (DJF/MAM/JJA/SON) aggregates and drought index (SPI-like simple)
- ACF/PACF plots and Augmented Dickey-Fuller test for stationarity

Usage examples:
    python src/explore.py data/test.csv --value reserva --start 2002-01-01 --end 2002-12-31 --output exploratory.png

The script produces a multi-panel figure saved to --output or shown interactively.
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from scipy import stats

from plot_single import load_time_series, slice_by_dates


def detect_anomalies(residuals: pd.Series, z_thresh: float = 3.0) -> pd.Series:
    """Detect anomalies in residuals using robust z-score (median and MAD). Returns boolean series."""
    med = residuals.median()
    mad = (np.abs(residuals - med)).median()
    if mad == 0:
        # fallback to standard zscore
        z = stats.zscore(residuals.fillna(0))
    else:
        z = 0.6745 * (residuals - med) / mad
    return np.abs(z) > z_thresh


def compute_seasonal_aggregates(series: pd.Series) -> pd.DataFrame:
    df = series.to_frame(name="value")
    df["year"] = df.index.year
    # meteorological seasons: DJF, MAM, JJA, SON
    df["season"] = ((df.index.month % 12 + 3) // 3)
    agg_year = df.groupby("year").value.mean()
    agg_season = df.groupby(["year", "season"]).value.mean().unstack()
    return pd.DataFrame({"annual_mean": agg_year}).join(agg_season)


def simple_drought_index(series: pd.Series) -> pd.Series:
    """A simple standardized anomaly index per month (like SPI but simpler):
    For each row, compute the z-score relative to the climatology of that calendar month
    (using mean and std computed across all years for that month).

    Returns a Series with the same index as the input.
    """
    s = series.copy()
    df = s.to_frame("value")
    df["month"] = df.index.month

    # monthly climatology across years
    climatology = df.groupby("month")["value"].agg(["mean", "std"]).reindex(range(1, 13))

    # map climatology to each row
    month_mean = df["month"].map(climatology["mean"])
    month_std = df["month"].map(climatology["std"])

    # avoid division by zero; where std is zero or NaN, result is NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        drought_index = (df["value"] - month_mean) / month_std

    drought_index.name = "drought_index"
    return drought_index


def analyze(series: pd.Series, value_col: str = "reserva", output: Optional[str] = None, lags: Optional[int] = None):
    # resample to daily if finer granularity; assume series index is datetime
    s = series.asfreq('D')

    # determine automatic number of lags if not provided
    n_obs = s.dropna().shape[0]
    if lags is None:
        # default: 10% of observations, at least 10, at most 365
        lags = min(365, max(10, int(max(1, n_obs * 0.1))))
    # ensure lags is not larger than available observations
    lags = int(min(lags, max(1, n_obs - 1)))

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    # 1. Raw series
    axes[0, 0].plot(s.index, s.values)
    axes[0, 0].set_title("Raw series")

    # 2. STL decomposition
    stl = STL(s.interpolate(), period=365, robust=True)
    res = stl.fit()
    res_trend = res.trend
    res_seasonal = res.seasonal
    res_resid = res.resid

    axes[0, 1].plot(res_trend.index, res_trend, label='trend')
    axes[0, 1].plot(res_seasonal.index, res_seasonal, alpha=0.7, label='seasonal')
    axes[0, 1].set_title('STL Trend + Seasonal')
    axes[0, 1].legend()

    # 3. anomalies
    anomalies = detect_anomalies(res_resid.dropna())
    axes[1, 0].plot(res_resid.index, res_resid.values, label='residual')
    axes[1, 0].scatter(res_resid.index[anomalies], res_resid[anomalies], color='red', label='anomaly')
    axes[1, 0].set_title('Residuals and anomalies')
    axes[1, 0].legend()

    # 4. annual/seasonal aggregates
    agg = compute_seasonal_aggregates(s.dropna())
    axes[1, 1].plot(agg.index, agg['annual_mean'], marker='o')
    axes[1, 1].set_title('Annual mean')

    # 5. drought index (monthly standardized anomalies)
    drought = simple_drought_index(s.dropna())
    axes[2, 0].plot(drought.index, drought.values)
    axes[2, 0].axhline(-1, color='orange', linestyle='--')
    axes[2, 0].axhline(-1.5, color='red', linestyle='--')
    axes[2, 0].set_title('Simple drought index (monthly z-score)')

    # 6. ACF
    plot_acf(s.dropna(), ax=axes[2, 1], lags=lags)
    axes[2, 1].set_title(f'ACF ({lags} lags)')

    # 7. PACF
    plot_pacf(s.dropna(), ax=axes[3, 0], lags=lags, method='ywm')
    axes[3, 0].set_title(f'PACF ({lags} lags)')

    # 8. Stationarity test
    adf_res = adfuller(s.dropna())
    axes[3, 1].text(0.01, 0.6, f"ADF stat: {adf_res[0]:.3f}")
    axes[3, 1].text(0.01, 0.4, f"p-value: {adf_res[1]:.3f}")
    axes[3, 1].text(0.01, 0.2, f"Used lags: {adf_res[2]}")
    axes[3, 1].axis('off')
    axes[3, 1].set_title('Stationarity (ADF)')

    plt.suptitle(f"Exploratory analysis — {value_col}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if output:
        plt.savefig(output, dpi=150)
        print(f"Exploration saved to: {output}")
    else:
        plt.show()


def main(argv: list[str]):
    parser = argparse.ArgumentParser(description="Exploratory analysis for reservoir time series")
    parser.add_argument("data_source", help="Path to CSV file or a sample file in data/")
    parser.add_argument("--value", choices=["reserva", "porcentaje"], default='reserva')
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--lags", type=int, default=None, help="Number of lags for ACF/PACF (default: automatic based on series length)")
    args = parser.parse_args(argv)

    try:
        df = load_time_series(args.data_source)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    sub = slice_by_dates(df, args.start, args.end)
    if sub.empty:
        print("No data in range")
        sys.exit(0)

    # pick the series
    if args.value not in sub.columns:
        print(f"Value column '{args.value}' not found in CSV. Available: {', '.join(sub.columns)}")
        sys.exit(1)

    series = pd.to_numeric(sub[args.value], errors='coerce')
    series.index = sub.index

    analyze(series, value_col=args.value, output=args.output, lags=args.lags)


if __name__ == "__main__":
    main(sys.argv[1:])
