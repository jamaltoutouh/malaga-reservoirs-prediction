"""
Evaluate statistical forecasting models (ARIMA, SARIMA, Holt-Winters).

This script is a renamed and lightly improved version of the previous
`evaluate_models.py`. It performs:
- train/test split
- small grid search for ARIMA and SARIMA
- Holt-Winters fitting
- forecasting for the test horizon
- computation of MAE, RMSE and MAPE
- prints a performance table and optionally saves a comparison plot

Usage:
    python src/evaluate_statistical_models.py data/test.csv --value reserva --test-days 90 --output compare.png

Notes:
- Keep the ARIMA/SARIMA grid ranges small unless you want long runtimes.
- Relies on `load_time_series` and `slice_by_dates` from `plot_single.py`.
"""

import argparse
import os
import sys
import warnings
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from plot_single import load_time_series, slice_by_dates


def split_train_test(series: pd.Series, test_days: int = 90) -> Tuple[pd.Series, pd.Series]:
    s = series.dropna()
    if s.shape[0] <= test_days + 10:
        raise ValueError("Series too short for the requested test horizon")
    train = s.iloc[:-test_days]
    test = s.iloc[-test_days:]
    return train, test


def metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    mask = ~np.isnan(true)
    true = np.asarray(true)[mask]
    pred = np.asarray(pred)[mask]
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    mape = np.mean(np.abs((true - pred) / np.where(true == 0, np.nan, true))) * 100.0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def fit_arima_grid(train: pd.Series, p_max: int = 2, d_vals=(0, 1), q_max: int = 2) -> Tuple[Any, Tuple[int, int, int]]:
    best_aic = np.inf
    best_model = None
    best_order = None
    warnings.filterwarnings("ignore")
    for p in range(0, p_max + 1):
        for d in d_vals:
            for q in range(0, q_max + 1):
                try:
                    model = ARIMA(train, order=(p, d, q)).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_model = model
                        best_order = (p, d, q)
                except Exception:
                    continue
    warnings.resetwarnings()
    return best_model, best_order


def fit_sarima_grid(train: pd.Series, seasonal_period: int = 365, p_max: int = 1, d_vals=(0, 1), q_max: int = 1, P_max: int = 1, D_vals=(0, 1), Q_max: int = 1) -> Tuple[Any, Dict[str, Any]]:
    best_aic = np.inf
    best_model = None
    best_cfg = None
    warnings.filterwarnings("ignore")
    for p in range(0, p_max + 1):
        for d in d_vals:
            for q in range(0, q_max + 1):
                for P in range(0, P_max + 1):
                    for D in D_vals:
                        for Q in range(0, Q_max + 1):
                            try:
                                model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, seasonal_period), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                                if model.aic < best_aic:
                                    best_aic = model.aic
                                    best_model = model
                                    best_cfg = {"order": (p, d, q), "seasonal_order": (P, D, Q, seasonal_period)}
                            except Exception:
                                continue
    warnings.resetwarnings()
    return best_model, best_cfg


def fit_holt_winters(train: pd.Series, seasonal_periods: int = 365) -> Any:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    try:
        model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_periods).fit()
    except Exception:
        model = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=seasonal_periods).fit()
    warnings.resetwarnings()
    return model


def forecast_and_evaluate(model, model_name: str, train: pd.Series, test: pd.Series) -> Dict[str, Any]:
    horizon = len(test)
    try:
        if hasattr(model, 'get_forecast'):
            pred = model.get_forecast(steps=horizon).predicted_mean
        elif hasattr(model, 'forecast'):
            pred = model.forecast(horizon)
        else:
            pred = model.predict(start=test.index[0], end=test.index[-1])
    except Exception:
        pred = pd.Series(np.repeat(train.iloc[-1], horizon), index=test.index)

    if isinstance(pred, (np.ndarray, list)):
        pred = pd.Series(pred, index=test.index)
    pred = pred.reindex(test.index)

    m = metrics(test.values, pred.values)
    res = {"model": model_name, **m, "model_obj": model, "pred_series": pred}
    return res


def run_evaluation(series: pd.Series, test_days: int = 90, seasonal_period: int = 365, arima_cfg: dict = None, sarima_cfg: dict = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    train, test = split_train_test(series, test_days)

    results = []

    # ARIMA grid
    p_max = (arima_cfg or {}).get('p_max', 2)
    q_max = (arima_cfg or {}).get('q_max', 2)
    d_vals = (arima_cfg or {}).get('d_vals', (0, 1))
    print('Fitting ARIMA grid...')
    arima_model, arima_order = fit_arima_grid(train, p_max=p_max, d_vals=d_vals, q_max=q_max)
    if arima_model is not None:
        results.append(forecast_and_evaluate(arima_model, f'ARIMA{arima_order}', train, test))

    # SARIMA grid
    print('Fitting SARIMA grid...')
    sarima_model, sarima_cfg_best = fit_sarima_grid(train, seasonal_period=seasonal_period, p_max=(sarima_cfg or {}).get('p_max', 1), d_vals=(sarima_cfg or {}).get('d_vals', (0, 1)), q_max=(sarima_cfg or {}).get('q_max', 1), P_max=(sarima_cfg or {}).get('P_max', 1), D_vals=(sarima_cfg or {}).get('D_vals', (0, 1)), Q_max=(sarima_cfg or {}).get('Q_max', 1))
    if sarima_model is not None:
        order = sarima_cfg_best.get('order') if sarima_cfg_best else None
        results.append(forecast_and_evaluate(sarima_model, f"SARIMA{order}+season{seasonal_period}", train, test))

    # Holt-Winters
    print('Fitting Holt-Winters...')
    try:
        hw_model = fit_holt_winters(train, seasonal_periods=seasonal_period)
        results.append(forecast_and_evaluate(hw_model, f'Holt-Winters(s={seasonal_period})', train, test))
    except Exception as e:
        print('Holt-Winters fit failed:', e)

    # Build results DataFrame
    records = []
    for r in results:
        rec = {k: v for k, v in r.items() if k not in ('model_obj', 'pred_series')}
        records.append(rec)
    df_res = pd.DataFrame(records).set_index('model')

    preds = {r['model']: r['pred_series'] for r in results}

    return df_res, train, test, preds


def plot_comparison(train: pd.Series, test: pd.Series, preds: dict, title: str = None, output: str = None):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train.values, label='train', color='gray')
    plt.plot(test.index, test.values, label='test', color='black')
    for name, p in preds.items():
        plt.plot(p.index, p.values, label=name)
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Value')
    if title:
        plt.title(title)
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150)
        print('Saved comparison to', output)
    else:
        plt.show()


def main(argv: list[str]):
    parser = argparse.ArgumentParser(description='Evaluate ARIMA, SARIMA and Holt-Winters on a time series')
    parser.add_argument('data_source')
    parser.add_argument('--value', choices=['reserva', 'porcentaje'], default='reserva')
    parser.add_argument('--start', default=None)
    parser.add_argument('--end', default=None)
    parser.add_argument('--test-days', type=int, default=90)
    parser.add_argument('--seasonal-period', type=int, default=365)
    parser.add_argument('--output', default=None, help='Path to save comparison plot')
    parser.add_argument('--p-max', type=int, default=2, help='max p for ARIMA grid')
    parser.add_argument('--q-max', type=int, default=2, help='max q for ARIMA grid')
    parser.add_argument('--p-sarimax', type=int, default=1, help='max p for SARIMA grid')
    args = parser.parse_args(argv)

    try:
        df = load_time_series(args.data_source)
    except Exception as e:
        print('Failed loading data:', e)
        sys.exit(1)

    sub = slice_by_dates(df, args.start, args.end)
    if sub.empty:
        print('No data in range')
        sys.exit(0)

    if args.value not in sub.columns:
        print('Value column not found')
        sys.exit(1)

    series = pd.to_numeric(sub[args.value], errors='coerce')
    series.index = sub.index

    df_res, train, test, preds = run_evaluation(series, test_days=args.test_days, seasonal_period=args.seasonal_period, arima_cfg={'p_max': args.p_max, 'q_max': args.q_max}, sarima_cfg={'p_max': args.p_sarimax})

    print('\nModel performance:')
    print(df_res)

    plot_comparison(train, test, preds, title=f"Model comparison ({args.value})", output=args.output)


if __name__ == '__main__':
    main(sys.argv[1:])
