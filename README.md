# Malaga Reservoirs — Time Series Visualization and Forecasting

This repository contains tools to visualize and analyse time series exported from InfluxDB for water reservoirs in the province of Málaga.

## Project structure
- data/: CSV files exported from InfluxDB (one file per reservoir/measurement)
- src/: Python scripts to visualize, explore and evaluate forecasting models
- notebooks/: demo notebooks showing usage of the scripts
- requirements.txt: Python dependencies

## Quick start
1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Basic usage
- Plot a single CSV file (choose `reserva` or `porcentaje`):

   ```bash
   python src/plot_single.py data/test.csv --value reserva
   ```

- Plot all CSV files in `data/` on one figure:

   ```bash
   python src/plot_all.py --data-dir data --value porcentaje --output all_reservoirs.png
   ```

## Exploratory analysis
Use `src/explore.py` to run decomposition, anomaly detection, aggregates, ACF/PACF and stationarity tests.

- Interactive (automatic lags):

   ```bash
   python src/explore.py data/test.csv --value reserva
   ```

- With custom ACF/PACF lags and save output:

   ```bash
   python src/explore.py data/test.csv --value reserva --lags 180 --output explore.png
   ```

## Model evaluation (forecasting)
Use `src/evaluate_statistical_models.py` to compare ARIMA, SARIMA and Holt-Winters on a chosen series.
The script performs a small grid search for ARIMA/SARIMA, fits Holt-Winters, forecasts the test horizon and reports MAE, RMSE and MAPE.

Example:

   ```bash
   python src/evaluate_statistical_models.py data/test.csv --value reserva --test-days 90 --output compare.png
   ```

## Notebooks
Demo notebooks are provided in `notebooks/`:
- `demo_plot_single.ipynb` — load and plot a single CSV using `plot_single` utilities.
- `demo_explore.ipynb` — run the exploratory analysis and visualize results (shows automatic and custom lags).
- `demo_evaluate_models.ipynb` — evaluate and compare ARIMA, SARIMA and Holt-Winters on a sample series (see description below).

### Demo notebook: demo_evaluate_models.ipynb
This notebook demonstrates the full model evaluation workflow using the project's evaluation utilities:
- Loads a sample CSV (`data/test.csv`) using the same loader as the CLI tools (handles InfluxDB CSV headers).
- Prepares the series (numeric conversion, index alignment, drop missing values).
- Runs `run_evaluation(...)` which fits ARIMA, SARIMA and Holt-Winters and computes MAE, RMSE and MAPE on a hold-out test horizon.
- Plots the train/test series and each model's forecasts for visual comparison.

#### How to run the notebook
1. Start Jupyter in the project root:

   ```bash
   jupyter lab
   ```

2. Open `notebooks/demo_evaluate_models.ipynb` and run the cells. The notebook runs a short demo evaluation (default uses a small grid for speed).

#### Command-line equivalent
You can run the same evaluation from the CLI without the notebook:

   ```bash
   python src/evaluate_statistical_models.py data/test.csv --value reserva --test-days 30 --output compare.png
   ```

## GRU demo (PyTorch)
A notebook demonstrating training and evaluation of a GRU predictor is available at `notebooks/demo_gru.ipynb`.

This notebook shows a complete workflow: loading data, preparing sequences, training a GRU, plotting training/validation loss, and comparing forecasts against a hold-out test set.

To run the notebook in Jupyter/Colab ensure PyTorch is installed appropriate to your platform. Example CPU install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Open the notebook and run cells or execute the equivalent script:

```bash
python src/gru_predictor.py data/test.csv --value reserva --test-days 30 --seq-len 30 --epochs 50 --output gru_compare.png --model-out gru_model.pth
```

## CSV format
CSV files are expected to come from InfluxDB with the second row containing column names, e.g.:

```csv
measurement,codigo,nombre,reserva,porcentaje,timestamp
```

The loader in `src/plot_single.py` detects the InfluxDB metadata line and uses the second row as header, then indexes the data by the timestamp column.

## Recommendations
- For model evaluation increase ARIMA/SARIMA grid ranges only when you can afford longer runtime.
- For ACF/PACF set `--lags` according to series length and frequency (daily: 30/90/365).

## Contributing / GitHub
- Initialize git locally: `git init` then `git add . && git commit -m "Initial commit"`.
- Create a GitHub repo and push: `git remote add origin <URL> && git push -u origin main`.

## License
MIT
