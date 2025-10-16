#Malaga Reservoirs — Time Series Visualization and Forecasting

This repository contains tools to visualize time series exported from InfluxDB for water reservoirs in the province of Málaga.

Structure
- data/: CSV files exported from InfluxDB (one file per reservoir or measurement)
- src/: Python scripts to visualize time series
- notebooks/: demo notebooks showing usage of the scripts
- requirements.txt: Python dependencies

Quick Start
1. Create and activate a Python virtual environment (recommended):

   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Plot a single CSV file:

   python src/plot_single.py data/test.csv --value reserva

4. Plot all CSV files from the `data/` folder on a single figure:

   python src/plot_all.py --data-dir data --value porcentaje --output all_reservoirs.png

Exploratory analysis
The repository includes `src/explore.py` to perform exploratory data analysis on a single reservoir time series. Features:
- STL decomposition (trend, seasonal, residual)
- Anomaly detection on residuals (robust z-score)
- Annual and seasonal aggregates
- A simple monthly drought index (standardized anomalies)
- ACF/PACF plots and Augmented Dickey-Fuller test for stationarity

Example usage:

   python src/explore.py data/test.csv --value reserva --start 2002-01-01 --end 2002-12-31 --output explore.png

Notebooks
Demo notebooks are provided in `notebooks/`:
- `notebooks/demo_plot_single.ipynb` — load and plot a single CSV using `plot_single` utilities.
- `notebooks/demo_explore.ipynb` — run the exploratory analysis and visualize results.

CSV format
The CSV files are expected to come from InfluxDB with columns similar to:

measurement,codigo,nombre,reserva,porcentaje,timestamp

The scripts automatically detect the date/time column (commonly `timestamp`) and index the data by datetime.

CLI options
- --start / --end: limit the plotted date range (YYYY-MM-DD)
- --value: choose which field to plot: `reserva` (total water) or `porcentaje` (percentage)
- --output: path to save the generated PNG figure (if omitted the figure is shown interactively)

License
MIT
