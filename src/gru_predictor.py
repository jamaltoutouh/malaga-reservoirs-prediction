"""
GRU-based predictor using PyTorch.

Usage example:
    python src/gru_predictor.py data/test.csv --value reserva --test-days 90 --seq-len 30 --epochs 50 --output gru_compare.png

Features:
- Loads CSV using existing loader (`load_time_series`) that supports InfluxDB CSV format.
- Prepares sequences (seq_len -> predict next step) with scaling.
- Trains a GRU to predict next value; supports GPU if available.
- Evaluates on hold-out test horizon using iterative forecasting.
- Computes MAE, RMSE, MAPE and saves model and comparison plot.
"""

import argparse
import os
import sys
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from plot_single import load_time_series, slice_by_dates


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.x = sequences.astype(np.float32)
        self.y = targets.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class GRUModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        # take last output
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


def create_sequences(values: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i + seq_len])
        y.append(values[i + seq_len])
    X = np.array(X)
    y = np.array(y)
    # reshape X to (n_samples, seq_len, input_dim)
    if X.ndim == 2:
        X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int = 20, lr: float = 1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = np.inf
    best_state = None
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses) if train_losses else np.nan
        val_loss = np.mean(val_losses) if val_losses else np.nan
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.6f}  val_loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def iterative_forecast(model: nn.Module, seed_sequence: np.ndarray, steps: int, device: torch.device, scaler: StandardScaler = None) -> np.ndarray:
    """Generate multi-step forecasts by feeding back predictions."""
    model.to(device)
    model.eval()
    seq = seed_sequence.copy().astype(np.float32)
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(seq.reshape(1, seq.shape[0], 1), dtype=torch.float32).to(device)
            out = model(x).cpu().numpy()
            preds.append(out.item())
            # append prediction and slide
            seq = np.roll(seq, -1)
            seq[-1] = out
    preds = np.array(preds)
    if scaler is not None:
        # inverse transform: scaler expects 2D
        preds = scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    return preds


def evaluate_predictions(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    mask = ~np.isnan(true)
    true = true[mask]
    pred = pred[mask]
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    mape = np.mean(np.abs((true - pred) / np.where(true == 0, np.nan, true))) * 100.0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def main(argv: list[str]):
    parser = argparse.ArgumentParser(description='Train and evaluate a GRU predictor (PyTorch)')
    parser.add_argument('data_source')
    parser.add_argument('--value', choices=['reserva', 'porcentaje'], default='reserva')
    parser.add_argument('--start', default=None)
    parser.add_argument('--end', default=None)
    parser.add_argument('--test-days', type=int, default=90)
    parser.add_argument('--seq-len', type=int, default=30)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', default=None, help='Path to save comparison plot')
    parser.add_argument('--model-out', default=None, help='Path to save trained model state (pth)')
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
    series = series.dropna()

    if len(series) < args.seq_len + 10:
        print('Series too short for the chosen sequence length')
        sys.exit(1)

    # split train/test
    train = series.iloc[:-args.test_days]
    test = series.iloc[-args.test_days:]

    # scaling (fit on train)
    scaler = StandardScaler()
    train_vals = train.values.reshape(-1, 1)
    scaler.fit(train_vals)
    train_scaled = scaler.transform(train_vals).ravel()
    all_scaled = scaler.transform(series.values.reshape(-1, 1)).ravel()

    # create sequences from train (use small validation split)
    X_all, y_all = create_sequences(all_scaled[:len(train_scaled)], args.seq_len)
    # split last 10% for validation
    n_train = int(len(X_all) * 0.9)
    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train:], y_all[n_train:]

    train_ds = TimeSeriesDataset(X_train, y_train)
    val_ds = TimeSeriesDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = GRUModel(input_size=1, hidden_size=args.hidden_size, num_layers=args.layers)
    model, history = train_model(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)

    # plot training/validation loss
    loss_out = None
    if args.output:
        loss_out = os.path.splitext(args.output)[0] + '_loss.png'

    try:
        plt.figure(figsize=(8, 4))
        plt.plot(history['train_loss'], label='train_loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.tight_layout()
        if loss_out:
            plt.savefig(loss_out, dpi=150)
            print('Saved training loss plot to', loss_out)
        else:
            plt.show()
    except Exception as e:
        print('Failed to plot/save training loss:', e)

    # iterative forecasting on test horizon: seed with last seq before test
    seed_start = len(series) - args.test_days - args.seq_len
    seed_seq = all_scaled[seed_start:seed_start + args.seq_len]
    preds = iterative_forecast(model, seed_seq, steps=args.test_days, device=device, scaler=scaler)

    # evaluate
    true = test.values
    metrics = evaluate_predictions(true, preds)
    print('\nGRU performance:')
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # build pred series with same index
    pred_series = pd.Series(preds, index=test.index)

    # plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train.values, label='train', color='gray')
    plt.plot(test.index, test.values, label='test', color='black')
    plt.plot(pred_series.index, pred_series.values, label='GRU', color='tab:blue')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel(args.value)
    plt.title('GRU forecast')
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print('Saved comparison to', args.output)
    else:
        plt.show()

    # save model
    if args.model_out:
        torch.save({'model_state_dict': model.state_dict(), 'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_}, args.model_out)
        print('Saved model to', args.model_out)


if __name__ == '__main__':
    main(sys.argv[1:])
