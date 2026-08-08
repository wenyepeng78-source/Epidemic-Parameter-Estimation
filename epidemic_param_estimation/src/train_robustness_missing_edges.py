"""
Missing-edge robustness training.
Uses models from models.py and robust feature extractors from data.py.
"""

import os
import gc
import gzip
import pickle

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import xgboost as xgb

from .models import set_seed, MSLELoss, EpidemicCNN, GTA
from .data import (
    drop_edges_fixed_count,
    extract_xgb_features_robust,
    extract_cnn_features_robust,
    StaticSIRDataset,
    gta_collate_fn,
)

try:
    from torch_geometric.utils import to_undirected
except ImportError:
    to_undirected = None

SEED = 520
set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = '/content/drive/MyDrive/real-data/Protein_optimized.pkl.gz'
CSV_PATH = '/content/drive/MyDrive/real_networks/Protein.csv'
SAVE_ROOT = '/content/drive/MyDrive/Robust-me'
os.makedirs(SAVE_ROOT, exist_ok=True)


def train_xgb(X_train, y_train, X_val, y_val, save_path):
    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.2,
        max_depth=5,
        min_child_weight=0.7,
        subsample=0.9,
        colsample_bytree=0.8,
        objective='reg:squaredlogerror',
        tree_method='hist',
        early_stopping_rounds=15,
        n_jobs=-1,
        random_state=520,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  [XGBoost] saved {os.path.basename(save_path)}")


def train_cnn(X, Y, n_steps, save_path):
    dataset = TensorDataset(X, Y)
    M = len(dataset)
    train_idx, val_idx = [], []
    for i in range(0, M, 50):
        train_idx.extend(range(i, min(i + 40, M)))
        val_idx.extend(range(i + 40, min(i + 45, M)))

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=64, shuffle=True, generator=g
    )
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64)

    model = EpidemicCNN(K=n_steps, input_channels=7).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = MSLELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best, wait = float('inf'), 0
    for _ in range(200):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

        model.eval()
        v = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                v += criterion(model(bx.to(device)), by.to(device)).item()
        v /= len(val_loader)
        scheduler.step(v)

        if v < best:
            best, wait = v, 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= 15:
                break
    print(f"  [CNN] best val={best:.6f}")


def train_gta(data_chunk, edges, num_nodes, days, save_path):
    dataset = StaticSIRDataset(data_chunk, edges, days, num_nodes)
    M = len(dataset)
    train_idx, val_idx = [], []
    for i in range(0, M, 50):
        train_idx.extend(range(i, min(i + 40, M)))
        val_idx.extend(range(i + 40, min(i + 45, M)))

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=64, shuffle=True,
        collate_fn=gta_collate_fn, generator=g,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=64, shuffle=False,
        collate_fn=gta_collate_fn,
    )

    model = GTA(num_times=len(days), d_model=256, nhead=8).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = MSLELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best, wait = float('inf'), 0
    for _ in range(200):
        model.train()
        for xl, el, ti, y in train_loader:
            optimizer.zero_grad()
            pred = model(
                [x.to(device) for x in xl],
                [e.to(device) for e in el],
                ti.to(device),
            )
            loss = criterion(pred, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        v = 0.0
        with torch.no_grad():
            for xl, el, ti, y in val_loader:
                pred = model(
                    [x.to(device) for x in xl],
                    [e.to(device) for e in el],
                    ti.to(device),
                )
                v += criterion(pred, y.to(device)).item()
        v /= len(val_loader)
        scheduler.step(v)

        if v < best:
            best, wait = v, 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= 15:
                break
    print(f"  [GTA] best val={best:.6f}")


def start_unified_robustness_benchmark(run_gta=False):
    gap, n_steps = 3, 6
    days = [gap * i for i in range(1, n_steps + 1)]

    with gzip.open(DATA_PATH, 'rb') as f:
        saved = pickle.load(f)
    data_chunk = saved['trajectories']
    node_mapping = saved['node_mapping']
    num_nodes = len(node_mapping)
    M = len(data_chunk)

    df = pd.read_csv(CSV_PATH)
    s_col = 'source' if 'source' in df.columns else 'Source'
    t_col = 'target' if 'target' in df.columns else 'Target'
    full_edges = torch.tensor(
        [[node_mapping[n] for n in df[s_col]],
         [node_mapping[n] for n in df[t_col]]],
        dtype=torch.long,
    )
    total = full_edges.shape[1]
    drop_n = int(total * 0.2)
    current = full_edges.clone()

    train_idx, val_idx = [], []
    for i in range(0, M, 50):
        train_idx.extend(range(i, min(i + 40, M)))
        val_idx.extend(range(i + 40, min(i + 45, M)))

    for i in range(5):
        ratio = (i * drop_n) / total
        print(f"\nRound {i} | missing={ratio:.1%} | edges={current.shape[1]}")

        Xtr, ytr = extract_xgb_features_robust(
            data_chunk, train_idx, current, num_nodes, days, "XGB-Train"
        )
        Xva, yva = extract_xgb_features_robust(
            data_chunk, val_idx, current, num_nodes, days, "XGB-Val"
        )
        train_xgb(
            Xtr, ytr, Xva, yva,
            os.path.join(SAVE_ROOT, f"xgb_robust_{ratio:.2f}.pkl"),
        )
        del Xtr, ytr, Xva, yva
        gc.collect()

        Xc, Yc = extract_cnn_features_robust(data_chunk, current, num_nodes, days)
        train_cnn(
            Xc, Yc, n_steps,
            os.path.join(SAVE_ROOT, f"cnn_robust_{ratio:.2f}.pth"),
        )
        del Xc, Yc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if run_gta and to_undirected is not None:
            gnn_e = to_undirected(current) if current.shape[1] > 0 else current
            train_gta(
                data_chunk, gnn_e, num_nodes, days,
                os.path.join(SAVE_ROOT, f"gta_robust_{ratio:.2f}.pth"),
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        current = drop_edges_fixed_count(current, drop_n, seed=42 + i)


if __name__ == "__main__":
    start_unified_robustness_benchmark(run_gta=False)