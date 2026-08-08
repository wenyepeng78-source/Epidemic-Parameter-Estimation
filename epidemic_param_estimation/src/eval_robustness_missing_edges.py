"""
Evaluate missing-edge robustness. Same edge-drop schedule as training.
"""

import os
import gc
import gzip
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .models import EpidemicCNN, GTA
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


def calculate_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-8
    mape_p = np.mean(np.abs((y_true[:, 0] - y_pred[:, 0]) / (y_true[:, 0] + eps))) * 100
    mape_q = np.mean(np.abs((y_true[:, 1] - y_pred[:, 1]) / (y_true[:, 1] + eps))) * 100
    return mape_p, mape_q


def start_unified_evaluation(eval_gta=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gap, n_steps = 3, 6
    days = [gap * i for i in range(1, n_steps + 1)]

    data_path = '/content/drive/MyDrive/real-data/Protein_optimized.pkl.gz'
    csv_path = '/content/drive/MyDrive/real_networks/Protein.csv'
    save_root = '/content/drive/MyDrive/Robust-me'

    with gzip.open(data_path, 'rb') as f:
        saved = pickle.load(f)
    data_chunk = saved['trajectories']
    node_mapping = saved['node_mapping']
    num_nodes = len(node_mapping)
    M = len(data_chunk)

    df = pd.read_csv(csv_path)
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

    test_idx = []
    for i in range(0, M, 50):
        test_idx.extend(range(i + 45, min(i + 50, M)))
    test_chunk = [data_chunk[j] for j in test_idx]

    results = {
        'Missing Ratio': [],
        'XGB_P_MAPE(%)': [], 'XGB_Q_MAPE(%)': [],
        'CNN_P_MAPE(%)': [], 'CNN_Q_MAPE(%)': [],
        'GTA_P_MAPE(%)': [], 'GTA_Q_MAPE(%)': [],
    }

    for i in range(5):
        ratio = (i * drop_n) / total
        ratio_str = f"{ratio:.2f}"
        results['Missing Ratio'].append(f"{ratio:.1%}")
        print(f"\nMissing={ratio:.1%} | edges={current.shape[1]}")

        xgb_path = os.path.join(save_root, f"xgb_robust_{ratio_str}.pkl")
        if os.path.exists(xgb_path):
            X, y = extract_xgb_features_robust(
                data_chunk, test_idx, current, num_nodes, days, "Eval-XGB"
            )
            with open(xgb_path, 'rb') as f:
                model = pickle.load(f)
            mape_p_x, mape_q_x = calculate_mape(y, model.predict(X))
        else:
            mape_p_x, mape_q_x = np.nan, np.nan
        results['XGB_P_MAPE(%)'].append(mape_p_x)
        results['XGB_Q_MAPE(%)'].append(mape_q_x)

        cnn_path = os.path.join(save_root, f"cnn_robust_{ratio_str}.pth")
        if os.path.exists(cnn_path):
            X, Y = extract_cnn_features_robust(test_chunk, current, num_nodes, days)
            model = EpidemicCNN(K=n_steps).to(device)
            model.load_state_dict(torch.load(cnn_path, map_location=device))
            model.eval()
            with torch.no_grad():
                pred = model(X.to(device)).cpu().numpy()
            mape_p_c, mape_q_c = calculate_mape(Y.numpy(), pred)
        else:
            mape_p_c, mape_q_c = np.nan, np.nan
        results['CNN_P_MAPE(%)'].append(mape_p_c)
        results['CNN_Q_MAPE(%)'].append(mape_q_c)

        gta_path = os.path.join(save_root, f"gta_robust_{ratio_str}.pth")
        if eval_gta and os.path.exists(gta_path) and to_undirected is not None:
            gnn_e = to_undirected(current) if current.shape[1] > 0 else current
            loader = DataLoader(
                StaticSIRDataset(test_chunk, gnn_e, days, num_nodes),
                batch_size=64, shuffle=False, collate_fn=gta_collate_fn,
            )
            model = GTA(num_times=len(days), d_model=256, nhead=8).to(device)
            model.load_state_dict(torch.load(gta_path, map_location=device))
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xl, el, ti, y in loader:
                    pred = model(
                        [x.to(device) for x in xl],
                        [e.to(device) for e in el],
                        ti.to(device),
                    )
                    preds.append(pred.cpu().numpy())
                    trues.append(y.numpy())
            mape_p_g, mape_q_g = calculate_mape(
                np.concatenate(trues), np.concatenate(preds)
            )
        else:
            mape_p_g, mape_q_g = np.nan, np.nan
        results['GTA_P_MAPE(%)'].append(mape_p_g)
        results['GTA_Q_MAPE(%)'].append(mape_q_g)

        print(f"  XGB  P={mape_p_x:.2f}% Q={mape_q_x:.2f}%")
        print(f"  CNN  P={mape_p_c:.2f}% Q={mape_q_c:.2f}%")

        current = drop_edges_fixed_count(current, drop_n, seed=42 + i)
        gc.collect()

    report = pd.DataFrame(results)
    out = os.path.join(save_root, "robustness_mape_report.csv")
    report.to_csv(out, index=False)
    print(report.to_string(index=False, float_format="%.2f"))
    print(f"Saved: {out}")
    return report


if __name__ == "__main__":
    start_unified_evaluation(eval_gta=False)