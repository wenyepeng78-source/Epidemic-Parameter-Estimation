"""
Evaluate node-state missing robustness.
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
    update_hidden_mask,
    extract_xgb_features_nodes,
    extract_cnn_features_nodes,
    StaticSIRDataset_Nodes,
    gta_collate_fn,
)

try:
    from torch_geometric.utils import to_undirected
except ImportError:
    to_undirected = None


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    eps = 1e-8
    mape_p = np.mean(np.abs((y_true[:, 0] - y_pred[:, 0]) / (y_true[:, 0] + eps))) * 100
    mape_q = np.mean(np.abs((y_true[:, 1] - y_pred[:, 1]) / (y_true[:, 1] + eps))) * 100
    return mape_p, mape_q


def start_node_unified_evaluation(eval_xgb=True, eval_cnn=True, eval_gta=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gap, n_steps = 3, 6
    days = [gap * i for i in range(1, n_steps + 1)]

    data_path = '/content/drive/MyDrive/real-data/Protein_optimized.pkl.gz'
    csv_path = '/content/drive/MyDrive/real_networks/Protein.csv'
    save_root = '/content/drive/MyDrive/XGBoost-ui'

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
    gnn_edges = to_undirected(full_edges) if to_undirected is not None else full_edges

    drop_n = int(num_nodes * 0.2)
    hidden = torch.zeros(num_nodes, dtype=torch.bool)

    test_idx = []
    for i in range(0, M, 50):
        test_idx.extend(range(i + 45, min(i + 50, M)))
    test_chunk = [data_chunk[j] for j in test_idx]

    results = {
        'Hidden Ratio': [],
        'XGB_P_MAPE(%)': [], 'XGB_Q_MAPE(%)': [],
        'CNN_P_MAPE(%)': [], 'CNN_Q_MAPE(%)': [],
        'GTA_P_MAPE(%)': [], 'GTA_Q_MAPE(%)': [],
    }

    for i in range(5):
        ratio = (i * drop_n) / num_nodes
        ratio_str = f"{ratio:.2f}"
        results['Hidden Ratio'].append(f"{ratio:.1%}")
        print(f"\nHidden={ratio:.1%} | nodes={int(hidden.sum())}")

        if eval_xgb:
            path = os.path.join(save_root, f"xgb_noderobust_{ratio_str}.pkl")
            if os.path.exists(path):
                X, y = extract_xgb_features_nodes(
                    data_chunk, test_idx, full_edges, num_nodes, days, hidden, "Eval-XGB"
                )
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                mp, mq = calculate_mape(y, model.predict(X))
            else:
                mp, mq = np.nan, np.nan
            results['XGB_P_MAPE(%)'].append(mp)
            results['XGB_Q_MAPE(%)'].append(mq)
            print(f"  XGB P={mp:.2f}% Q={mq:.2f}%")

        if eval_cnn:
            path = os.path.join(save_root, f"cnn_noderobust_{ratio_str}.pth")
            if os.path.exists(path):
                X, Y = extract_cnn_features_nodes(
                    test_chunk, full_edges, num_nodes, days, hidden
                )
                model = EpidemicCNN(K=n_steps, input_channels=8).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                with torch.no_grad():
                    pred = model(X.to(device)).cpu().numpy()
                mp, mq = calculate_mape(Y.numpy(), pred)
            else:
                mp, mq = np.nan, np.nan
            results['CNN_P_MAPE(%)'].append(mp)
            results['CNN_Q_MAPE(%)'].append(mq)
            print(f"  CNN P={mp:.2f}% Q={mq:.2f}%")

        if eval_gta:
            path = os.path.join(save_root, f"gta_noderobust_{ratio_str}.pth")
            if not os.path.exists(path):
                legacy = os.path.join(save_root, f"gtn_noderobust_{ratio_str}.pth")
                path = legacy if os.path.exists(legacy) else path
            if os.path.exists(path):
                loader = DataLoader(
                    StaticSIRDataset_Nodes(
                        test_chunk, gnn_edges, days, num_nodes, hidden
                    ),
                    batch_size=64, shuffle=False, collate_fn=gta_collate_fn,
                )
                model = GTA(
                    num_times=len(days), node_in_dim=4, d_model=256, nhead=8
                ).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
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
                mp, mq = calculate_mape(np.concatenate(trues), np.concatenate(preds))
            else:
                mp, mq = np.nan, np.nan
            results['GTA_P_MAPE(%)'].append(mp)
            results['GTA_Q_MAPE(%)'].append(mq)
            print(f"  GTA P={mp:.2f}% Q={mq:.2f}%")
        else:
            results['GTA_P_MAPE(%)'].append(np.nan)
            results['GTA_Q_MAPE(%)'].append(np.nan)

        hidden = update_hidden_mask(hidden, drop_n, seed=42 + i)
        gc.collect()

    report = pd.DataFrame(results)
    out = os.path.join(save_root, "node_robustness_mape_report.csv")
    report.to_csv(out, index=False)
    print(report.to_string(index=False, float_format="%.2f"))
    print(f"Saved: {out}")
    return report


if __name__ == "__main__":
    start_node_unified_evaluation(eval_xgb=True, eval_cnn=True, eval_gta=False)