"""
Evaluate Only-I observation robustness across n_steps.
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
    extract_features_only_i,
    StaticSIRDataset_OnlyI,
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


def start_time_robustness_evaluation(eval_xgb=True, eval_cnn=True, eval_gta=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gap = 3
    n_steps_list = [2, 4, 8, 16, 32]

    data_path = '/content/drive/MyDrive/real-data/Protein_optimized.pkl.gz'
    csv_path = '/content/drive/MyDrive/real_networks/Protein.csv'
    save_root = '/content/drive/MyDrive/OnlyI_Robustness'

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

    test_idx = []
    for i in range(0, M, 50):
        test_idx.extend(range(i + 45, min(i + 50, M)))
    test_chunk = [data_chunk[j] for j in test_idx]

    results = {
        'n_steps': [],
        'XGB_P_MAPE(%)': [], 'XGB_Q_MAPE(%)': [],
        'CNN_P_MAPE(%)': [], 'CNN_Q_MAPE(%)': [],
        'GTA_P_MAPE(%)': [], 'GTA_Q_MAPE(%)': [],
    }

    for n_steps in n_steps_list:
        days = [gap * i for i in range(1, n_steps + 1)]
        results['n_steps'].append(n_steps)
        print(f"\nn_steps={n_steps} | span={days[-1]}")

        X_xgb, X_cnn, Y = extract_features_only_i(
            data_chunk, test_idx, full_edges, num_nodes, days, "Eval"
        )

        if eval_xgb:
            path = os.path.join(save_root, f"xgb_time_robust_n{n_steps}.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                mp, mq = calculate_mape(Y.numpy(), model.predict(X_xgb))
            else:
                mp, mq = np.nan, np.nan
            results['XGB_P_MAPE(%)'].append(mp)
            results['XGB_Q_MAPE(%)'].append(mq)
            print(f"  XGB P={mp:.2f}% Q={mq:.2f}%")

        if eval_cnn:
            path = os.path.join(save_root, f"cnn_time_robust_n{n_steps}.pth")
            if os.path.exists(path):
                model = EpidemicCNN(K=n_steps, input_channels=4).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                with torch.no_grad():
                    pred = model(X_cnn.to(device)).cpu().numpy()
                mp, mq = calculate_mape(Y.numpy(), pred)
            else:
                mp, mq = np.nan, np.nan
            results['CNN_P_MAPE(%)'].append(mp)
            results['CNN_Q_MAPE(%)'].append(mq)
            print(f"  CNN P={mp:.2f}% Q={mq:.2f}%")

        if eval_gta:
            path = os.path.join(save_root, f"gta_time_robust_n{n_steps}.pth")
            if not os.path.exists(path):
                legacy = os.path.join(save_root, f"gtn_time_robust_n{n_steps}.pth")
                path = legacy if os.path.exists(legacy) else path
            if os.path.exists(path):
                loader = DataLoader(
                    StaticSIRDataset_OnlyI(test_chunk, gnn_edges, days, num_nodes),
                    batch_size=64, shuffle=False, collate_fn=gta_collate_fn,
                )
                model = GTA(
                    num_times=len(days), node_in_dim=2, d_model=256, nhead=8
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

        gc.collect()

    report = pd.DataFrame(results)
    out = os.path.join(save_root, "time_robustness_mape_report.csv")
    report.to_csv(out, index=False)
    print(report.to_string(index=False, float_format="%.2f"))
    print(f"Saved: {out}")
    return report


if __name__ == "__main__":
    start_time_robustness_evaluation(eval_xgb=True, eval_cnn=True, eval_gta=False)