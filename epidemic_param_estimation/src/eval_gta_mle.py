"""
Evaluate GTA with late-fusion MLE of q.
Reports Beta MAPE (GTA) and Gamma MAPE (pure MLE).
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .models import set_seed, GTA_MLE, get_mape
from .data import (
    load_gta_network_data,
    EvalDataset_MLE,
    eval_mle_collate_fn
)

SEED = 520
set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_gnn_evaluation_suite(net_name, tasks, DATA_ROOT, NETWORK_ROOT, SAVE_ROOT):
    print(f"\nEvaluating network: {net_name} (GTA + MLE) | Device: {device}")

    BASE_N, BASE_K, BASE_GAP, BASE_NSTEPS = 1000, 4, 6, 6

    is_fixed_network = all(t['type'] not in ['N', 'K'] for t in tasks)
    fixed_data, fixed_edges, fixed_num_nodes = None, None, None
    if is_fixed_network:
        try:
            fixed_data, fixed_edges, fixed_num_nodes = load_gta_network_data(
                net_name, DATA_ROOT, NETWORK_ROOT
            )
        except Exception as e:
            print(f"Skipping {net_name}: {e}")
            return

    for task in tasks:
        exp_dir = os.path.join(SAVE_ROOT, task['name'])
        if not os.path.exists(exp_dir):
            print(f"Experiment directory not found: {task['name']}")
            continue

        results = []
        for val in task['values']:
            N_val = val if task['type'] == 'N' else BASE_N
            K_val = val if task['type'] == 'K' else BASE_K
            gap = val if task['type'] == 'gap' else BASE_GAP
            n_steps = val if task['type'] == 'n_steps' else BASE_NSTEPS

            if is_fixed_network:
                data_chunk, edges, num_nodes = fixed_data, fixed_edges, fixed_num_nodes
            else:
                try:
                    data_chunk, edges, num_nodes = load_gta_network_data(
                        net_name, DATA_ROOT, NETWORK_ROOT, N_val, K_val
                    )
                except Exception as e:
                    print(f"Failed to load data: {e}")
                    continue

            M = len(data_chunk)
            test_indices = [
                idx for i in range(0, M, 50) for idx in range(i + 45, min(i + 50, M))
            ]
            test_data_chunk = [data_chunk[i] for i in test_indices]

            days = [gap * i for i in range(1, n_steps + 1)]

            eval_dataset = EvalDataset_MLE(test_data_chunk, edges, days, num_nodes)
            eval_loader = DataLoader(
                eval_dataset, batch_size=64, shuffle=False, collate_fn=eval_mle_collate_fn
            )

            save_name = (
                f"best_model_gap{gap}_n{n_steps}.pth"
                if task['type'] in ['gap', 'n_steps']
                else f"best_model_{task['type']}_{val}.pth"
            )
            model_path = os.path.join(exp_dir, save_name)

            if not os.path.exists(model_path):
                print(f"  Weight file not found: {save_name}")
                continue

            model = GTA_MLE(
                num_times=len(days), node_in_dim=3, gnn_mid_dim=16, gnn_hidden_dim=32,
                gnn_num_layers=2, d_model=256, nhead=8, num_layers=1
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            preds_p, preds_q_mle, trues_p, trues_q = [], [], [], []

            with torch.no_grad():
                for x_list, edge_index_list, time_idx, theta_true, q_mle in eval_loader:
                    x_list = [x.to(device) for x in x_list]
                    edge_index_list = [e.to(device) for e in edge_index_list]
                    time_idx = time_idx.to(device)
                    q_mle = q_mle.to(device)

                    p_pred = model(x_list, edge_index_list, time_idx, q_mle)

                    preds_p.extend(p_pred.cpu().numpy().flatten())
                    preds_q_mle.extend(q_mle.cpu().numpy().flatten())
                    trues_p.extend(theta_true[:, 0].numpy())
                    trues_q.extend(theta_true[:, 1].numpy())

            preds_p = np.array(preds_p)
            trues_p = np.array(trues_p)
            trues_q = np.array(trues_q)
            preds_q_mle = np.array(preds_q_mle)
            preds_q_mle_safe = np.where(preds_q_mle == 0.0, np.nan, preds_q_mle)

            m_p = get_mape(trues_p, preds_p)
            m_q = get_mape(trues_q, preds_q_mle_safe)

            results.append({
                "Configuration": f"{task['type']}={val}",
                "Beta_MAPE (GNN)": m_p,
                "Gamma_MAPE (MLE)": m_q
            })
            print(
                f"  [{task['type']}={val}] "
                f"Beta MAPE (GNN): {m_p:.2%} | Gamma MAPE (MLE): {m_q:.2%}"
            )

            if not is_fixed_network:
                del data_chunk, edges, test_data_chunk
            del eval_dataset, eval_loader, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if results:
            report_path = os.path.join(exp_dir, f"Final_GNN_Evaluation_{task['type']}.csv")
            pd.DataFrame(results).to_csv(report_path, index=False)
            print(f"  Evaluation results saved: {report_path}\n")


if __name__ == "__main__":
    D_ROOT = '/content/drive/MyDrive/real-data'
    N_ROOT = '/content/drive/MyDrive/real_networks'
    S_ROOT = '/content/drive/MyDrive/real-GTA-mle6.15'

    network_names = ["Protein"]

    for net_name in network_names:
        tasks = [
            {"name": f"{net_name}_Var_n", "type": "n_steps", "values": [4, 6, 8, 10, 12]},
            {"name": f"{net_name}_Var_Gap", "type": "gap", "values": [4, 6, 8, 10, 12]},
        ]
        try:
            run_gnn_evaluation_suite(net_name, tasks, D_ROOT, N_ROOT, S_ROOT)
        except Exception as e:
            print(f"Execution failed for {net_name}: {e}")