"""
Evaluate plain GTA on held-out test trajectories.
Reports Beta/Gamma/R0 MAPE and Beta/Gamma MSE.
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .models import set_seed, GTA, get_mape
from .data import load_gta_network_data, StaticSIRDataset, gta_collate_fn

SEED = 520
set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_gnn_evaluation_suite(net_name, tasks, DATA_ROOT, NETWORK_ROOT, SAVE_ROOT):
    print(f"\nStarting evaluation | Device: {device}")

    BASE_N, BASE_K, BASE_GAP, BASE_NSTEPS = 1000, 4, 15, 2

    is_fixed_network = all(t['type'] not in ['N', 'K'] for t in tasks)
    fixed_data, fixed_edges, fixed_num_nodes = None, None, None
    if is_fixed_network:
        fixed_data, fixed_edges, fixed_num_nodes = load_gta_network_data(
            net_name, DATA_ROOT, NETWORK_ROOT, N=BASE_N, K_avg=BASE_K
        )

    for task in tasks:
        exp_dir = os.path.join(SAVE_ROOT, task['name'])
        if not os.path.exists(exp_dir):
            print(f"Skipping group {task['name']}: Model directory not found.")
            continue

        print(f"\nEvaluating group: {task['name']}")
        group_results = []

        for val in task['values']:
            N_val = val if task['type'] == 'N' else BASE_N
            K_val = val if task['type'] == 'K' else BASE_K
            gap = val if task['type'] == 'gap' else BASE_GAP
            n_steps = val if task['type'] == 'n_steps' else BASE_NSTEPS

            if is_fixed_network:
                data, edges, num_nodes = fixed_data, fixed_edges, fixed_num_nodes
            else:
                try:
                    data, edges, num_nodes = load_gta_network_data(
                        net_name, DATA_ROOT, NETWORK_ROOT, N=N_val, K_avg=K_val
                    )
                except Exception as e:
                    print(f"Failed to load data: {e}")
                    continue

            # Test indices: last 5 of every block of 50
            M = len(data)
            test_indices = []
            for i in range(0, M, 50):
                test_indices.extend(list(range(i + 45, min(i + 50, M))))

            test_data_chunk = [data[i] for i in test_indices]
            days = [gap * i for i in range(1, n_steps + 1)]

            test_dataset = StaticSIRDataset(test_data_chunk, edges, days, num_nodes)
            test_loader = DataLoader(
                test_dataset, batch_size=64, shuffle=False, collate_fn=gta_collate_fn
            )

            model_file = (
                f"best_model_gap{gap}_n{n_steps}.pth"
                if task['type'] in ['gap', 'n_steps']
                else f"best_model_{task['type']}_{val}.pth"
            )
            model_path = os.path.join(exp_dir, model_file)

            if not os.path.exists(model_path):
                print(f"Model not found: {model_file}")
                continue

            model = GTA(
                num_times=len(days), node_in_dim=3, gnn_mid_dim=16, gnn_hidden_dim=32,
                gnn_num_layers=2, d_model=256, nhead=8, num_layers=1
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            all_preds, all_trues = [], []
            with torch.no_grad():
                for x_list, edge_index_list, time_idx, theta_true in test_loader:
                    x_list = [x.to(device) for x in x_list]
                    edge_index_list = [e.to(device) for e in edge_index_list]
                    time_idx = time_idx.to(device)

                    preds = model(x_list, edge_index_list, time_idx)
                    all_preds.append(preds.cpu().numpy())
                    all_trues.append(theta_true.numpy())

            y_pred_np = np.concatenate(all_preds, axis=0)
            y_true_np = np.concatenate(all_trues, axis=0)

            mape_p = get_mape(y_true_np[:, 0], y_pred_np[:, 0])
            mape_q = get_mape(y_true_np[:, 1], y_pred_np[:, 1])

            r0_true = y_true_np[:, 0] / np.where(y_true_np[:, 1] == 0, 1e-7, y_true_np[:, 1])
            r0_pred = y_pred_np[:, 0] / np.where(y_pred_np[:, 1] == 0, 1e-7, y_pred_np[:, 1])
            mape_r0 = get_mape(r0_true, r0_pred)

            mse_p = np.mean((y_true_np[:, 0] - y_pred_np[:, 0]) ** 2)
            mse_q = np.mean((y_true_np[:, 1] - y_pred_np[:, 1]) ** 2)

            group_results.append({
                task['type']: val,
                "Beta_MAPE": mape_p,
                "Gamma_MAPE": mape_q,
                "R0_MAPE": mape_r0,
                "Beta_MSE": mse_p,
                "Gamma_MSE": mse_q
            })

            print(
                f"[Complete] {task['type']}={val} | "
                f"Beta(p) MAPE: {mape_p:.2%} | Gamma(q) MAPE: {mape_q:.2%} | R0 MAPE: {mape_r0:.2%}"
            )

            if not is_fixed_network:
                del data, edges
            del test_data_chunk, test_dataset, test_loader, model, all_preds, all_trues
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if group_results:
            df_report = pd.DataFrame(group_results)
            report_path = os.path.join(exp_dir, f"Final_GNN_Evaluation_{task['type']}.csv")
            df_report.to_csv(report_path, index=False)
            print(f"Evaluation report saved to: {report_path}")


if __name__ == "__main__":
    D_ROOT = '/content/drive/MyDrive/real-data'
    N_ROOT = '/content/drive/MyDrive/real_networks'
    S_ROOT = '/content/drive/MyDrive/real-fixed-GTA'

    network_names = ["Protein"]

    for net_name in network_names:
        my_tasks = [
            {"name": f"{net_name}_Var_Gap", "type": "gap", "values": [15]},
        ]
        run_gnn_evaluation_suite(net_name, my_tasks, D_ROOT, N_ROOT, S_ROOT)