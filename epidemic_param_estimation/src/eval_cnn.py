"""
Evaluate plain 1D-CNN on held-out test trajectories.
"""

import os
import gc
import torch
import pandas as pd

from .models import EpidemicCNN, get_mape
from .data import load_network_and_data, extract_cnn_features


def run_evaluation_suite(net_name, tasks, DATA_ROOT, NETWORK_ROOT, SAVE_ROOT):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nEvaluating network: {net_name} | Device: {device}")

    BASE_N, BASE_K, BASE_GAP, BASE_NSTEPS = 1000, 4, 2, 2
    is_fixed_network = all(t['type'] not in ['N', 'K'] for t in tasks)
    fixed_data_chunk, fixed_edges, fixed_num_nodes = None, None, None

    if is_fixed_network:
        try:
            fixed_data_chunk, fixed_edges, fixed_num_nodes = load_network_and_data(
                net_name, DATA_ROOT, NETWORK_ROOT
            )
        except Exception as e:
            print(f"Failed to load {net_name}: {e}")
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
                data_chunk, edges, num_nodes = fixed_data_chunk, fixed_edges, fixed_num_nodes
            else:
                try:
                    data_chunk, edges, num_nodes = load_network_and_data(
                        net_name, DATA_ROOT, NETWORK_ROOT, N_val, K_val
                    )
                except Exception as e:
                    print(f"Failed to load data for {task['type']}={val}: {e}")
                    continue

            M = len(data_chunk)
            test_indices = [idx for i in range(0, M, 50) for idx in range(i + 45, min(i + 50, M))]
            test_data_chunk = [data_chunk[i] for i in test_indices]

            days = [gap * i for i in range(1, n_steps + 1)]
            X_test, Y_test = extract_cnn_features(test_data_chunk, edges, num_nodes, days)

            model_path = os.path.join(exp_dir, f"cnn_{task['type']}_{val}.pth")
            if not os.path.exists(model_path):
                print(f"Weights not found: cnn_{task['type']}_{val}.pth")
                continue

            model = EpidemicCNN(K=n_steps).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            with torch.no_grad():
                preds = model(X_test.to(device)).cpu().numpy()

            y_true = Y_test.numpy()
            m_p = get_mape(y_true[:, 0], preds[:, 0])
            m_q = get_mape(y_true[:, 1], preds[:, 1])
            results.append({task['type']: val, "Beta_MAPE": m_p, "Gamma_MAPE": m_q})
            print(f"[{task['type']}={val}] Beta MAPE: {m_p:.2%} | Gamma MAPE: {m_q:.2%}")

            if not is_fixed_network:
                del data_chunk, edges, test_data_chunk
            del X_test, Y_test
            gc.collect()

        if results:
            report_path = os.path.join(exp_dir, f"Final_CNN_Evaluation_{task['type']}.csv")
            pd.DataFrame(results).to_csv(report_path, index=False)
            print(f"Evaluation report saved to: {report_path}")


if __name__ == "__main__":
    D_ROOT = '/content/drive/MyDrive/real-data'
    N_ROOT = '/content/drive/MyDrive/real_networks'
    S_ROOT = '/content/drive/MyDrive/real-fixed-CNN'

    network_names = ["Protein"]
    for name in network_names:
        tasks = [
            {"name": f"{name}_Var_Gap", "type": "gap", "values": [15]},
        ]
        run_evaluation_suite(name, tasks, D_ROOT, N_ROOT, S_ROOT)