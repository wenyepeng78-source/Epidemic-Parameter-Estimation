"""
Evaluate 1D-CNN with late-fusion MLE of q.
Reports MAPE for p (CNN), q (CNN-calibrated), and q (pure MLE).
"""

import os
import gc
import torch
import numpy as np
import pandas as pd

from .models import EpidemicCNN_MLE, get_mape
from .data import load_network_and_data, extract_cnn_features_mle


def run_evaluation_suite(net_name, tasks, DATA_ROOT, NETWORK_ROOT, SAVE_ROOT):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 60}")
    print(f"Evaluating network: {net_name} (CNN + MLE Late Fusion) | Device: {device}")
    print(f"{'=' * 60}")

    BASE_N, BASE_K = 1000, 4
    is_fixed_network = all(t['type'] not in ['N', 'K'] for t in tasks)
    fixed_data_chunk, fixed_edges, fixed_num_nodes = None, None, None

    if is_fixed_network:
        try:
            fixed_data_chunk, fixed_edges, fixed_num_nodes = load_network_and_data(
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
        n_steps = task.get('n_steps', 6)
        gap = task.get('gap', 8)
        val_name = f"n{n_steps}_gap{gap}"

        if is_fixed_network:
            data_chunk, edges, num_nodes = fixed_data_chunk, fixed_edges, fixed_num_nodes
        else:
            try:
                data_chunk, edges, num_nodes = load_network_and_data(
                    net_name, DATA_ROOT, NETWORK_ROOT, BASE_N, BASE_K
                )
            except Exception as e:
                print(f"Failed to load data: {e}")
                continue

        M = len(data_chunk)
        test_indices = [idx for i in range(0, M, 50) for idx in range(i + 45, min(i + 50, M))]
        test_data_chunk = [data_chunk[i] for i in test_indices]

        days = [gap * i for i in range(1, n_steps + 1)]
        X_test, Q_mle_test, Y_test = extract_cnn_features_mle(
            test_data_chunk, edges, num_nodes, days
        )

        model_file = f"cnn_latefusion_{val_name}.pth"
        model_path = os.path.join(exp_dir, model_file)
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_file}")
            continue

        model = EpidemicCNN_MLE(K=n_steps).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            preds = model(X_test.to(device), Q_mle_test.to(device)).cpu().numpy()
            preds_p = preds[:, 0]
            preds_q_cnn = preds[:, 1]

        y_true = Y_test.numpy()
        true_p, true_q = y_true[:, 0], y_true[:, 1]

        preds_q_mle = Q_mle_test.numpy().flatten()
        preds_q_mle_safe = np.where(preds_q_mle == 0.0, np.nan, preds_q_mle)

        m_p = get_mape(true_p, preds_p)
        m_q_cnn = get_mape(true_q, preds_q_cnn)
        m_q_mle = get_mape(true_q, preds_q_mle_safe)

        results.append({
            "Configuration": val_name,
            "Beta_MAPE (CNN)": m_p,
            "Gamma_MAPE (CNN calibrated)": m_q_cnn,
            "Gamma_MAPE (pure MLE)": m_q_mle
        })
        print(
            f"  [{val_name}] Beta MAPE: {m_p:.2%} | "
            f"Gamma MAPE (CNN): {m_q_cnn:.2%} | Gamma MAPE (MLE): {m_q_mle:.2%}"
        )

        if not is_fixed_network:
            del data_chunk, edges, test_data_chunk
        del X_test, Q_mle_test, Y_test
        gc.collect()

        if results:
            report_path = os.path.join(exp_dir, "Final_CNN_LateFusion_Evaluation.csv")
            pd.DataFrame(results).to_csv(report_path, index=False)
            print(f"  Evaluation report saved to: {report_path}\n")


if __name__ == "__main__":
    D_ROOT = '/content/drive/MyDrive/real-data'
    N_ROOT = '/content/drive/MyDrive/real_networks'
    S_ROOT = '/content/drive/MyDrive/real-CNN-mle1'

    network_names = ["Protein"]
    for name in network_names:
        tasks = [
            {"name": f"{name}_LateFusion_Fix", "type": "fixed", "n_steps": 6, "gap": 8},
        ]
        try:
            run_evaluation_suite(name, tasks, D_ROOT, N_ROOT, S_ROOT)
        except Exception as e:
            print(f"{name} failed: {e}")