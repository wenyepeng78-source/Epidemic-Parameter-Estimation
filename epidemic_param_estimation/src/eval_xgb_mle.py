"""
Evaluate XGBoost + late-fusion MLE.
Reports MAPE for p (XGBoost) and q (pure MLE feature).
"""

import os
import gc
import pickle
import numpy as np
import pandas as pd

from .data import load_experiment_data_np, extract_xgb_features_mle
from .models import get_mape


def run_evaluation_suite(net_name, tasks, DATA_ROOT, NETWORK_ROOT, SAVE_ROOT):
    BASE_N, BASE_K = 1000, 4
    BASE_GAP, BASE_NSTEPS = 6, 6

    print(f"\n{'=' * 60}")
    print(f"Starting evaluation for network: {net_name} (Late Fusion with MLE)")
    print(f"{'=' * 60}")

    is_fixed = all(t['type'] not in ['N', 'K'] for t in tasks)
    fixed_data = fixed_edges = fixed_n = None
    if is_fixed:
        print("Fixed network detected. Loading data once.")
        try:
            fixed_data, fixed_edges, fixed_n = load_experiment_data_np(net_name, DATA_ROOT, NETWORK_ROOT)
        except Exception as e:
            print(f"Skipping {net_name}: {e}")
            return

    for task in tasks:
        exp_dir = os.path.join(SAVE_ROOT, task['name'])
        if not os.path.exists(exp_dir):
            print(f"Experiment directory not found: {task['name']}")
            continue

        eval_results = []
        for val in task['values']:
            n_steps = val if task['type'] == 'n_steps' else BASE_NSTEPS
            gap = val if task['type'] == 'gap' else BASE_GAP
            val_name = f"gap{gap}_n{n_steps}"
            print(f"  Evaluating config: {task['type']}={val} (n={n_steps}, gap={gap})")

            if is_fixed:
                data, edges, num_nodes = fixed_data, fixed_edges, fixed_n
            else:
                try:
                    data, edges, num_nodes = load_experiment_data_np(
                        net_name, DATA_ROOT, NETWORK_ROOT, BASE_N, BASE_K
                    )
                except Exception as e:
                    print(f"Failed to load data: {e}")
                    continue

            M = len(data)
            test_idx = []
            for i in range(0, M, 50):
                test_idx.extend(range(i + 45, min(i + 50, M)))

            days = [gap * i for i in range(1, n_steps + 1)]
            X_test, y_test = extract_xgb_features_mle(
                data, test_idx, edges, num_nodes, days, f"TestFeat_{val_name}", return_pq=True
            )

            model_path = os.path.join(exp_dir, f"xgb_latefusion_{val_name}.pkl")
            if not os.path.exists(model_path):
                print(f"  Model not found: {model_path}")
                continue

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            preds_p = model.predict(X_test)
            mape_p = get_mape(y_test[:, 0], preds_p)

            # q_mle is the 8th feature of the first time step (index 7)
            preds_q_mle = X_test[:, 7]
            mape_q = get_mape(y_test[:, 1], preds_q_mle)

            eval_results.append({
                "Variable": task['type'],
                "Value": val,
                "n_steps": n_steps,
                "gap": gap,
                "Beta_MAPE (XGBoost)": mape_p,
                "Gamma_MAPE (MLE)": mape_q
            })
            print(f"  [Result] Beta MAPE: {mape_p:.2%} | Gamma MAPE: {mape_q:.2%}")

            if not is_fixed:
                del data, edges
            del X_test, y_test, model
            gc.collect()

        if eval_results:
            df = pd.DataFrame(eval_results)
            report = os.path.join(exp_dir, f"Evaluation_{task['name']}.csv")
            df.to_csv(report, index=False)
            print(f"  Summary saved to: {report}\n")


if __name__ == "__main__":
    D_ROOT = '/content/drive/MyDrive/real-data'
    NETWORK_ROOT = '/content/drive/MyDrive/real_networks'
    S_ROOT = '/content/drive/MyDrive/real-xg-mle'

    for name in ["Protein"]:
        tasks = [
            {"name": f"{name}_Var_n", "type": "n_steps", "values": [4, 6, 8, 10, 12]},
            {"name": f"{name}_Var_Gap", "type": "gap", "values": [4, 6, 8, 10, 12]},
        ]
        try:
            run_evaluation_suite(name, tasks, D_ROOT, NETWORK_ROOT, S_ROOT)
        except Exception as e:
            print(f"Network {name} evaluation interrupted: {e}")