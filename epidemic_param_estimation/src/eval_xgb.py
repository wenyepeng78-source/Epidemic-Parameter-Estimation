"""
Evaluate multi-output XGBoost on held-out test trajectories.
"""

import os
import gc
import pickle
import pandas as pd

from .data import load_experiment_data_np, extract_xgb_features
from .models import get_mape


def run_evaluation_suite(net_name, tasks, DATA_ROOT, NETWORK_ROOT, SAVE_ROOT):
    BASE_N, BASE_K, BASE_GAP, BASE_NSTEPS = 1000, 4, 6, 8

    print(f"\n{'=' * 60}")
    print(f"Starting evaluation for network: {net_name}")
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
            N_val = val if task['type'] == 'N' else BASE_N
            K_val = val if task['type'] == 'K' else BASE_K
            gap = val if task['type'] == 'gap' else BASE_GAP
            n_steps = val if task['type'] == 'n_steps' else BASE_NSTEPS

            if is_fixed:
                data, edges, num_nodes = fixed_data, fixed_edges, fixed_n
            else:
                try:
                    data, edges, num_nodes = load_experiment_data_np(
                        net_name, DATA_ROOT, NETWORK_ROOT, N_val, K_val
                    )
                except Exception as e:
                    print(f"Failed to load {task['type']}={val}: {e}")
                    continue

            M = len(data)
            test_idx = []
            for i in range(0, M, 50):
                test_idx.extend(range(i + 45, min(i + 50, M)))

            days = [gap * i for i in range(1, n_steps + 1)]
            X_test, y_test = extract_xgb_features(
                data, test_idx, edges, num_nodes, days, f"TestFeat_{val}"
            )

            model_path = os.path.join(exp_dir, f"xgb_{task['type']}_{val}.pkl")
            if not os.path.exists(model_path):
                print(f"  Model not found: {model_path}")
                continue

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            preds = model.predict(X_test)
            mape_p = get_mape(y_test[:, 0], preds[:, 0])
            mape_q = get_mape(y_test[:, 1], preds[:, 1])
            eval_results.append({
                "Value": val,
                "Beta_MAPE": mape_p,
                "Gamma_MAPE": mape_q
            })
            print(f"  [{task['type']}={val}] Beta MAPE: {mape_p:.2%} | Gamma MAPE: {mape_q:.2%}")

            if not is_fixed:
                del data, edges
            del X_test, y_test, model
            gc.collect()

        if eval_results:
            df = pd.DataFrame(eval_results)
            df.columns = [task['type'], 'Beta_MAPE', 'Gamma_MAPE']
            report = os.path.join(exp_dir, f"Evaluation_Summary_{task['type']}.csv")
            df.to_csv(report, index=False)
            print(f"  Summary saved to: {report}\n")


if __name__ == "__main__":
    D_ROOT = '/content/drive/MyDrive/real-data'
    NETWORK_ROOT = '/content/drive/MyDrive/real_networks'
    S_ROOT = '/content/drive/MyDrive/real-xg-fix'

    for name in ["Protein"]:
        tasks = [
            {"name": f"{name}_Var_n", "type": "n_steps", "values": [5]},
        ]
        try:
            run_evaluation_suite(name, tasks, D_ROOT, NETWORK_ROOT, S_ROOT)
        except Exception as e:
            print(f"Network {name} evaluation interrupted: {e}")