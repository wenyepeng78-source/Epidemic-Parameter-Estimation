"""
Train multi-output XGBoost for joint estimation of (p, q).
"""

import os
import gc
import pickle
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from .data import load_experiment_data_np, extract_xgb_features


def run_single_train(X_train, y_train, X_val, y_val, save_path):
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
        random_state=520
    )

    print(f"  Fitting model (Train: {len(X_train)}, Val: {len(X_val)})...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    results = model.evals_result()
    epochs = len(results['validation_0']['rmsle'])
    x_axis = range(epochs)

    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, results['validation_0']['rmsle'], label='Train RMSLE')
    plt.plot(x_axis, results['validation_1']['rmsle'], label='Val RMSLE')
    plt.legend()
    plt.title('XGBoost Multi-output Learning Curve')
    plt.ylabel('RMSLE')
    plt.xlabel('Iterations (Trees)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_path = save_path.replace('.pkl', '_loss.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print("  Training finished. Model and loss curve saved.")


def run_experiment_suite(net_name, tasks, DATA_ROOT, NETWORK_ROOT, SAVE_ROOT):
    BASE_N, BASE_K, BASE_GAP, BASE_NSTEPS = 1000, 4, 6, 8

    is_fixed = all(t['type'] not in ['N', 'K'] for t in tasks)
    fixed_data = fixed_edges = fixed_n = None
    if is_fixed:
        print("Fixed network detected. Loading data once.")
        fixed_data, fixed_edges, fixed_n = load_experiment_data_np(net_name, DATA_ROOT, NETWORK_ROOT)

    for task in tasks:
        exp_dir = os.path.join(SAVE_ROOT, task['name'])
        os.makedirs(exp_dir, exist_ok=True)
        print(f"\nStarting experiment group: {task['name']}")

        for val in task['values']:
            N_val = val if task['type'] == 'N' else BASE_N
            K_val = val if task['type'] == 'K' else BASE_K
            gap = val if task['type'] == 'gap' else BASE_GAP
            n_steps = val if task['type'] == 'n_steps' else BASE_NSTEPS

            if is_fixed:
                data, edges, num_nodes = fixed_data, fixed_edges, fixed_n
            else:
                data, edges, num_nodes = load_experiment_data_np(
                    net_name, DATA_ROOT, NETWORK_ROOT, N_val, K_val
                )

            M = len(data)
            train_idx, val_idx = [], []
            for i in range(0, M, 50):
                train_idx.extend(range(i, min(i + 40, M)))
                val_idx.extend(range(i + 40, min(i + 45, M)))

            days = [gap * i for i in range(1, n_steps + 1)]
            print(f"  [{task['type']}={val}] days={days}")

            X_train, y_train = extract_xgb_features(data, train_idx, edges, num_nodes, days, "Train Feat")
            X_val, y_val = extract_xgb_features(data, val_idx, edges, num_nodes, days, "Val Feat")

            save_name = f"xgb_{task['type']}_{val}.pkl"
            run_single_train(X_train, y_train, X_val, y_val, os.path.join(exp_dir, save_name))

            if not is_fixed:
                del data, edges
            del X_train, y_train, X_val, y_val
            gc.collect()


if __name__ == "__main__":
    D_ROOT = '/content/drive/MyDrive/real-data'
    NETWORK_ROOT = '/content/drive/MyDrive/real_networks'
    S_ROOT = '/content/drive/MyDrive/real-xg-fix'

    for name in ["Protein"]:
        print("\n" + "=" * 50)
        print(f"Processing network: {name}")
        print("=" * 50)
        tasks = [
            {"name": f"{name}_Var_n", "type": "n_steps", "values": [5]},
        ]
        try:
            run_experiment_suite(name, tasks, D_ROOT, NETWORK_ROOT, S_ROOT)
        except Exception as e:
            print(f"{name} training interrupted: {e}")






            #         tasks = [
#     {"name": f"{name}_FixWindow", "type": "pair", "values": [
#         (30, 1), (15, 2), (10, 3), (6, 5), (5, 6), (3, 10), (2, 15)
#     ]},
# ]