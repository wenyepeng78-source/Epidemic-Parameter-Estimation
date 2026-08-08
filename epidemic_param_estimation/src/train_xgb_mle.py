"""
Train single-target XGBoost for p with late-fusion MLE of q.
"""

import os
import gc
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt

from .data import load_experiment_data_np, extract_xgb_features_mle


def run_single_train(X_train, y_train, X_val, y_val, save_path):
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=0.8,
        subsample=0.9,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',
        n_jobs=-1,
        random_state=1314
    )

    print(f"  Fitting model (Train: {len(X_train)}, Val: {len(X_val)})...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    results = model.evals_result()
    epochs = len(results['validation_0']['rmse'])
    x_axis = range(epochs)

    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, results['validation_0']['rmse'], label='Train RMSE', color='#79828D', linewidth=2)
    plt.plot(x_axis, results['validation_1']['rmse'], label='Val RMSE', color='#C39A92', linewidth=2)
    plt.legend(frameon=False)
    plt.title('XGBoost Learning Curve (Late Fusion with q_MLE)')
    plt.ylabel('RMSE')
    plt.xlabel('Iterations (Trees)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plot_path = save_path.replace('.pkl', '_loss.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print("  Training finished. Model and loss curve saved.")


def run_experiment_suite(net_name, tasks, DATA_ROOT, NETWORK_ROOT, SAVE_ROOT):
    BASE_N, BASE_K = 1000, 4
    BASE_GAP, BASE_NSTEPS = 6, 6

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
            n_steps = val if task['type'] == 'n_steps' else BASE_NSTEPS
            gap = val if task['type'] == 'gap' else BASE_GAP
            print(f"  Training config: {task['type']}={val} (n={n_steps}, gap={gap})")

            if is_fixed:
                data, edges, num_nodes = fixed_data, fixed_edges, fixed_n
            else:
                data, edges, num_nodes = load_experiment_data_np(
                    net_name, DATA_ROOT, NETWORK_ROOT, BASE_N, BASE_K
                )

            M = len(data)
            train_idx, val_idx = [], []
            for i in range(0, M, 50):
                train_idx.extend(range(i, min(i + 40, M)))
                val_idx.extend(range(i + 40, min(i + 45, M)))

            days = [gap * i for i in range(1, n_steps + 1)]
            X_train, y_train = extract_xgb_features_mle(
                data, train_idx, edges, num_nodes, days, f"Train Feat (val={val})", return_pq=False
            )
            X_val, y_val = extract_xgb_features_mle(
                data, val_idx, edges, num_nodes, days, f"Val Feat (val={val})", return_pq=False
            )

            save_name = f"xgb_latefusion_gap{gap}_n{n_steps}.pkl"
            run_single_train(X_train, y_train, X_val, y_val, os.path.join(exp_dir, save_name))

            if not is_fixed:
                del data, edges
            del X_train, y_train, X_val, y_val
            gc.collect()


if __name__ == "__main__":
    D_ROOT = '/content/drive/MyDrive/real-data'
    NETWORK_ROOT = '/content/drive/MyDrive/real_networks'
    S_ROOT = '/content/drive/MyDrive/real-xg-mle'

    for name in ["Protein"]:
        print("\n" + "=" * 50)
        print(f"Processing network: {name} (Late Fusion with MLE)")
        print("=" * 50)
        tasks = [
            {"name": f"{name}_Var_n", "type": "n_steps", "values": [4, 6, 8, 10, 12]},
            {"name": f"{name}_Var_Gap", "type": "gap", "values": [4, 6, 8, 10, 12]},
        ]
        try:
            run_experiment_suite(name, tasks, D_ROOT, NETWORK_ROOT, S_ROOT)
        except Exception as e:
            print(f"{name} training interrupted: {e}")