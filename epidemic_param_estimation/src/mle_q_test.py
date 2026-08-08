"""
MLE baseline for recovery rate q.

Two experiment axes on the held-out test split (indices 45-49 per block of 50):
  1. Fixed gap, vary n_steps
  2. Fixed n_steps, vary gap
"""

import os
import gzip
import pickle

import numpy as np
import pandas as pd

from .models import estimate_q_mle


DATA_PATH = '/content/drive/MyDrive/real-data/Protein_optimized.pkl.gz'
SAVE_ROOT = '/content/drive/MyDrive/MLE_baseline'
os.makedirs(SAVE_ROOT, exist_ok=True)


def load_test_trajectories(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    with gzip.open(data_path, 'rb') as f:
        saved = pickle.load(f)

    trajectories = saved['trajectories']
    M = len(trajectories)
    test_indices = [
        idx
        for i in range(0, M, 50)
        for idx in range(i + 45, min(i + 50, M))
    ]
    test_traj = [trajectories[i] for i in test_indices]
    print(f"Loaded {M} trajectories | test size={len(test_traj)}")
    return test_traj


def evaluate_mle(test_traj, days, dt):
    """
    Return dict with valid count, MAE, MAPE (%) for q estimates.
    """
    abs_errors, percent_errors = [], []
    max_day = max(days)

    for (real_p, real_q), history_matrix in test_traj:
        if max_day >= history_matrix.shape[0]:
            continue

        q_hat, _, _ = estimate_q_mle(history_matrix, days, dt)
        if q_hat is None:
            continue

        abs_err = abs(q_hat - real_q)
        perc_err = abs_err / max(real_q, 1e-7)
        abs_errors.append(abs_err)
        percent_errors.append(perc_err)

    n = len(abs_errors)
    if n == 0:
        return {'valid': 0, 'mae': np.nan, 'mape': np.nan}

    return {
        'valid': n,
        'mae': float(np.mean(abs_errors)),
        'mape': float(np.mean(percent_errors) * 100),
    }


def run_vary_n_steps(test_traj, gap=6, n_steps_list=None):
    if n_steps_list is None:
        n_steps_list = [4, 6, 8, 10, 12]

    rows = []
    print(f"\n=== MLE: fixed gap={gap}, vary n_steps ===")
    for n_steps in n_steps_list:
        days = [gap * i for i in range(1, n_steps + 1)]
        res = evaluate_mle(test_traj, days, dt=gap)
        rows.append({
            'axis': 'n_steps',
            'value': n_steps,
            'gap': gap,
            'n_steps': n_steps,
            'valid': res['valid'],
            'MAE': res['mae'],
            'MAPE(%)': res['mape'],
        })
        print(
            f"  n_steps={n_steps} | days={days} | "
            f"valid={res['valid']} | MAE={res['mae']:.5f} | MAPE={res['mape']:.2f}%"
        )
    return rows


def run_vary_gap(test_traj, n_steps=6, gap_list=None):
    if gap_list is None:
        gap_list = [2, 4, 6, 8, 10]

    rows = []
    print(f"\n=== MLE: fixed n_steps={n_steps}, vary gap ===")
    for gap in gap_list:
        days = [gap * i for i in range(1, n_steps + 1)]
        res = evaluate_mle(test_traj, days, dt=gap)
        rows.append({
            'axis': 'gap',
            'value': gap,
            'gap': gap,
            'n_steps': n_steps,
            'valid': res['valid'],
            'MAE': res['mae'],
            'MAPE(%)': res['mape'],
        })
        print(
            f"  gap={gap} | days={days} | "
            f"valid={res['valid']} | MAE={res['mae']:.5f} | MAPE={res['mape']:.2f}%"
        )
    return rows


def main():
    test_traj = load_test_trajectories(DATA_PATH)

    rows = []
    rows.extend(run_vary_n_steps(test_traj, gap=6, n_steps_list=[4, 6, 8, 10, 12]))
    rows.extend(run_vary_gap(test_traj, n_steps=15, gap_list=[2, 4, 6, 8, 10]))

    df = pd.DataFrame(rows)
    out = os.path.join(SAVE_ROOT, "mle_baseline_report.csv")
    df.to_csv(out, index=False)

    print("\n" + "=" * 70)
    print("MLE baseline summary")
    print("=" * 70)
    print(df.to_string(index=False, float_format="%.4f"))
    print(f"\nReport saved to: {out}")


if __name__ == "__main__":
    main()