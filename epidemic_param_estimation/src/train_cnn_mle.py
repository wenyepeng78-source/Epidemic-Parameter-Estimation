"""
Train 1D-CNN with late-fusion of an MLE estimate of q.
"""

import os
import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .models import set_seed, MSLELoss, EpidemicCNN_MLE
from .data import load_network_and_data, extract_cnn_features_mle

set_seed(520)


def run_cnn_training(X, Q_mle, Y, n_steps, save_path, device):
    """Train EpidemicCNN_MLE with early stopping on validation MSLE."""
    dataset = TensorDataset(X, Q_mle, Y)
    train_idx, val_idx = [], []
    for i in range(0, len(dataset), 50):
        train_idx.extend(range(i, min(i + 40, len(dataset))))
        val_idx.extend(range(i + 40, min(i + 45, len(dataset))))

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64)

    model = EpidemicCNN_MLE(K=n_steps).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = MSLELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    patience_cnt = 0
    MAX_PATIENCE = 15

    for epoch in range(1, 201):
        model.train()
        for bx, bq, by in train_loader:
            bx, bq, by = bx.to(device), bq.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx, bq), by)
            loss.backward()
            optimizer.step()

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for bx, bq, by in val_loader:
                v_loss += criterion(
                    model(bx.to(device), bq.to(device)), by.to(device)
                ).item()
        avg_v_loss = v_loss / len(val_loader)
        scheduler.step(avg_v_loss)

        if avg_v_loss < best_loss:
            best_loss = avg_v_loss
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= MAX_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    return best_loss


def run_experiment_suite(net_name, tasks, DATA_ROOT, NETWORK_ROOT, SAVE_ROOT):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_N, BASE_K = 1000, 4

    is_fixed_network = all(t['type'] not in ['N', 'K'] for t in tasks)
    fixed_data, fixed_edges, fixed_num_nodes = None, None, None

    if is_fixed_network:
        print("Fixed network detected. Loading data once for reuse.")
        fixed_data, fixed_edges, fixed_num_nodes = load_network_and_data(
            net_name, DATA_ROOT, NETWORK_ROOT
        )

    for task in tasks:
        exp_dir = os.path.join(SAVE_ROOT, task['name'])
        os.makedirs(exp_dir, exist_ok=True)
        print(f"\nStarting experiment group: {task['name']}")

        n_steps = task.get('n_steps', 6)
        gap = task.get('gap', 8)

        if is_fixed_network:
            data, edges, num_nodes = fixed_data, fixed_edges, fixed_num_nodes
        else:
            data, edges, num_nodes = load_network_and_data(
                net_name, DATA_ROOT, NETWORK_ROOT, BASE_N, BASE_K
            )

        days = [gap * i for i in range(1, n_steps + 1)]
        print(f"  Config: n_steps={n_steps}, gap={gap} -> days={days}")

        X, Q_mle, Y = extract_cnn_features_mle(data, edges, num_nodes, days)
        save_name = f"cnn_latefusion_n{n_steps}_gap{gap}.pth"
        best_loss = run_cnn_training(
            X, Q_mle, Y, n_steps, os.path.join(exp_dir, save_name), device
        )
        print(f"  Finished [n={n_steps}, gap={gap}] | Best Val Loss: {best_loss:.6f}")

        if not is_fixed_network:
            del data, edges
        del X, Q_mle, Y
        gc.collect()


if __name__ == "__main__":
    D_ROOT = '/content/drive/MyDrive/real-data'
    N_ROOT = '/content/drive/MyDrive/real_networks'
    S_ROOT = '/content/drive/MyDrive/real-CNN-mle1'

    network_names = ["Protein"]
    for name in network_names:
        print("\n" + "=" * 50)
        print(f"Processing network: {name} (CNN + MLE Late Fusion)")
        print("=" * 50)

        tasks = [
            {"name": f"{name}_LateFusion_Fix", "type": "fixed", "n_steps": 6, "gap": 8},
        ]
        try:
            run_experiment_suite(name, tasks, D_ROOT, N_ROOT, S_ROOT)
        except Exception as e:
            print(f"{name} failed: {e}")