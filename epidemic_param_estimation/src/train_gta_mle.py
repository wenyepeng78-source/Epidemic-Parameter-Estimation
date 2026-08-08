"""
Train GTA with late-fusion of an MLE estimate of q.
Predicts infection rate p only; q is provided by the MLE feature.
"""

import os
import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from .models import set_seed, GTA_MLE, MSLELossSingle
from .data import (
    load_gta_network_data,
    StaticSIRDataset_MLE,
    gta_mle_collate_fn
)

SEED = 520
set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training device ready: {device}")


def plot_experiment_curve(history, title, save_path):
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='#1f77b4')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', color='#d62728')
    plt.title(f'GTA Loss (Late Fusion): {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSLE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['lr'], color='#2ca02c')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_gnn_training(data_chunk, edges, num_nodes, days, save_path):
    dataset = StaticSIRDataset_MLE(data_chunk, edges, days, num_nodes)

    train_idx, val_idx = [], []
    for i in range(0, len(dataset), 50):
        train_idx.extend(range(i, min(i + 40, len(dataset))))
        val_idx.extend(range(i + 40, min(i + 45, len(dataset))))

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=64, shuffle=True,
        collate_fn=gta_mle_collate_fn, generator=g
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=64, shuffle=False,
        collate_fn=gta_mle_collate_fn, generator=g
    )

    model = GTA_MLE(
        num_times=len(days), node_in_dim=3, gnn_mid_dim=16, gnn_hidden_dim=32,
        gnn_num_layers=2, d_model=256, nhead=8, num_layers=1
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = MSLELossSingle()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    patience_cnt = 0

    for epoch in range(1, 201):
        model.train()
        train_sum = 0.0

        for x_list, edge_index_list, time_idx, p_true, q_mle in train_loader:
            x_list = [x.to(device) for x in x_list]
            edge_index_list = [e.to(device) for e in edge_index_list]
            time_idx = time_idx.to(device)
            p_true = p_true.to(device)
            q_mle = q_mle.to(device)

            optimizer.zero_grad()
            p_pred = model(x_list, edge_index_list, time_idx, q_mle)
            loss = criterion(p_pred, p_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.item()

        avg_train_loss = train_sum / len(train_loader)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for x_list, edge_index_list, time_idx, p_true, q_mle in val_loader:
                x_list = [x.to(device) for x in x_list]
                edge_index_list = [e.to(device) for e in edge_index_list]
                time_idx = time_idx.to(device)
                p_true = p_true.to(device)
                q_mle = q_mle.to(device)

                p_pred = model(x_list, edge_index_list, time_idx, q_mle)
                val_sum += criterion(p_pred, p_true).item()

        avg_val_loss = val_sum / len(val_loader)
        scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if epoch % 10 == 0:
            print(
                f"  Epoch [{epoch:03d}/200] | "
                f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= 15:
                print(f"  Early stopping triggered (Epoch {epoch})")
                break

    del train_loader, val_loader, model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_loss, history


def run_gnn_experiment_suite(net_name, tasks, DATA_ROOT, NETWORK_ROOT, SAVE_ROOT):
    BASE_N, BASE_K, BASE_GAP, BASE_NSTEPS = 1000, 4, 6, 6

    is_fixed_network = all(t['type'] not in ['N', 'K'] for t in tasks)
    fixed_data, fixed_edges, fixed_num_nodes = None, None, None
    if is_fixed_network:
        print("Using fixed network configuration - loading static data.")
        fixed_data, fixed_edges, fixed_num_nodes = load_gta_network_data(
            net_name, DATA_ROOT, NETWORK_ROOT, N=BASE_N, K_avg=BASE_K
        )

    for task in tasks:
        exp_dir = os.path.join(SAVE_ROOT, task['name'])
        os.makedirs(exp_dir, exist_ok=True)
        print(f"\nStarting experiment group: {task['name']}")

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

            days = [gap * i for i in range(1, n_steps + 1)]

            save_name = (
                f"best_model_gap{gap}_n{n_steps}.pth"
                if task['type'] in ['gap', 'n_steps']
                else f"best_model_{task['type']}_{val}.pth"
            )
            save_path = os.path.join(exp_dir, save_name)

            print(f"  [{task['type']}={val}] Observation days: {days}")
            best_v, history = run_gnn_training(data, edges, num_nodes, days, save_path)

            img_name = (
                f"learning_curve_gap{gap}_n{n_steps}.png"
                if task['type'] in ['gap', 'n_steps']
                else f"learning_curve_{task['type']}_{val}.png"
            )
            plot_experiment_curve(
                history, f"{task['type']}={val}", os.path.join(exp_dir, img_name)
            )

            print(f"  [Complete] {task['type']}={val} | Best Val Loss: {best_v:.6f}")

            if not is_fixed_network:
                del data, edges
            gc.collect()


if __name__ == "__main__":
    D_ROOT = '/content/drive/MyDrive/real-data'
    N_ROOT = '/content/drive/MyDrive/real_networks'
    S_ROOT = '/content/drive/MyDrive/real-GTA-mle6.14'

    network_names = ["Protein"]

    for net_name in network_names:
        print("\n" + "=" * 50)
        print(f"Starting Network: {net_name} (GTA + Late Fusion MLE)")
        print("=" * 50)

        my_tasks = [
            {"name": f"{net_name}_Var_n", "type": "n_steps", "values": [4, 6, 8, 10, 12]},
            {"name": f"{net_name}_Var_Gap", "type": "gap", "values": [4, 6, 8, 10, 12]},
        ]

        run_gnn_experiment_suite(net_name, my_tasks, D_ROOT, N_ROOT, S_ROOT)
        print(f"Network {net_name} processing complete!\n")