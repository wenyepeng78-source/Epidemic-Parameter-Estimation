"""Train plain GTA for joint (p, q) estimation."""

import os, gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from .models import set_seed, MSLELoss, GTA
from .data import load_gta_network_data, StaticSIRDataset, gta_collate_fn

SEED = 520
set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_gnn_training(data_chunk, edges, num_nodes, days, save_path):
    dataset = StaticSIRDataset(data_chunk, edges, days, num_nodes)
    train_idx, val_idx = [], []
    for i in range(0, len(dataset), 50):
        train_idx.extend(range(i, min(i + 40, len(dataset))))
        val_idx.extend(range(i + 40, min(i + 45, len(dataset))))

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=64, shuffle=True,
        collate_fn=gta_collate_fn, generator=g
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=64, shuffle=False, collate_fn=gta_collate_fn
    )

    model = GTA(
        num_times=len(days), d_model=256, nhead=8, num_layers=1
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    criterion = MSLELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss, patience_cnt = float('inf'), 0

    for epoch in range(1, 201):
        model.train()
        train_sum = 0.0
        for x_list, edge_list, time_idx, theta in train_loader:
            x_list = [x.to(device) for x in x_list]
            edge_list = [e.to(device) for e in edge_list]
            time_idx, theta = time_idx.to(device), theta.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_list, edge_list, time_idx), theta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.item()
        avg_train = train_sum / len(train_loader)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for x_list, edge_list, time_idx, theta in val_loader:
                x_list = [x.to(device) for x in x_list]
                edge_list = [e.to(device) for e in edge_list]
                val_sum += criterion(
                    model(x_list, edge_list, time_idx.to(device)), theta.to(device)
                ).item()
        avg_val = val_sum / len(val_loader)
        scheduler.step(avg_val)

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if epoch % 10 == 0:
            print(f"  Epoch [{epoch:03d}] Train {avg_train:.6f} | Val {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss, patience_cnt = avg_val, 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= 15:
                print(f"  Early stopping at epoch {epoch}")
                break

    del train_loader, val_loader, model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_val_loss, history


def run_gnn_experiment_suite(net_name, tasks, DATA_ROOT, NETWORK_ROOT, SAVE_ROOT):
    BASE_N, BASE_K, BASE_GAP, BASE_NSTEPS = 1000, 4, 6, 2
    is_fixed = all(t['type'] not in ['N', 'K'] for t in tasks)
    fixed = None
    if is_fixed:
        print("Fixed network detected.")
        fixed = load_gta_network_data(net_name, DATA_ROOT, NETWORK_ROOT, BASE_N, BASE_K)

    for task in tasks:
        exp_dir = os.path.join(SAVE_ROOT, task['name'])
        os.makedirs(exp_dir, exist_ok=True)
        print(f"\nStarting: {task['name']}")

        for val in task['values']:
            N_val = val if task['type'] == 'N' else BASE_N
            K_val = val if task['type'] == 'K' else BASE_K
            gap = val if task['type'] == 'gap' else BASE_GAP
            n_steps = val if task['type'] == 'n_steps' else BASE_NSTEPS

            if is_fixed:
                data, edges, num_nodes = fixed
            else:
                data, edges, num_nodes = load_gta_network_data(
                    net_name, DATA_ROOT, NETWORK_ROOT, N_val, K_val
                )

            days = [gap * i for i in range(1, n_steps + 1)]
            save_name = (
                f"best_model_gap{gap}_n{n_steps}.pth"
                if task['type'] in ['gap', 'n_steps']
                else f"best_model_{task['type']}_{val}.pth"
            )
            save_path = os.path.join(exp_dir, save_name)
            print(f"  [{task['type']}={val}] days={days}")
            best_v, history = run_gnn_training(data, edges, num_nodes, days, save_path)

            # optional: save learning curve
            img = os.path.join(
                exp_dir,
                f"learning_curve_gap{gap}_n{n_steps}.png"
                if task['type'] in ['gap', 'n_steps']
                else f"learning_curve_{task['type']}_{val}.png"
            )
            epochs_range = range(1, len(history['train_loss']) + 1)
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, history['train_loss'], label='Train')
            plt.plot(epochs_range, history['val_loss'], label='Val')
            plt.legend(); plt.title(f'GTA Loss {task["type"]}={val}'); plt.grid(True, alpha=0.3)
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, history['lr'])
            plt.yscale('log'); plt.title('LR'); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(img); plt.close()

            print(f"  Done {task['type']}={val} | Best Val: {best_v:.6f}")
            if not is_fixed:
                del data, edges
            gc.collect()


if __name__ == "__main__":
    D_ROOT = '/content/drive/MyDrive/real-data'
    N_ROOT = '/content/drive/MyDrive/real_networks'
    S_ROOT = '/content/drive/MyDrive/real-fixed-GTA'

    for net_name in ["Protein"]:
        print("\n" + "=" * 50)
        print(f"Processing network: {net_name}")
        print("=" * 50)
        tasks = [
            {"name": f"{net_name}_Var_Gap", "type": "gap", "values": [15]},
        ]
        run_gnn_experiment_suite(net_name, tasks, D_ROOT, N_ROOT, S_ROOT)




#         tasks = [
#     {"name": f"{name}_FixWindow", "type": "pair", "values": [
#         (30, 1), (15, 2), (10, 3), (6, 5), (5, 6), (3, 10), (2, 15)
#     ]},
# ]