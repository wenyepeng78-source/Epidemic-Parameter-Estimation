"""
Data loading and macroscopic feature extraction for CNN-based models.
"""

import os
import pickle
import gzip
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .models import estimate_q_mle


def load_network_and_data(net_name, DATA_ROOT, NETWORK_ROOT, N=None, K_avg=None):
    """
    Load trajectory data and static network edges.
    Supports both real networks and synthetic networks named with (N=...,K=...).
    Returns:
        data_chunk  : list of ((p, q), history_matrix)
        edge_index  : LongTensor of shape (2, E)
        n_nodes     : int
    """
    if N is not None and K_avg is not None:
        data_path = os.path.join(DATA_ROOT, f"{net_name}(N={N},K={K_avg})_optimized.pkl.gz")
        csv_path = os.path.join(NETWORK_ROOT, f"{net_name}(N={N},K={K_avg}).csv")
    else:
        data_path = os.path.join(DATA_ROOT, f"{net_name}_optimized.pkl.gz")
        csv_path = os.path.join(NETWORK_ROOT, f"{net_name}.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Trajectory data not found: {data_path}")

    with gzip.open(data_path, 'rb') as f:
        saved_data = pickle.load(f)

    data_chunk = saved_data['trajectories']
    node_mapping = saved_data['node_mapping']

    static_df = pd.read_csv(csv_path)
    s_col = 'Source' if 'Source' in static_df.columns else 'source'
    t_col = 'Target' if 'Target' in static_df.columns else 'target'

    sources = [node_mapping[n] for n in static_df[s_col]]
    targets = [node_mapping[n] for n in static_df[t_col]]
    edge_index = torch.tensor([sources, targets], dtype=torch.long)

    return data_chunk, edge_index, len(node_mapping)


def _compute_degrees(edge_index, n_nodes):
    """Return degree vector and its maximum (for normalisation)."""
    n_edges = edge_index.shape[1]
    degrees = torch.zeros(n_nodes)
    ones = torch.ones(n_edges)
    degrees.scatter_add_(0, edge_index[0], ones)
    degrees.scatter_add_(0, edge_index[1], ones)
    max_deg = degrees.max().item() + 1e-6
    return degrees, max_deg, n_edges


def extract_cnn_features(data_chunk, edge_index, n_nodes, days):
    """
    Extract macroscopic features for the plain CNN.
    Returns:
        X : FloatTensor (B, C=7, T)
        Y : FloatTensor (B, 2)  [p, q]
    """
    degrees, max_deg, n_edges = _compute_degrees(edge_index, n_nodes)
    t_first = days[0] if days else 0

    all_x, all_y = [], []
    for (p, q), history_matrix in tqdm(data_chunk, desc="Extracting Features", leave=False):
        time_step_features = []
        for t in days:
            status_t = torch.from_numpy(history_matrix[min(t, 100)]).long()
            s_mask = (status_t == 0)
            i_mask = (status_t == 1)
            u_st = status_t[edge_index[0]]
            v_st = status_t[edge_index[1]]
            si_count = ((u_st == 1) & (v_st == 0) | (u_st == 0) & (v_st == 1)).sum().item()

            f_t = torch.tensor([
                s_mask.sum().float() / n_nodes,
                i_mask.sum().float() / n_nodes,
                (status_t == 2).sum().float() / n_nodes,
                si_count / n_edges,
                degrees[s_mask].mean() / max_deg if s_mask.any() else 0.0,
                degrees[i_mask].mean() / max_deg if i_mask.any() else 0.0,
                (t - t_first) / 100.0
            ])
            time_step_features.append(f_t)

        all_x.append(torch.stack(time_step_features).float())
        all_y.append(torch.tensor([p, q], dtype=torch.float32))

    X = torch.stack(all_x).float().transpose(1, 2)  # (B, C, T)
    Y = torch.stack(all_y).float()
    return X, Y


def extract_cnn_features_mle(data_chunk, edge_index, n_nodes, days):
    """
    Extract macroscopic features plus a late-fusion MLE of q.
    Returns:
        X     : FloatTensor (B, C=7, T)
        Q_mle : FloatTensor (B, 1)
        Y     : FloatTensor (B, 2)  [p, q]
    """
    degrees, max_deg, n_edges = _compute_degrees(edge_index, n_nodes)
    t_first = days[0] if days else 0
    dt = days[1] - days[0] if len(days) > 1 else 1

    all_x, all_q_mle, all_y = [], [], []
    for (p, q), history_matrix in tqdm(data_chunk, desc="Extracting Features", leave=False):
        q_hat, _, _ = estimate_q_mle(history_matrix, days, dt)
        all_q_mle.append(torch.tensor([q_hat], dtype=torch.float32))

        time_step_features = []
        for t in days:
            status_t = torch.from_numpy(history_matrix[min(t, 100)]).long()
            s_mask = (status_t == 0)
            i_mask = (status_t == 1)
            u_st = status_t[edge_index[0]]
            v_st = status_t[edge_index[1]]
            si_count = ((u_st == 1) & (v_st == 0) | (u_st == 0) & (v_st == 1)).sum().item()

            f_t = torch.tensor([
                s_mask.sum().float() / n_nodes,
                i_mask.sum().float() / n_nodes,
                (status_t == 2).sum().float() / n_nodes,
                si_count / n_edges,
                degrees[s_mask].mean() / max_deg if s_mask.any() else 0.0,
                degrees[i_mask].mean() / max_deg if i_mask.any() else 0.0,
                (t - t_first) / 100.0
            ])
            time_step_features.append(f_t)

        # stack then transpose -> (C, T); final stack yields (B, C, T)
        all_x.append(torch.stack(time_step_features).t().float())
        all_y.append(torch.tensor([p, q], dtype=torch.float32))

    return (
        torch.stack(all_x).float(),
        torch.stack(all_q_mle).float(),
        torch.stack(all_y).float()
    )


# -------------------- XGBoost feature extraction --------------------

def load_experiment_data_np(net_name, DATA_ROOT, NETWORK_ROOT, N=None, K=None):
    """
    Load trajectory data and edges as NumPy arrays (for XGBoost).
    Same interface as load_network_and_data, but edge_index is np.ndarray.
    """
    if N is not None and K is not None:
        data_path = os.path.join(DATA_ROOT, f"{net_name}(N={N},K={K})_optimized.pkl.gz")
        csv_path = os.path.join(NETWORK_ROOT, f"{net_name}(N={N},K={K}).csv")
    else:
        data_path = os.path.join(DATA_ROOT, f"{net_name}_optimized.pkl.gz")
        csv_path = os.path.join(NETWORK_ROOT, f"{net_name}.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Trajectory data not found: {data_path}")

    print(f"[Loading] {os.path.basename(data_path)}")
    with gzip.open(data_path, 'rb') as f:
        saved_data = pickle.load(f)

    data_chunk = saved_data['trajectories']
    node_mapping = saved_data['node_mapping']

    static_df = pd.read_csv(csv_path)
    s_col = 'Source' if 'Source' in static_df.columns else 'source'
    t_col = 'Target' if 'Target' in static_df.columns else 'target'

    sources = [node_mapping[n] for n in static_df[s_col]]
    targets = [node_mapping[n] for n in static_df[t_col]]
    edge_index = np.array([sources, targets], dtype=np.int64)

    return data_chunk, edge_index, len(node_mapping)


def extract_xgb_features(data_chunk, indices, edge_index, n_nodes, days, desc="Features"):
    """
    Macroscopic features for plain multi-output XGBoost.
    Returns X (N, T*7), Y (N, 2) with [p, q].
    """
    n_edges = edge_index.shape[1]
    deg_out = np.bincount(edge_index[0], minlength=n_nodes)
    deg_in = np.bincount(edge_index[1], minlength=n_nodes)
    degrees = deg_out + deg_in
    max_deg = degrees.max() + 1e-6
    t_first = days[0] if len(days) > 0 else 0

    X_list, y_list = [], []
    for idx in tqdm(indices, desc=desc, leave=False):
        (p, q), history_matrix = data_chunk[idx]
        sample_features = []
        for t in days:
            t_idx = min(t, 100)
            status_t = history_matrix[t_idx]
            s_mask = (status_t == 0)
            i_mask = (status_t == 1)
            r_mask = (status_t == 2)

            u_st = status_t[edge_index[0]]
            v_st = status_t[edge_index[1]]
            si_count = np.sum(((u_st == 1) & (v_st == 0)) | ((u_st == 0) & (v_st == 1)))

            feat = [
                s_mask.sum() / n_nodes,
                i_mask.sum() / n_nodes,
                r_mask.sum() / n_nodes,
                si_count / n_edges,
                (degrees[s_mask].mean() if s_mask.any() else 0.0) / max_deg,
                (degrees[i_mask].mean() if i_mask.any() else 0.0) / max_deg,
                (t - t_first) / 100.0
            ]
            sample_features.extend(feat)

        X_list.append(sample_features)
        y_list.append([p, q])

    return np.array(X_list), np.array(y_list)


def extract_xgb_features_mle(data_chunk, indices, edge_index, n_nodes, days, desc="Features", return_pq=False):
    """
    Macroscopic features + late-fusion MLE of q.
    If return_pq=False (training): Y is p only.
    If return_pq=True  (evaluation): Y is [p, q].
    """
    n_edges = edge_index.shape[1]
    deg_out = np.bincount(edge_index[0], minlength=n_nodes)
    deg_in = np.bincount(edge_index[1], minlength=n_nodes)
    degrees = deg_out + deg_in
    max_deg = degrees.max() + 1e-6
    t_first = days[0] if len(days) > 0 else 0
    dt = days[1] - days[0] if len(days) > 1 else 1

    X_list, y_list = [], []
    for idx in tqdm(indices, desc=desc, leave=False):
        (p, q), history_matrix = data_chunk[idx]

        q_hat, _, _ = estimate_q_mle(history_matrix, days, dt)
        q_mle_val = q_hat if q_hat is not None else np.nan

        sample_features = []
        for t in days:
            t_idx = min(t, history_matrix.shape[0] - 1)
            status_t = history_matrix[t_idx]
            s_mask = (status_t == 0)
            i_mask = (status_t == 1)
            r_mask = (status_t == 2)

            u_st = status_t[edge_index[0]]
            v_st = status_t[edge_index[1]]
            si_count = np.sum(((u_st == 1) & (v_st == 0)) | ((u_st == 0) & (v_st == 1)))

            feat = [
                s_mask.sum() / n_nodes,
                i_mask.sum() / n_nodes,
                r_mask.sum() / n_nodes,
                si_count / n_edges,
                (degrees[s_mask].mean() if s_mask.any() else 0.0) / max_deg,
                (degrees[i_mask].mean() if i_mask.any() else 0.0) / max_deg,
                (t - t_first) / 100.0,
                q_mle_val
            ]
            sample_features.extend(feat)

        X_list.append(sample_features)
        y_list.append([p, q] if return_pq else p)

    return np.array(X_list), np.array(y_list)






# -------------------- GTA data utilities --------------------

import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from torch_geometric.utils import to_undirected
except ImportError:
    to_undirected = None

from .models import estimate_q_mle


def load_gta_network_data(net_name, DATA_ROOT, NETWORK_ROOT, N=None, K_avg=None):
    """
    Load trajectories and undirected edge_index for GTA.
    """
    path_synthetic = os.path.join(
        DATA_ROOT, f"{net_name}(N={N},K={K_avg})_optimized.pkl.gz"
    )
    path_real = os.path.join(DATA_ROOT, f"{net_name}_optimized.pkl.gz")

    if N is not None and K_avg is not None and os.path.exists(path_synthetic):
        data_path = path_synthetic
        csv_path = os.path.join(NETWORK_ROOT, f"{net_name}(N={N},K={K_avg}).csv")
    elif os.path.exists(path_real):
        data_path = path_real
        csv_path = os.path.join(NETWORK_ROOT, f"{net_name}.csv")
    else:
        raise FileNotFoundError(f"Dataset for '{net_name}' not found in {DATA_ROOT}")

    print(f"Loading data: {os.path.basename(data_path)}")
    with gzip.open(data_path, 'rb') as f:
        saved_data = pickle.load(f)

    data_chunk = saved_data['trajectories']
    node_mapping = saved_data['node_mapping']

    static_df = pd.read_csv(csv_path)
    s_col = 'Source' if 'Source' in static_df.columns else 'source'
    t_col = 'Target' if 'Target' in static_df.columns else 'target'

    sources = [node_mapping[n] for n in static_df[s_col]]
    targets = [node_mapping[n] for n in static_df[t_col]]
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    if to_undirected is not None:
        edge_index = to_undirected(edge_index)

    return data_chunk, edge_index, len(node_mapping)


class StaticSIRDataset(Dataset):
    """Plain GTA dataset: returns (x_list, edge_list, time_idx, [p, q])."""

    def __init__(self, data_chunk, edge_index_static, days_to_extract, n_nodes):
        self.data = data_chunk
        self.edge_index_static = edge_index_static
        self.n_nodes = n_nodes
        self.days = sorted(days_to_extract)
        self.K = len(self.days)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (p, q), history_matrix = self.data[idx]
        theta_true = torch.tensor([p, q], dtype=torch.float32)
        start_day = self.days[0]
        relative_days = [t - start_day for t in self.days]

        x_list = []
        for t in self.days:
            t_safe = min(t, 100)
            status_tensor = torch.from_numpy(history_matrix[t_safe]).long()
            x_t = F.one_hot(status_tensor, num_classes=3).float()
            x_list.append(x_t)

        edge_index_list = [self.edge_index_static for _ in range(self.K)]
        time_idx = torch.tensor(relative_days, dtype=torch.long)
        return x_list, edge_index_list, time_idx, theta_true


def gta_collate_fn(batch):
    B, K = len(batch), len(batch[0][0])
    x_list_batched, edge_index_list_batched = [], []
    time_idx = torch.stack([item[2] for item in batch])
    theta_true_batched = torch.stack([item[3] for item in batch])

    for k in range(K):
        x_k = torch.stack([item[0][k] for item in batch])
        x_list_batched.append(x_k)
        edge_index_list_batched.append(batch[0][1][0])

    return x_list_batched, edge_index_list_batched, time_idx, theta_true_batched


class StaticSIRDataset_MLE(Dataset):
    """GTA + MLE training dataset: target is p only; also returns q_mle."""

    def __init__(self, data_chunk, edge_index_static, days_to_extract, n_nodes):
        self.data = data_chunk
        self.edge_index_static = edge_index_static
        self.n_nodes = n_nodes
        self.days = sorted(days_to_extract)
        self.K = len(self.days)
        self.dt = self.days[1] - self.days[0] if len(self.days) > 1 else 1

        self.q_mle_list = []
        for (_, _), history_matrix in self.data:
            q_hat, _, _ = estimate_q_mle(history_matrix, self.days, self.dt)
            self.q_mle_list.append(q_hat if q_hat is not None else 0.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (p, _), history_matrix = self.data[idx]
        p_true = torch.tensor([p], dtype=torch.float32)
        q_mle_val = torch.tensor([self.q_mle_list[idx]], dtype=torch.float32)
        start_day = self.days[0]
        relative_days = [t - start_day for t in self.days]

        x_list = []
        for t in self.days:
            t_safe = min(t, 100)
            status_tensor = torch.from_numpy(history_matrix[t_safe]).long()
            x_t = F.one_hot(status_tensor, num_classes=3).float()
            x_list.append(x_t)

        edge_index_list = [self.edge_index_static for _ in range(self.K)]
        time_idx = torch.tensor(relative_days, dtype=torch.long)
        return x_list, edge_index_list, time_idx, p_true, q_mle_val


def gta_mle_collate_fn(batch):
    B, K = len(batch), len(batch[0][0])
    x_list_batched, edge_index_list_batched = [], []
    time_idx = torch.stack([item[2] for item in batch])
    p_true_batched = torch.stack([item[3] for item in batch])
    q_mle_batched = torch.stack([item[4] for item in batch])

    for k in range(K):
        x_k = torch.stack([item[0][k] for item in batch])
        x_list_batched.append(x_k)
        edge_index_list_batched.append(batch[0][1][0])

    return x_list_batched, edge_index_list_batched, time_idx, p_true_batched, q_mle_batched


class EvalDataset_MLE(Dataset):
    """GTA + MLE evaluation: returns true [p, q] and q_mle for metrics."""

    def __init__(self, data_chunk, edge_index_static, days_to_extract, n_nodes):
        self.data = data_chunk
        self.edge_index_static = edge_index_static
        self.n_nodes = n_nodes
        self.days = sorted(days_to_extract)
        self.K = len(self.days)
        self.dt = self.days[1] - self.days[0] if len(self.days) > 1 else 1

        self.q_mle_list = []
        for (_, _), history_matrix in self.data:
            q_hat, _, _ = estimate_q_mle(history_matrix, self.days, self.dt)
            self.q_mle_list.append(q_hat if q_hat is not None else 0.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (p, q), history_matrix = self.data[idx]
        theta_true = torch.tensor([p, q], dtype=torch.float32)
        q_mle_val = torch.tensor([self.q_mle_list[idx]], dtype=torch.float32)
        start_day = self.days[0]
        relative_days = [t - start_day for t in self.days]

        x_list = []
        for t in self.days:
            t_safe = min(t, 100)
            status_tensor = torch.from_numpy(history_matrix[t_safe]).long()
            x_t = F.one_hot(status_tensor, num_classes=3).float()
            x_list.append(x_t)

        edge_index_list = [self.edge_index_static for _ in range(self.K)]
        time_idx = torch.tensor(relative_days, dtype=torch.long)
        return x_list, edge_index_list, time_idx, theta_true, q_mle_val


def eval_mle_collate_fn(batch):
    B, K = len(batch), len(batch[0][0])
    x_list_batched, edge_index_list_batched = [], []
    time_idx = torch.stack([item[2] for item in batch])
    theta_true_batched = torch.stack([item[3] for item in batch])
    q_mle_batched = torch.stack([item[4] for item in batch])

    for k in range(K):
        x_k = torch.stack([item[0][k] for item in batch])
        x_list_batched.append(x_k)
        edge_index_list_batched.append(batch[0][1][0])

    return x_list_batched, edge_index_list_batched, time_idx, theta_true_batched, q_mle_batched



# -------------------- Missing-edge robustness utilities --------------------

def drop_edges_fixed_count(edge_index, drop_count, seed=42):
    """
    Randomly remove a fixed number of edges (directed).
    edge_index: LongTensor (2, E) or np.ndarray (2, E).
    """
    if drop_count <= 0:
        return edge_index

    is_torch = torch.is_tensor(edge_index)
    num_edges = edge_index.shape[1]
    num_to_keep = max(0, num_edges - drop_count)

    if is_torch:
        torch.manual_seed(seed)
        perm = torch.randperm(num_edges)
        return edge_index[:, perm[:num_to_keep]]
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(num_edges)
        return edge_index[:, perm[:num_to_keep]]


def extract_xgb_features_robust(data_chunk, indices, edge_index, n_nodes, days, desc="XGB Features"):
    """
    Same 7-D macroscopic features as extract_xgb_features, but safe when
    the graph has zero edges (missing-edge robustness).
    edge_index: np.ndarray or torch.Tensor (2, E).
    """
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()

    n_edges = max(int(edge_index.shape[1]), 0)
    if n_edges > 0:
        deg_out = np.bincount(edge_index[0], minlength=n_nodes)
        deg_in = np.bincount(edge_index[1], minlength=n_nodes)
        degrees = deg_out + deg_in
    else:
        degrees = np.zeros(n_nodes)
    max_deg = degrees.max() + 1e-6
    t_first = days[0] if len(days) > 0 else 0
    n_edges_safe = max(n_edges, 1e-7)

    X_list, y_list = [], []
    for idx in tqdm(indices, desc=desc, leave=False):
        (p, q), history_matrix = data_chunk[idx]
        sample_features = []
        for t in days:
            status_t = history_matrix[min(t, 100)]
            s_mask = (status_t == 0)
            i_mask = (status_t == 1)
            r_mask = (status_t == 2)

            if n_edges > 0:
                u_st = status_t[edge_index[0]]
                v_st = status_t[edge_index[1]]
                si_count = np.sum(
                    ((u_st == 1) & (v_st == 0)) | ((u_st == 0) & (v_st == 1))
                )
            else:
                si_count = 0

            feat = [
                s_mask.sum() / n_nodes,
                i_mask.sum() / n_nodes,
                r_mask.sum() / n_nodes,
                si_count / n_edges_safe,
                (degrees[s_mask].mean() if s_mask.any() else 0.0) / max_deg,
                (degrees[i_mask].mean() if i_mask.any() else 0.0) / max_deg,
                (t - t_first) / 100.0,
            ]
            sample_features.extend(feat)

        X_list.append(sample_features)
        y_list.append([p, q])

    return np.array(X_list), np.array(y_list)


def extract_cnn_features_robust(data_chunk, edge_index, n_nodes, days):
    """
    Same macroscopic CNN features as extract_cnn_features, safe for empty graphs.
    Returns X (B, C=7, T), Y (B, 2).
    """
    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)

    n_edges = edge_index.shape[1]
    degrees = torch.zeros(n_nodes)
    if n_edges > 0:
        ones = torch.ones(n_edges)
        degrees.scatter_add_(0, edge_index[0], ones)
        degrees.scatter_add_(0, edge_index[1], ones)
    max_deg = degrees.max().item() + 1e-6
    n_edges_safe = max(n_edges, 1e-7)
    t_first = days[0] if days else 0

    all_x, all_y = [], []
    for (p, q), history_matrix in tqdm(data_chunk, desc="CNN Features", leave=False):
        time_step_features = []
        for t in days:
            status_t = torch.from_numpy(history_matrix[min(t, 100)]).long()
            s_mask = (status_t == 0)
            i_mask = (status_t == 1)

            if n_edges > 0:
                u_st = status_t[edge_index[0]]
                v_st = status_t[edge_index[1]]
                si_count = (
                    ((u_st == 1) & (v_st == 0)) | ((u_st == 0) & (v_st == 1))
                ).sum().item()
            else:
                si_count = 0

            f_t = torch.tensor([
                s_mask.sum().float() / n_nodes,
                i_mask.sum().float() / n_nodes,
                (status_t == 2).sum().float() / n_nodes,
                si_count / n_edges_safe,
                degrees[s_mask].mean() / max_deg if s_mask.any() else 0.0,
                degrees[i_mask].mean() / max_deg if i_mask.any() else 0.0,
                (t - t_first) / 100.0,
            ])
            time_step_features.append(f_t)

        # (T, 7) -> transpose path consistent with extract_cnn_features: (B, C, T)
        all_x.append(torch.stack(time_step_features).float())
        all_y.append(torch.tensor([p, q], dtype=torch.float32))

    X = torch.stack(all_x).float().transpose(1, 2)
    Y = torch.stack(all_y).float()
    return X, Y



# -------------------- Node-state missing robustness --------------------

def update_hidden_mask(current_mask, drop_count, seed=42):
    """
    Hide additional nodes by setting True in a boolean mask.
    current_mask: BoolTensor of shape (N,). True = unobserved.
    """
    if drop_count <= 0:
        return current_mask.clone()

    new_mask = current_mask.clone()
    available = torch.nonzero(~new_mask, as_tuple=False).squeeze(-1)
    if available.numel() == 0:
        return new_mask
    if available.dim() == 0:
        available = available.unsqueeze(0)

    num_to_hide = min(drop_count, available.numel())
    torch.manual_seed(seed)
    perm = torch.randperm(available.numel())
    hide_idx = available[perm[:num_to_hide]]
    new_mask[hide_idx] = True
    return new_mask


def extract_xgb_features_nodes(
    data_chunk, indices, edge_index, n_nodes, days, hidden_mask, desc="XGB Features"
):
    """
    8-D macroscopic features per time step (includes unknown fraction).
    Hidden nodes are marked as state 3.
    """
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()
    n_edges = max(int(edge_index.shape[1]), 0)
    n_edges_safe = max(n_edges, 1e-7)

    if n_edges > 0:
        degrees = (
            np.bincount(edge_index[0], minlength=n_nodes)
            + np.bincount(edge_index[1], minlength=n_nodes)
        )
    else:
        degrees = np.zeros(n_nodes)
    max_deg = degrees.max() + 1e-6
    t_first = days[0] if days else 0
    hidden_np = (
        hidden_mask.cpu().numpy() if torch.is_tensor(hidden_mask) else hidden_mask
    )

    X_list, y_list = [], []
    for idx in tqdm(indices, desc=desc, leave=False):
        (p, q), history_matrix = data_chunk[idx]
        sample = []
        for t in days:
            status_t = history_matrix[min(t, 100)].copy()
            status_t[hidden_np] = 3

            s_mask = status_t == 0
            i_mask = status_t == 1
            r_mask = status_t == 2
            u_mask = status_t == 3

            if n_edges > 0:
                u_st, v_st = status_t[edge_index[0]], status_t[edge_index[1]]
                si = np.sum(((u_st == 1) & (v_st == 0)) | ((u_st == 0) & (v_st == 1)))
            else:
                si = 0

            sample.extend([
                s_mask.sum() / n_nodes,
                i_mask.sum() / n_nodes,
                r_mask.sum() / n_nodes,
                u_mask.sum() / n_nodes,
                si / n_edges_safe,
                (degrees[s_mask].mean() if s_mask.any() else 0.0) / max_deg,
                (degrees[i_mask].mean() if i_mask.any() else 0.0) / max_deg,
                (t - t_first) / 100.0,
            ])
        X_list.append(sample)
        y_list.append([p, q])

    return np.array(X_list), np.array(y_list)


def extract_cnn_features_nodes(data_chunk, edge_index, n_nodes, days, hidden_mask):
    """
    8-channel macroscopic features for CNN under node-state missing.
    Returns X (B, C=8, T), Y (B, 2).
    """
    if not torch.is_tensor(edge_index):
        edge_index = torch.tensor(edge_index, dtype=torch.long)

    n_edges = edge_index.shape[1]
    n_edges_safe = max(n_edges, 1e-7)
    degrees = torch.zeros(n_nodes)
    if n_edges > 0:
        ones = torch.ones(n_edges)
        degrees.scatter_add_(0, edge_index[0], ones)
        degrees.scatter_add_(0, edge_index[1], ones)
    max_deg = degrees.max().item() + 1e-6
    t_first = days[0] if days else 0

    all_x, all_y = [], []
    for (p, q), history_matrix in tqdm(data_chunk, desc="CNN Features", leave=False):
        steps = []
        for t in days:
            status_t = torch.from_numpy(history_matrix[min(t, 100)]).long()
            status_t = status_t.clone()
            status_t[hidden_mask] = 3

            s_mask = status_t == 0
            i_mask = status_t == 1
            r_mask = status_t == 2
            u_mask = status_t == 3

            if n_edges > 0:
                u_st, v_st = status_t[edge_index[0]], status_t[edge_index[1]]
                si = (
                    ((u_st == 1) & (v_st == 0)) | ((u_st == 0) & (v_st == 1))
                ).sum().item()
            else:
                si = 0

            steps.append(torch.tensor([
                s_mask.sum().float() / n_nodes,
                i_mask.sum().float() / n_nodes,
                r_mask.sum().float() / n_nodes,
                u_mask.sum().float() / n_nodes,
                si / n_edges_safe,
                degrees[s_mask].mean() / max_deg if s_mask.any() else 0.0,
                degrees[i_mask].mean() / max_deg if i_mask.any() else 0.0,
                (t - t_first) / 100.0,
            ]))
        all_x.append(torch.stack(steps).float())
        all_y.append(torch.tensor([p, q], dtype=torch.float32))

    X = torch.stack(all_x).float().transpose(1, 2)  # (B, 8, T)
    Y = torch.stack(all_y).float()
    return X, Y


class StaticSIRDataset_Nodes(Dataset):
    """
    GTA dataset with unobserved nodes marked as a 4th class (S,I,R,U).
    """

    def __init__(self, data_chunk, edge_index_static, days_to_extract, n_nodes, hidden_mask):
        self.data = data_chunk
        self.edge_index_static = edge_index_static
        self.n_nodes = n_nodes
        self.days = sorted(days_to_extract)
        self.K = len(self.days)
        self.hidden_mask = (
            hidden_mask if torch.is_tensor(hidden_mask)
            else torch.tensor(hidden_mask, dtype=torch.bool)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (p, q), history_matrix = self.data[idx]
        theta_true = torch.tensor([p, q], dtype=torch.float32)
        start_day = self.days[0]
        relative_days = [t - start_day for t in self.days]

        x_list = []
        hidden_np = self.hidden_mask.numpy()
        for t in self.days:
            status = history_matrix[min(t, 100)].copy()
            status[hidden_np] = 3
            status_t = torch.from_numpy(status).long()
            x_t = F.one_hot(status_t, num_classes=4).float()
            x_list.append(x_t)

        edge_index_list = [self.edge_index_static for _ in range(self.K)]
        time_idx = torch.tensor(relative_days, dtype=torch.long)
        return x_list, edge_index_list, time_idx, theta_true





    # -------------------- Only-I observation (time-window robustness) --------------------

def extract_features_only_i(data_chunk, indices, edge_index, n_nodes, days, desc="Features"):
    """
    Only infected nodes are observed; S and R are treated as unknown (U).

    Per time step (4-D):
      [I fraction, U fraction, mean degree of I (norm), relative time]

    Returns:
        X_xgb : np.ndarray (N, T*4)
        X_cnn : FloatTensor (B, C=4, T)
        Y     : FloatTensor (B, 2)
    """
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()

    n_edges = int(edge_index.shape[1])
    if n_edges > 0:
        degrees = (
            np.bincount(edge_index[0], minlength=n_nodes)
            + np.bincount(edge_index[1], minlength=n_nodes)
        )
    else:
        degrees = np.zeros(n_nodes)
    max_deg = degrees.max() + 1e-6
    t_first = days[0] if days else 0

    X_xgb_list, y_list, cnn_x_list = [], [], []
    for idx in tqdm(indices, desc=desc, leave=False):
        (p, q), history_matrix = data_chunk[idx]
        xgb_feats, cnn_steps = [], []
        for t in days:
            status_t = history_matrix[min(t, 100)]
            i_mask = (status_t == 1)
            u_mask = (status_t != 1)

            feat = [
                i_mask.sum() / n_nodes,
                u_mask.sum() / n_nodes,
                (degrees[i_mask].mean() if i_mask.any() else 0.0) / max_deg,
                (t - t_first) / 100.0,
            ]
            xgb_feats.extend(feat)
            cnn_steps.append(torch.tensor(feat, dtype=torch.float32))

        X_xgb_list.append(xgb_feats)
        cnn_x_list.append(torch.stack(cnn_steps).float())  # (T, 4)
        y_list.append([p, q])

    X_xgb = np.array(X_xgb_list)
    X_cnn = torch.stack(cnn_x_list).float().transpose(1, 2)  # (B, 4, T)
    Y = torch.tensor(y_list, dtype=torch.float32)
    return X_xgb, X_cnn, Y


class StaticSIRDataset_OnlyI(Dataset):
    """
    GTA dataset under Only-I observation.
    Node features: 2-D one-hot [I, U] where U = S or R.
    """

    def __init__(self, data_chunk, edge_index_static, days_to_extract, n_nodes):
        self.data = data_chunk
        self.edge_index_static = edge_index_static
        self.n_nodes = n_nodes
        self.days = sorted(days_to_extract)
        self.K = len(self.days)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (p, q), history_matrix = self.data[idx]
        theta_true = torch.tensor([p, q], dtype=torch.float32)
        start_day = self.days[0]
        relative_days = [t - start_day for t in self.days]

        x_list = []
        for t in self.days:
            status_t = history_matrix[min(t, 100)]
            i_mask = (status_t == 1)
            u_mask = (status_t != 1)
            x_t = torch.tensor(
                np.stack([i_mask, u_mask], axis=1), dtype=torch.float32
            )
            x_list.append(x_t)

        edge_index_list = [self.edge_index_static for _ in range(self.K)]
        time_idx = torch.tensor(relative_days, dtype=torch.long)
        return x_list, edge_index_list, time_idx, theta_true