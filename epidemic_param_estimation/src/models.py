"""
Model definitions, loss functions, MLE estimator, and utilities.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=520):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class MSLELoss(nn.Module):
    """Mean Squared Logarithmic Error summed over the two outputs (p, q)."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, actual):
        loss_matrix = self.mse(torch.log1p(pred), torch.log1p(actual))
        return torch.mean(torch.sum(loss_matrix, dim=1))


class EpidemicCNN(nn.Module):
    """1D-CNN for joint estimation of infection rate p and recovery rate q."""

    def __init__(self, K, input_channels=7):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(K, 16), nn.ReLU(),
            nn.Linear(16, 2), nn.Sigmoid()
        )

    def forward(self, x):
        features = self.conv_net(x).view(x.size(0), -1)
        return self.regressor(features)


class EpidemicCNN_MLE(nn.Module):
    """
    1D-CNN with late-fusion of an MLE estimate of q.
    The MLE value is concatenated to the CNN features before the final MLP.
    """

    def __init__(self, K, input_channels=7):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(K + 1, 32), nn.ReLU(),
            nn.Linear(32, 2), nn.Sigmoid()
        )

    def forward(self, x, q_mle):
        features = self.conv_net(x).view(x.size(0), -1)
        fused = torch.cat([features, q_mle.view(-1, 1)], dim=-1)
        return self.regressor(fused)


def estimate_q_mle(history_matrix, days, dt):
    """
    Maximum-likelihood estimate of recovery rate q from discrete observations.
    Counts I->R (A) and I->I (B) transitions between consecutive observation times.
    Returns (q_hat, A, B). q_hat is 0.0 when no informative transitions exist.
    """
    A, B = 0, 0
    for k in range(len(days) - 1):
        t_curr, t_next = days[k], days[k + 1]
        if t_curr >= history_matrix.shape[0] or t_next >= history_matrix.shape[0]:
            continue
        state_curr = history_matrix[t_curr]
        state_next = history_matrix[t_next]
        A += np.sum((state_curr == 1) & (state_next == 2))
        B += np.sum((state_curr == 1) & (state_next == 1))

    if A + B > 0:
        return 1.0 - (B / (A + B)) ** (1.0 / dt), A, B
    return 0.0, A, B


def get_mape(y_true, y_pred):
    """Mean Absolute Percentage Error with protection against zero and NaN."""
    y_true_safe = np.maximum(y_true, 1e-7)
    return np.nanmean(np.abs((y_true - y_pred) / y_true_safe))


# -------------------- GTA (GNN + Transformer) --------------------

import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

try:
    from torch_geometric.nn import GINConv, global_add_pool
except ImportError:
    GINConv = None
    global_add_pool = None


class GINModel(nn.Module):
    """Graph Isomorphism Network backbone used by GTA."""

    def __init__(self, in_dim=3, mid_dim=32, hidden_dim=64, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        mlp1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim), nn.BatchNorm1d(mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )
        self.convs.append(GINConv(mlp1, train_eps=True))
        for _ in range(num_layers - 1):
            mlp_next = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
            )
            self.convs.append(GINConv(mlp_next, train_eps=True))

    def forward(self, x, edge_index):
        hidden_states = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            hidden_states.append(x)
        return hidden_states


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, time_idx):
        phase = time_idx.unsqueeze(-1).float() * self.div_term
        pe = torch.zeros(
            time_idx.shape[0], time_idx.shape[1], self.d_model, device=time_idx.device
        )
        pe[:, :, 0::2] = torch.sin(phase)
        pe[:, :, 1::2] = torch.cos(phase)
        return pe







class GTA(nn.Module):
    """
    Graph-Temporal Attention model for joint estimation of (p, q).
    GIN over node states -> sequence of graph embeddings -> Transformer -> dual readout.
    """

    def __init__(
        self, num_times, node_in_dim=3, gnn_mid_dim=16, gnn_hidden_dim=32,
        gnn_num_layers=2, d_model=256, nhead=8, num_layers=1
    ):
        super().__init__()
        self.gnn = GINModel(
            in_dim=node_in_dim, mid_dim=gnn_mid_dim,
            hidden_dim=gnn_hidden_dim, num_layers=gnn_num_layers
        )
        self.gnn_out_dim = node_in_dim + (gnn_num_layers * gnn_hidden_dim)

        self.input_proj = nn.Linear(self.gnn_out_dim, d_model)
        self.time_embedding = SinusoidalTimeEmbedding(d_model=d_model)
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
            ),
            num_layers=num_layers
        )
        self.ln = nn.LayerNorm(d_model)

        self.proj_p = nn.Linear(d_model, d_model)
        self.proj_q = nn.Linear(d_model, d_model)
        self.scorer_p = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1)
        )
        self.readout_p = nn.Sequential(
            nn.Linear(d_model, 64), nn.LeakyReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.scorer_q = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1)
        )
        self.readout_q = nn.Sequential(
            nn.Linear(d_model, 64), nn.LeakyReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x_list, edge_index_list, time_idx):
        B, K, N = x_list[0].shape[0], len(x_list), x_list[0].shape[1]

        x_gnn_input = torch.stack(x_list, dim=0).view(K * B * N, -1)
        base_edge_index = edge_index_list[0]
        batch_offset = torch.arange(K * B, device=x_list[0].device) * N
        batched_edge_index = (
            base_edge_index.unsqueeze(0) + batch_offset.view(K * B, 1, 1)
        ).transpose(0, 1).reshape(2, -1)
        batch_index = torch.arange(K * B, device=x_list[0].device).repeat_interleave(N)

        pooled_feats = [
            global_add_pool(h, batch_index)
            for h in self.gnn(x_gnn_input, batched_edge_index)
        ]
        gnn_seq = torch.cat(pooled_feats, dim=-1).view(K, B, -1).transpose(0, 1)

        seq_out = self.transformer(
            self.ln(self.input_proj(gnn_seq) + self.time_embedding(time_idx))
        )

        weights_p = F.softmax(self.scorer_p(F.gelu(self.proj_p(seq_out))), dim=1)
        pred_p = self.readout_p(torch.sum(seq_out * weights_p, dim=1))

        weights_q = F.softmax(self.scorer_q(F.gelu(self.proj_q(seq_out))), dim=1)
        pred_q = self.readout_q(torch.sum(seq_out * weights_q, dim=1))

        return torch.cat([pred_p, pred_q], dim=1)


class GTA_MLE(nn.Module):
    """
    GTA with late-fusion of an MLE estimate of q.
    Predicts only infection rate p; q is taken from the MLE feature.
    """

    def __init__(
        self, num_times, node_in_dim=3, gnn_mid_dim=16, gnn_hidden_dim=32,
        gnn_num_layers=2, d_model=256, nhead=8, num_layers=1
    ):
        super().__init__()
        self.gnn = GINModel(
            in_dim=node_in_dim, mid_dim=gnn_mid_dim,
            hidden_dim=gnn_hidden_dim, num_layers=gnn_num_layers
        )
        self.gnn_out_dim = node_in_dim + (gnn_num_layers * gnn_hidden_dim)

        self.input_proj = nn.Linear(self.gnn_out_dim, d_model)
        self.time_embedding = SinusoidalTimeEmbedding(d_model=d_model)
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
            ),
            num_layers=num_layers
        )
        self.ln = nn.LayerNorm(d_model)

        self.proj_p = nn.Linear(d_model, d_model)
        self.scorer_p = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1)
        )
        self.readout_p = nn.Sequential(
            nn.Linear(d_model + 1, 64), nn.LeakyReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x_list, edge_index_list, time_idx, q_mle):
        B, K, N = x_list[0].shape[0], len(x_list), x_list[0].shape[1]

        x_gnn_input = torch.stack(x_list, dim=0).view(K * B * N, -1)
        base_edge_index = edge_index_list[0]
        batch_offset = torch.arange(K * B, device=x_list[0].device) * N
        batched_edge_index = (
            base_edge_index.unsqueeze(0) + batch_offset.view(K * B, 1, 1)
        ).transpose(0, 1).reshape(2, -1)
        batch_index = torch.arange(K * B, device=x_list[0].device).repeat_interleave(N)

        pooled_feats = [
            global_add_pool(h, batch_index)
            for h in self.gnn(x_gnn_input, batched_edge_index)
        ]
        gnn_seq = torch.cat(pooled_feats, dim=-1).view(K, B, -1).transpose(0, 1)

        seq_out = self.transformer(
            self.ln(self.input_proj(gnn_seq) + self.time_embedding(time_idx))
        )

        weights_p = F.softmax(self.scorer_p(F.gelu(self.proj_p(seq_out))), dim=1)
        spatio_temporal_feat = torch.sum(seq_out * weights_p, dim=1)
        fused_feat = torch.cat([spatio_temporal_feat, q_mle.view(-1, 1)], dim=-1)
        pred_p = self.readout_p(fused_feat)
        return pred_p


class MSLELossSingle(nn.Module):
    """MSLE for single-output (p only) used by GTA_MLE."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(torch.log1p(pred), torch.log1p(actual))