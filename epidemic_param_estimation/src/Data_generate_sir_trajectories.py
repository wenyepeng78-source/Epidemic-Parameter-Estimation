"""
Generate SIR trajectory datasets on a static network for parameter estimation.

For each (p, q) pair, run simulations until 50 valid outbreaks are collected
(cumulative infections >= 5% of nodes). Results are saved as a compressed pickle.
"""

import os
import time
import gzip
import pickle
import random
from collections import Counter
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
import networkx as nx


# ==================== Configuration ====================
CSV_PATH = '/content/drive/MyDrive/real_networks/Nips.csv'
SAVE_PATH = '/content/drive/MyDrive/real-data/Nips_optimized.pkl.gz'

NUM_INIT_NODES = 1
REQUIRED_PER_PQ = 50
MAX_ATTEMPTS = 10000
BASE_SEED = 520
N_PROCESSES = 2

P_LIST = np.linspace(0.03, 0.30, 10)
Q_LIST = np.linspace(0.03, 0.30, 10)


# ==================== Load network ====================
def load_network(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Network file not found: {csv_path}")

    data = pd.read_csv(csv_path)
    G = nx.from_pandas_edgelist(data, source='source', target='target')
    all_nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    N = len(all_nodes)

    print(f"Network loaded: {N} nodes, {G.number_of_edges()} edges")
    return G, all_nodes, node_to_idx, N


# ==================== SIR simulation ====================
def simulation(G, all_nodes, node_to_idx, N, p, q, seed, num_init, min_infected):
    """
    Run one SIR trajectory for 100 steps.
    Returns ((p, q), history_matrix) or None if outbreak is too small.
    history_matrix: (101, N) int8, states 0=S, 1=I, 2=R.
    """
    np.random.seed(seed)
    random.seed(seed)

    init_nodes = np.random.choice(all_nodes, size=num_init, replace=False)
    I = set(init_nodes)
    R = set()
    S = set(all_nodes) - I
    cumulative_infected = len(I)

    history_matrix = np.zeros((101, N), dtype=np.int8)
    history_matrix[0, [node_to_idx[n] for n in I]] = 1

    for step in range(1, 101):
        # Infection: count exposures of susceptible nodes from infected neighbors
        threatened = []
        for inode in I:
            threatened.extend(nb for nb in G.neighbors(inode) if nb in S)
        exposure_counts = Counter(threatened)

        new_infect = set()
        for node, n_inf in exposure_counts.items():
            if np.random.random() < (1.0 - (1.0 - p) ** n_inf):
                new_infect.add(node)

        # Recovery
        new_recover = {node for node in I if np.random.random() < q}

        I -= new_recover
        I |= new_infect
        R |= new_recover
        S -= new_infect
        cumulative_infected += len(new_infect)

        if I:
            history_matrix[step, [node_to_idx[n] for n in I]] = 1
        if R:
            history_matrix[step, [node_to_idx[n] for n in R]] = 2

        # Early stop if no remaining infected nodes
        if len(I) == 0:
            history_matrix[step + 1:] = history_matrix[step]
            break

    if cumulative_infected < min_infected:
        return None
    return (p, q), history_matrix


# ==================== Worker for one (p, q) ====================
def run_one_pq(args):
    """Collect REQUIRED_PER_PQ valid trajectories for a single (p, q)."""
    p, q, G, all_nodes, node_to_idx, N, num_init, min_infected = args
    valid = []
    attempts = 0

    while len(valid) < REQUIRED_PER_PQ and attempts < MAX_ATTEMPTS:
        attempts += 1
        raw_seed = (
            BASE_SEED
            + attempts * 997
            + int(p * 100000) * 137
            + int(q * 100000) * 19
        )
        safe_seed = raw_seed % (2**32 - 1)

        result = simulation(
            G, all_nodes, node_to_idx, N, p, q, safe_seed, num_init, min_infected
        )
        if result is not None:
            valid.append(result)

    if len(valid) < REQUIRED_PER_PQ:
        print(
            f"Discarded p={p:.3f}, q={q:.3f} | "
            f"got {len(valid)}/{REQUIRED_PER_PQ} after {attempts} attempts"
        )
        return []

    return valid


# ==================== Main ====================
if __name__ == '__main__':
    total_start = time.time()

    G, all_nodes, node_to_idx, N = load_network(CSV_PATH)
    min_infected = max(1, int(N * 0.05))

    print("=" * 40)
    print(f"Initial infected nodes : {NUM_INIT_NODES}")
    print(f"Min cumulative infected: {min_infected} (5% of N)")
    print(f"Samples per (p, q)     : {REQUIRED_PER_PQ}")
    print("=" * 40)

    pq_pairs = list(product(P_LIST, Q_LIST))
    worker_args = [
        (p, q, G, all_nodes, node_to_idx, N, NUM_INIT_NODES, min_infected)
        for p, q in pq_pairs
    ]

    print(f"Planned parameter pairs: {len(pq_pairs)}")
    print("Running parallel simulations...")

    sim_start = time.time()
    with Pool(processes=N_PROCESSES) as pool:
        results = pool.map(run_one_pq, worker_args)
    sim_end = time.time()
    print(f"Simulations finished in {sim_end - sim_start:.2f} s")

    all_valid = [item for sublist in results for item in sublist]
    n_pairs_kept = len(all_valid) // REQUIRED_PER_PQ

    print("\n--- Summary ---")
    print(f"Planned pairs : {len(pq_pairs)}")
    print(f"Kept pairs    : {n_pairs_kept}")
    print(f"Discarded     : {len(pq_pairs) - n_pairs_kept}")
    print(f"Total trajectories: {len(all_valid)}")

    if len(all_valid) > 0:
        os.makedirs(os.path.dirname(SAVE_PATH) or '.', exist_ok=True)
        print(f"\nSaving {len(all_valid)} trajectories to {SAVE_PATH} ...")
        save_start = time.time()
        with gzip.open(SAVE_PATH, 'wb') as f:
            pickle.dump(
                {'node_mapping': node_to_idx, 'trajectories': all_valid},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"Saved in {time.time() - save_start:.2f} s")
    else:
        print("No valid trajectories generated; nothing saved.")

    total_elapsed = time.time() - total_start
    print(f"\nDone. Total time: {total_elapsed:.2f} s ({total_elapsed / 60:.2f} min)")