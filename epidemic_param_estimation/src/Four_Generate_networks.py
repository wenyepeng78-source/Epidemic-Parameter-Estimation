"""
Generate synthetic networks for epidemic parameter estimation experiments.

Four topologies: BA, RRN (random regular), INC (inclusivity), Torus (2D lattice).

Experiment grid (unique (N, k) pairs, no duplicates):
  - Fixed N=1000, k in {2, 4, 8, 16, 32, 64}
  - Fixed k=4,   N in {100, 300, 1000, 3000, 10000}

The overlapping case (N=1000, k=4) is generated only once.
"""

import os
import networkx as nx
import numpy as np
import pandas as pd


def save_network(G, net_type, N, k, output_dir):
    """Save edge list as CSV: {net_type}(N={N},K={k}).csv"""
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{net_type}(N={N},K={k}).csv"
    file_path = os.path.join(output_dir, file_name)

    edges_df = nx.to_pandas_edgelist(G)[['source', 'target']]
    edges_df.to_csv(file_path, index=False)

    actual_k = sum(dict(G.degree()).values()) / G.number_of_nodes()
    print(
        f"  Saved {file_name} | nodes={G.number_of_nodes()} | "
        f"edges={len(edges_df)} | mean_degree={actual_k:.3f}"
    )


def generate_inclusivity_network(N, k, r=1, rng=None):
    """Preferential attachment with local random walk (inclusivity model)."""
    if rng is None:
        rng = np.random.default_rng()

    m = k // 2
    m0 = 2 * m + 1
    G = nx.complete_graph(m0)

    for t in range(m0, N):
        G.add_node(t)
        nodes_before = list(G.nodes())
        nodes_before.remove(t)
        degrees = np.array([G.degree(node) for node in nodes_before])
        probs = degrees / degrees.sum()
        first_target = rng.choice(nodes_before, p=probs)
        G.add_edge(t, first_target)

        edges_needed = m - 1
        edges_added = 0
        attempts = 0
        while edges_added < edges_needed and attempts < 100:
            t_neighbors = list(G.neighbors(t))
            current = rng.choice(t_neighbors)
            for _ in range(r):
                neighs = list(G.neighbors(current))
                current = rng.choice(neighs)
            if current != t and not G.has_edge(t, current):
                G.add_edge(t, current)
                edges_added += 1
            attempts += 1

    return G


def generate_2d_torus_lattice(N, k, rng, D=2):
    """2D torus lattice with approximate mean degree k."""
    L = int(round(N ** (1 / D)))
    N_actual = L ** D
    coords = np.array(np.unravel_index(np.arange(N_actual), [L] * D))

    dist_mat_sq = np.zeros((N_actual, N_actual))
    for i in range(D):
        pos_i = coords[i, :].reshape(-1, 1)
        d_vec = np.abs(pos_i - pos_i.T)
        d_vec = np.minimum(d_vec, L - d_vec)
        dist_mat_sq += d_vec ** 2
    dist_mat = np.sqrt(dist_mat_sq)

    distances_from_first = dist_mat[0, :]

    def get_deg(radius):
        return np.sum((distances_from_first <= radius) & (distances_from_first > 0))

    r_upper = 0
    while get_deg(r_upper) <= k:
        r_upper += 1

    sigma_r = 0.01
    rr = np.arange(0, r_upper + sigma_r, sigma_r)
    kk_counts = np.array([get_deg(r_val) for r_val in rr])

    idx_k = np.where(kk_counts >= k)[0][0]
    r = rr[idx_k]
    rL = rr[idx_k - 1] if idx_k > 0 else 0

    A = (dist_mat <= r) & (dist_mat > 0)
    current_k = int(np.sum(A[0]))
    num_to_delete = current_k - k

    if num_to_delete > 0:
        most_dist_neighbours = np.where(
            (distances_from_first <= r) & (distances_from_first > rL)
        )[0]
        num_to_delete = num_to_delete // 2
        if num_to_delete > 0 and len(most_dist_neighbours) >= num_to_delete:
            neighb_to_delete_idx = rng.choice(
                most_dist_neighbours, num_to_delete, replace=False
            )
            offsets = coords[:, neighb_to_delete_idx] - coords[:, 0:1]
            for kk_del in range(num_to_delete):
                offset = offsets[:, kk_del]
                for ii in range(N_actual):
                    target_coords = (coords[:, ii] + offset) % L
                    jj = np.ravel_multi_index(target_coords.astype(int), [L] * D)
                    A[ii, jj] = False
                    A[jj, ii] = False

    return nx.from_numpy_array(A)


def generate_nets(N, k, seed=None):
    """Generate four network types with given N and target mean degree k."""
    rng = np.random.default_rng(seed)
    networks = {}

    # 1. Barabasi-Albert
    m = max(1, k // 2)
    networks['BA'] = nx.barabasi_albert_graph(N, m, seed=seed)

    # 2. Random regular (degree must be integer and N*k even)
    k_rrn = k if (N * k) % 2 == 0 else k - 1
    k_rrn = max(1, min(k_rrn, N - 1))
    if (N * k_rrn) % 2 != 0:
        k_rrn -= 1
    networks['RRN'] = nx.random_regular_graph(k_rrn, N, seed=seed)

    # 3. Inclusivity
    networks['INC'] = generate_inclusivity_network(N, k, r=1, rng=rng)

    # 4. 2D Torus
    networks['Torus'] = generate_2d_torus_lattice(N, k, rng=rng)

    return networks


def build_unique_configs():
    """
    Build unique (N, k) pairs from the two experimental axes:
      - Fixed N=1000, vary k
      - Fixed k=4,   vary N
    """
    fixed_N = 1000
    ks = [2, 4, 8, 16, 32, 64]
    fixed_k = 4
    Ns = [100, 300, 1000, 3000, 10000]

    configs = set()
    for k in ks:
        configs.add((fixed_N, k))
    for N in Ns:
        configs.add((N, fixed_k))

    # Sort for reproducible generation order
    return sorted(configs, key=lambda x: (x[0], x[1]))


if __name__ == "__main__":
    OUTPUT_DIRECTORY = "/content/drive/MyDrive/Generate_net"
    BASE_SEED = 520

    configs = build_unique_configs()
    print(f"Unique (N, k) configs to generate: {len(configs)}")
    for N, k in configs:
        print(f"  N={N}, k={k}")

    print(f"\nOutput directory: {OUTPUT_DIRECTORY}\n")

    for i, (N, k) in enumerate(configs):
        seed = BASE_SEED + i
        print(f"[{i + 1}/{len(configs)}] Generating nets for N={N}, k={k} (seed={seed})")
        nets_dict = generate_nets(N, k, seed=seed)

        for net_type, G in nets_dict.items():
            save_network(G, net_type, N, k, OUTPUT_DIRECTORY)

        print()

    print(f"Done. Files saved under: {os.path.abspath(OUTPUT_DIRECTORY)}")