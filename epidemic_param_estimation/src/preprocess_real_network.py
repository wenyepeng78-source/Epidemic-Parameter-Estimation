"""
Preprocess a real-world edge list into a clean undirected simple graph.

Steps:
  1. Load CSV and normalize column names to source / target.
  2. Remove self-loops.
  3. Remap node IDs to contiguous integers 0..N-1.
  4. Orient each edge so source < target (undirected canonical form).
  5. Drop duplicate edges.
  6. Save the cleaned edge list.
"""
#protein、email、egos、power and brain

import os
import numpy as np
import pandas as pd


def preprocess_real_network(
    input_path,
    output_path=None,
    source_col=None,
    target_col=None,
):
    """
    Clean a real network edge list.

    Parameters
    ----------
    input_path : str
        Path to the raw edge-list CSV.
    output_path : str or None
        Where to write the cleaned CSV. Defaults to overwriting input_path.
    source_col, target_col : str or None
        Column names if not auto-detected (Source/source, Target/target).

    Returns
    -------
    df : pd.DataFrame
        Cleaned edge list with columns ['source', 'target'].
    n_nodes : int
        Number of unique nodes after remapping.
    """
    if output_path is None:
        output_path = input_path

    df = pd.read_csv(input_path)
    print(f"Loaded {input_path} | rows={len(df)}")
    print(df.head())

    # Normalize column names
    if source_col is None:
        source_col = 'Source' if 'Source' in df.columns else 'source'
    if target_col is None:
        target_col = 'Target' if 'Target' in df.columns else 'target'

    df = df.rename(columns={source_col: 'source', target_col: 'target'})
    df = df[['source', 'target']].copy()

    # Remove self-loops
    df = df[df['source'] != df['target']].copy()
    print(f"After removing self-loops: {len(df)} edges")

    # Remap node IDs to 0 .. N-1
    unique_nodes = np.unique(df[['source', 'target']].values)
    mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    df['source'] = df['source'].map(mapping)
    df['target'] = df['target'].map(mapping)
    n_nodes = len(unique_nodes)

    # Canonical undirected orientation: source < target
    df[['source', 'target']] = np.sort(df[['source', 'target']].values, axis=1)

    # Drop duplicate undirected edges
    df = df.drop_duplicates(subset=['source', 'target']).reset_index(drop=True)
    print(f"After deduplication: {len(df)} edges | nodes={n_nodes}")
    print(df.head(10))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned network to: {output_path}")

    return df, n_nodes


if __name__ == "__main__":
    INPUT = '/content/drive/MyDrive/real_networks/email.csv'
    OUTPUT = '/content/drive/MyDrive/real_networks/email.csv'

    preprocess_real_network(INPUT, OUTPUT)