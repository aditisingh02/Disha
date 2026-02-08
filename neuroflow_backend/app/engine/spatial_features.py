"""
NeuroFlow BharatFlow — Spatial feature enrichment for Residual GCN
Adds: neighbor_baseline_gradient, neighbor_residual_mean, corridor_bottleneck_score, adjacency_degree.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger("neuroflow.spatial_features")


def build_default_adjacency(num_nodes: int) -> List[List[int]]:
    """Default: linear chain (segment i connected to i-1, i+1). Override with graph when available."""
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        if i > 0:
            adj[i].append(i - 1)
        if i < num_nodes - 1:
            adj[i].append(i + 1)
    return adj


def enrich_spatial_features(
    base_features: np.ndarray,
    baseline_q50: np.ndarray,
    residual_mean: np.ndarray | None = None,
    adjacency: List[List[int]] | None = None,
) -> np.ndarray:
    """
    Append 4 spatial features to base_features.
    base_features: (N, 9)
    baseline_q50: (N, 48) — use mean over horizons for gradient
    residual_mean: (N,) optional — neighbor residual mean (0 if not available)
    adjacency: list of neighbor indices per node; default linear chain.
    Returns: (N, 13)
    """
    N = base_features.shape[0]
    if adjacency is None:
        adjacency = build_default_adjacency(N)
    if residual_mean is None:
        residual_mean = np.zeros(N, dtype=np.float32)
    baseline_mean = baseline_q50.mean(axis=1)
    neighbor_baseline_gradient = np.zeros(N, dtype=np.float32)
    neighbor_residual_mean_out = np.zeros(N, dtype=np.float32)
    adjacency_degree = np.zeros(N, dtype=np.float32)
    corridor_bottleneck_score = np.zeros(N, dtype=np.float32)
    for i in range(N):
        neighbors = adjacency[i] if i < len(adjacency) else []
        adjacency_degree[i] = len(neighbors)
        if neighbors:
            nb_baseline = np.mean([baseline_mean[j] for j in neighbors if 0 <= j < N])
            neighbor_baseline_gradient[i] = nb_baseline - baseline_mean[i]
            neighbor_residual_mean_out[i] = np.mean([residual_mean[j] for j in neighbors if 0 <= j < N])
        corridor_bottleneck_score[i] = 1.0 / (1.0 + adjacency_degree[i])
    extra = np.stack(
        [neighbor_baseline_gradient, neighbor_residual_mean_out, corridor_bottleneck_score, adjacency_degree],
        axis=1,
    )
    return np.concatenate([base_features, extra], axis=1).astype(np.float32)
