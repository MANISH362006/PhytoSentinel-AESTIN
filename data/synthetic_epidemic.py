"""
Synthetic Epidemic Dataset Generator for PhytoSentinel-AESTIN.

Simulates plant disease spread on a spatial crop-field graph using
a physics-based SEIR model. Meteorological features drive edge weights.
Generated data is used to train and evaluate DAGCA.
"""

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from scipy.spatial.distance import cdist
from typing import List, Tuple
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phyto_config as cfg


def _wind_dispersal_kernel(positions: np.ndarray, wind_vec: np.ndarray,
                            distance_matrix: np.ndarray) -> np.ndarray:
    """
    Physics-informed dispersal kernel: combines distance decay with wind alignment.
    Returns an N×N weight matrix representing raw dispersal probability.
    """
    N = len(positions)
    # direction from node i to node j
    diff = positions[:, None, :] - positions[None, :, :]   # (N, N, 2)
    norm = np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-8
    direction = diff / norm  # unit vectors

    # alignment of dispersal direction with wind direction
    wind_unit = wind_vec / (np.linalg.norm(wind_vec) + 1e-8)
    alignment = (direction @ wind_unit).squeeze(-1)  # (N, N) in [-1, 1]
    alignment = (alignment + 1) / 2  # shift to [0, 1]

    # exponential distance decay
    decay_rate = 0.5
    dist_decay = np.exp(-decay_rate * distance_matrix)

    kernel = alignment * dist_decay
    np.fill_diagonal(kernel, 0)
    return kernel


def simulate_seir(N: int, positions: np.ndarray, wind_vec: np.ndarray,
                  humidity: np.ndarray, T: int = 30,
                  beta_base: float = 0.3, sigma: float = 0.2,
                  gamma: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SEIR epidemic simulation on a spatial graph.

    Returns:
        states: (T, N) — disease state (0=S, 1=E, 2=I, 3=R)
        node_features: (N, NODE_FEAT_DIM)
        edge_features_full: (N, N, EDGE_FEAT_DIM)
    """
    dist_matrix = cdist(positions, positions)

    # Build meteorological edge features: [wind_speed, humidity_ij, temperature, distance]
    wind_speed = np.linalg.norm(wind_vec)
    edge_feat_full = np.zeros((N, N, cfg.EDGE_FEAT_DIM))
    for i in range(N):
        for j in range(N):
            humidity_ij = (humidity[i] + humidity[j]) / 2.0
            edge_feat_full[i, j] = [
                wind_speed,
                humidity_ij,
                0.5,                          # normalized temperature (placeholder)
                dist_matrix[i, j] / dist_matrix.max(),
            ]

    dispersal = _wind_dispersal_kernel(positions, wind_vec, dist_matrix)
    humidity_effect = (humidity[:, None] + humidity[None, :]) / 2.0
    transmission = beta_base * dispersal * (0.5 + humidity_effect)

    # SEIR states
    S = np.ones(N)
    E = np.zeros(N)
    I = np.zeros(N)
    R = np.zeros(N)

    # seed one infected node
    seed = np.random.randint(N)
    S[seed] = 0
    I[seed] = 1

    states = np.zeros((T, N), dtype=np.int32)

    for t in range(T):
        states[t] = np.argmax(np.stack([S, E, I, R], axis=-1), axis=-1)

        force_of_infection = transmission @ I
        new_exposed = np.random.binomial(1, np.clip(S * force_of_infection, 0, 1))
        new_infected = np.random.binomial(1, np.clip(E * sigma, 0, 1))
        new_recovered = np.random.binomial(1, np.clip(I * gamma, 0, 1))

        S = np.maximum(S - new_exposed, 0)
        E = np.maximum(E + new_exposed - new_infected, 0)
        I = np.maximum(I + new_infected - new_recovered, 0)
        R = np.minimum(R + new_recovered, 1)

    # Node features: S, E, I, R fractions + position + humidity + crop_type
    crop_type = np.random.randint(0, 3, N).astype(float) / 2.0
    node_features = np.column_stack([
        S, E, I, R,
        positions[:, 0] / positions[:, 0].max(),
        positions[:, 1] / positions[:, 1].max(),
        humidity,
        crop_type,
    ]).astype(np.float32)  # (N, 8)

    return states, node_features, edge_feat_full.astype(np.float32)


def _build_pyg_graph(positions: np.ndarray, node_features: np.ndarray,
                     edge_feat_full: np.ndarray, current_states: np.ndarray,
                     next_states: np.ndarray, k_neighbors: int = 10) -> Data:
    """
    Convert simulation snapshot to a PyTorch Geometric Data object.
    Uses k-NN graph topology; edge features from meteorological simulation.
    """
    N = len(positions)
    dist_matrix = cdist(positions, positions)
    np.fill_diagonal(dist_matrix, np.inf)

    # k-NN edges (undirected)
    edge_index_list = []
    for i in range(N):
        neighbors = np.argsort(dist_matrix[i])[:k_neighbors]
        for j in neighbors:
            edge_index_list.append([i, j])
            edge_index_list.append([j, i])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    # Deduplicate
    edge_index = torch.unique(edge_index, dim=1)

    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    edge_attr = torch.tensor(edge_feat_full[src, dst], dtype=torch.float)

    x = torch.tensor(node_features, dtype=torch.float)
    # Label: will node be infected (I or E state) at next step?
    y = torch.tensor((next_states >= 1).astype(np.int64), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                pos=torch.tensor(positions, dtype=torch.float))


def generate_dataset(num_graphs: int = cfg.NUM_GRAPHS,
                     N: int = cfg.NUM_NODES,
                     T: int = cfg.NUM_TIMESTEPS,
                     seed: int = cfg.RANDOM_SEED) -> List[Data]:
    """
    Generate a list of PyG Data objects representing epidemic snapshots.
    Each graph = one field configuration at time t, predicting spread at t+1.
    """
    np.random.seed(seed)
    graphs = []

    for g in range(num_graphs):
        positions = np.random.uniform(0, 100, (N, 2))
        wind_angle = np.random.uniform(0, 2 * np.pi)
        wind_speed = np.random.uniform(1, 10)
        wind_vec = wind_speed * np.array([np.cos(wind_angle), np.sin(wind_angle)])
        humidity = np.random.beta(2, 2, N)

        beta_base = np.random.uniform(0.2, 0.5)
        states, node_features, edge_feat_full = simulate_seir(
            N, positions, wind_vec, humidity, T=T, beta_base=beta_base
        )

        # sample a random timestep (not last)
        t = np.random.randint(0, T - 1)
        graph = _build_pyg_graph(
            positions, node_features, edge_feat_full,
            current_states=states[t],
            next_states=states[t + 1],
        )
        graphs.append(graph)

    print(f"[Dataset] Generated {len(graphs)} graphs | "
          f"avg nodes={N} | avg edges={sum(g.num_edges for g in graphs) // len(graphs)}")
    return graphs


class EpidemicDataset(InMemoryDataset):
    """Wraps generated graphs into a PyG InMemoryDataset for DataLoader compatibility."""

    def __init__(self, graphs: List[Data]):
        super().__init__(root=None)
        self.data, self.slices = self.collate(graphs)

    def __repr__(self):
        return f"EpidemicDataset({len(self)} graphs)"
