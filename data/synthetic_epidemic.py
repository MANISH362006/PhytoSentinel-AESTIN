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


# Available ground-truth dispersal physics. DAGCA receives the SAME edge features
# regardless of which one generated the data, so training on one and testing on the
# other (see experiments/generalization.py) is a genuine out-of-distribution test —
# the key result that turns "synthetic, possibly circular" into "transfers across
# dispersal models".
PHYSICS_MODELS = ("cosine", "plume")


def _wind_dispersal_kernel(positions: np.ndarray, wind_vec: np.ndarray,
                            distance_matrix: np.ndarray,
                            physics: str = "cosine") -> np.ndarray:
    """
    Ground-truth spore-dispersal kernel. Returns an N×N matrix of raw dispersal
    probability from node i to node j.

    physics="cosine" : distance-decay × wind-direction cosine alignment (smooth,
                       isotropic-ish falloff biased downwind).
    physics="plume"  : an anisotropic Gaussian-plume model — spores travel ALONG the
                       wind axis and spread laterally (Gaussian in crosswind distance),
                       with almost no upwind transport. Functionally very different
                       from the cosine model, which is what makes it a fair OOD test.
    """
    diff = positions[None, :, :] - positions[:, None, :]    # i -> j vectors (N,N,2)
    dist_norm = distance_matrix / (distance_matrix.max() + 1e-8)
    wind_unit = wind_vec / (np.linalg.norm(wind_vec) + 1e-8)

    if physics == "cosine":
        norm = np.linalg.norm(diff, axis=-1) + 1e-8
        cos  = np.einsum('ijk,k->ij', diff, wind_unit) / norm   # (N,N) in [-1,1]
        alignment  = (cos + 1) / 2                              # [0,1]
        dist_decay = np.exp(-12.0 * dist_norm)
        kernel = alignment * dist_decay

    elif physics == "plume":
        # along-wind (downwind positive) and crosswind components, normalized
        d_par  = np.einsum('ijk,k->ij', diff, wind_unit)        # signed along-wind
        d_perp2 = np.clip((diff ** 2).sum(-1) - d_par ** 2, 0, None)
        scale  = distance_matrix.max() + 1e-8
        dpar_n = d_par / scale
        dperp_n = np.sqrt(d_perp2) / scale
        downwind = (d_par > 0).astype(float)                    # negligible upwind
        kernel = downwind * np.exp(-6.0 * dpar_n) * np.exp(-(dperp_n ** 2) / (2 * 0.04))

    else:
        raise ValueError(f"Unknown physics model: {physics!r}. Use one of {PHYSICS_MODELS}.")

    np.fill_diagonal(kernel, 0)
    return kernel


def simulate_seir(N: int, positions: np.ndarray, wind_vec: np.ndarray,
                  humidity: np.ndarray, T: int = 30,
                  beta_base: float = 0.4, sigma: float = 0.3,
                  gamma: float = 0.1, num_seeds: int = 3,
                  physics: str = "cosine"
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SEIR epidemic simulation on a spatial graph.

    Returns:
        states: (T, N) — disease state (0=S, 1=E, 2=I, 3=R)
        node_features: (N, NODE_FEAT_DIM)
        edge_features_full: (N, N, EDGE_FEAT_DIM)
    """
    dist_matrix = cdist(positions, positions)

    # Build meteorological edge features (vectorized):
    #   [wind_speed, humidity_ij, wind_alignment_ij, normalized_distance_ij]
    # wind_alignment is the cosine between the i->j direction and the wind vector,
    # shifted to [0,1] — this is the directional physics that actually drives
    # dispersal, so DAGCA can now learn the "downwind" effect (the old feature 2
    # was a constant temperature placeholder carrying no signal).
    wind_speed = float(np.linalg.norm(wind_vec))
    wind_unit  = wind_vec / (np.linalg.norm(wind_vec) + 1e-8)

    diff      = positions[:, None, :] - positions[None, :, :]          # (N,N,2)
    direction = diff / (np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-8)
    alignment = (np.einsum('ijk,k->ij', direction, wind_unit) + 1) / 2  # (N,N)∈[0,1]
    humidity_ij = (humidity[:, None] + humidity[None, :]) / 2.0         # (N,N)
    dist_norm   = dist_matrix / (dist_matrix.max() + 1e-8)              # (N,N)

    edge_feat_full = np.zeros((N, N, cfg.EDGE_FEAT_DIM))
    edge_feat_full[:, :, 0] = wind_speed
    edge_feat_full[:, :, 1] = humidity_ij
    edge_feat_full[:, :, 2] = alignment
    edge_feat_full[:, :, 3] = dist_norm

    dispersal = _wind_dispersal_kernel(positions, wind_vec, dist_matrix, physics=physics)
    # normalize so the two physics models have comparable overall transmission scale,
    # keeping the epidemic in a meaningful regime regardless of kernel shape.
    dispersal = dispersal / (dispersal.max() + 1e-8)
    humidity_effect = (humidity[:, None] + humidity[None, :]) / 2.0
    transmission = beta_base * dispersal * (0.5 + humidity_effect)

    # SEIR states
    S = np.ones(N)
    E = np.zeros(N)
    I = np.zeros(N)
    R = np.zeros(N)

    # seed a few infected nodes (a single seed rarely ignites a spatial epidemic)
    seeds = np.random.choice(N, size=min(num_seeds, N), replace=False)
    S[seeds] = 0
    I[seeds] = 1

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

    # STATIC node features only: position + humidity + crop_type. (N, 4)
    # The time-varying SEIR state is intentionally NOT baked in here — it is
    # attached per-snapshot in _build_pyg_graph using the state at the *sampled*
    # timestep t (not the final state), to avoid leaking the future outcome.
    crop_type = np.random.randint(0, 3, N).astype(float) / 2.0
    static_features = np.column_stack([
        positions[:, 0] / positions[:, 0].max(),
        positions[:, 1] / positions[:, 1].max(),
        humidity,
        crop_type,
    ]).astype(np.float32)  # (N, 4)

    return states, static_features, edge_feat_full.astype(np.float32)


def _state_one_hot(state_vec: np.ndarray) -> np.ndarray:
    """One-hot encode an SEIR state vector (values in {0,1,2,3}) → (N, 4)."""
    oh = np.zeros((state_vec.shape[0], 4), dtype=np.float32)
    oh[np.arange(state_vec.shape[0]), state_vec.astype(int)] = 1.0
    return oh


def _build_pyg_graph(positions: np.ndarray, static_features: np.ndarray,
                     edge_feat_full: np.ndarray, current_states: np.ndarray,
                     next_states: np.ndarray, k_neighbors: int = 10) -> Data:
    """
    Convert one simulation snapshot (state at time t) to a PyG Data object.

    LEAKAGE-SAFE TASK DEFINITION
    ----------------------------
    The SEIR process is monotone (S -> E -> I -> R, no return to S). Two leakage
    paths exist if handled naively, and both are closed here:

      1. Node features must encode the state at the *sampled* timestep t — NOT
         the final state of the simulation. We build the SEIR one-hot from
         `current_states` (state at t) and concatenate it with static features.

      2. A node already E/I/R at t trivially satisfies (next_state >= 1), so its
         label is a copy of its own current-state feature. We therefore evaluate
         ONLY on nodes that are Susceptible at t (`eval_mask`). For those nodes the
         SEIR one-hot is the constant [1,0,0,0] and carries no label information —
         the model must infer next-step infection from position, humidity, and the
         states of neighbouring nodes via message passing.

    All nodes stay in the graph for message passing; the mask restricts the loss
    and metrics to the genuinely non-trivial infection frontier.
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

    # Node features = [SEIR one-hot at time t] ++ [static features]  → (N, 8)
    state_oh = _state_one_hot(current_states)                       # (N, 4)
    node_features = np.concatenate([state_oh, static_features], 1)  # (N, 8)
    x = torch.tensor(node_features, dtype=torch.float)

    # Label: will the node be infected (E/I/R) at the next step?
    y = torch.tensor((next_states >= 1).astype(np.int64), dtype=torch.long)

    # Evaluation mask: only nodes Susceptible at t define the prediction frontier.
    eval_mask = torch.tensor(current_states == 0, dtype=torch.bool)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                eval_mask=eval_mask,
                pos=torch.tensor(positions, dtype=torch.float))


def generate_dataset(num_graphs: int = cfg.NUM_GRAPHS,
                     N: int = cfg.NUM_NODES,
                     T: int = cfg.NUM_TIMESTEPS,
                     seed: int = cfg.RANDOM_SEED,
                     physics: str = "cosine",
                     verbose: bool = True) -> List[Data]:
    """
    Generate a list of PyG Data objects representing epidemic snapshots.
    Each graph = one field configuration at time t, predicting spread at t+1.

    `physics` selects the ground-truth dispersal kernel ("cosine" or "plume").
    `seed` makes generation fully reproducible, so the same call yields identical
    graphs — this is what lets every ablation config and baseline train/test on the
    *same* data for a fair comparison.
    """
    np.random.seed(seed)
    graphs = []

    for g in range(num_graphs):
        positions = np.random.uniform(0, 100, (N, 2))
        wind_angle = np.random.uniform(0, 2 * np.pi)
        wind_speed = np.random.uniform(1, 10)
        wind_vec = wind_speed * np.array([np.cos(wind_angle), np.sin(wind_angle)])
        humidity = np.random.beta(2, 2, N)

        beta_base = np.random.uniform(0.3, 0.5)
        states, static_features, edge_feat_full = simulate_seir(
            N, positions, wind_vec, humidity, T=T, beta_base=beta_base, physics=physics
        )

        # sample a random timestep (not last)
        t = np.random.randint(0, T - 1)
        graph = _build_pyg_graph(
            positions, static_features, edge_feat_full,
            current_states=states[t],
            next_states=states[t + 1],
        )
        graphs.append(graph)

    if verbose:
        # Class balance on the *evaluated* (susceptible-at-t) frontier — the honest stat.
        sus_labels = torch.cat([g.y[g.eval_mask] for g in graphs])
        n_eval = int(sus_labels.numel())
        pos_rate = float(sus_labels.float().mean()) if n_eval else 0.0
        print(f"[Dataset:{physics}] Generated {len(graphs)} graphs | "
              f"nodes={N} | avg edges={sum(g.num_edges for g in graphs) // len(graphs)}")
        print(f"[Dataset:{physics}] Frontier: {n_eval} susceptible-node decisions | "
              f"{pos_rate:.1%} newly infected at t+1 (positive class)")
    return graphs


def make_splits(num_graphs: int = cfg.NUM_GRAPHS, seed: int = cfg.RANDOM_SEED,
                physics: str = "cosine", verbose: bool = True):
    """
    Build a reproducible (train, val, test) split of epidemic graphs.
    Reused across every ablation config / baseline so comparisons are fair.
    """
    graphs = generate_dataset(num_graphs=num_graphs, seed=seed,
                              physics=physics, verbose=verbose)
    n_total = len(graphs)
    n_train = int(n_total * cfg.TRAIN_SPLIT)
    n_val   = int(n_total * cfg.VAL_SPLIT)
    return (graphs[:n_train],
            graphs[n_train:n_train + n_val],
            graphs[n_train + n_val:])


class EpidemicDataset(InMemoryDataset):
    """Wraps generated graphs into a PyG InMemoryDataset for DataLoader compatibility."""

    def __init__(self, graphs: List[Data]):
        super().__init__(root=None)
        self.data, self.slices = self.collate(graphs)

    def __repr__(self):
        return f"EpidemicDataset({len(self)} graphs)"
