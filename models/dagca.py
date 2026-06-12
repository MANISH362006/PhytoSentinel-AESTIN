"""
DAGCA — Dynamic Atmospheric Graph Construction Algorithm

Core novel contribution of PhytoSentinel-AESTIN.

Learns edge weights (dispersal likelihoods) from meteorological features
using a differentiable MLP viability function. Unlike static GNNs that use
fixed adjacency, DAGCA's graph structure is itself learned end-to-end:
gradients flow through the edge-weight computation back to the MLP.

Architecture:
  edge_features (EDGE_FEAT_DIM) → MLP → sigmoid → edge_weight ∈ (0,1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phyto_config as cfg


class ViabilityMLP(nn.Module):
    """Differentiable viability function: maps meteorological features → dispersal weight."""

    def __init__(self, in_dim: int = cfg.EDGE_FEAT_DIM,
                 hidden: int = cfg.DAGCA_HIDDEN,
                 num_layers: int = cfg.DAGCA_LAYERS,
                 dropout: float = cfg.DAGCA_DROPOUT):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * (num_layers - 1) + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, edge_attr: Tensor) -> Tensor:
        """
        Args:
            edge_attr: (E, EDGE_FEAT_DIM) meteorological features per edge
        Returns:
            weights: (E, 1) dispersal likelihood in (0, 1)
        """
        return torch.sigmoid(self.net(edge_attr))


class DAGCA(nn.Module):
    """
    Dynamic Atmospheric Graph Construction Algorithm.

    Augments a k-NN skeleton graph with learned, physics-informed edge weights.
    The graph structure (which edges carry signal) becomes a function of weather,
    enabling the model to learn how atmospheric conditions drive disease spread.

    Usage:
        dagca = DAGCA()
        weighted_edge_attr, edge_weight = dagca(data.edge_attr, data.edge_index)
    """

    def __init__(self, edge_feat_dim: int = cfg.EDGE_FEAT_DIM,
                 hidden: int = cfg.DAGCA_HIDDEN,
                 threshold: float = cfg.EDGE_THRESHOLD):
        super().__init__()
        self.threshold = threshold
        self.viability = ViabilityMLP(in_dim=edge_feat_dim, hidden=hidden)

        # learnable scale: makes the initial gradient landscape smoother
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, edge_attr: Tensor, edge_index: Tensor,
                prune: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            edge_attr:  (E, EDGE_FEAT_DIM) meteorological edge features
            edge_index: (2, E) graph connectivity
            prune:      if True, remove edges below threshold (inference only)

        Returns:
            edge_attr_weighted: (E', EDGE_FEAT_DIM) — features scaled by weight
            edge_weight:        (E',) — scalar weight per edge
            edge_index_out:     (2, E') — possibly pruned edge index
        """
        scale = torch.exp(self.log_scale)
        edge_weight = self.viability(edge_attr * scale).squeeze(-1)  # (E,)

        if prune:
            mask = edge_weight >= self.threshold
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]
            edge_index = edge_index[:, mask]

        # scale features by learned dispersal weight
        edge_attr_weighted = edge_attr * edge_weight.unsqueeze(-1)

        return edge_attr_weighted, edge_weight, edge_index

    def get_adjacency_matrix(self, edge_index: Tensor, edge_weight: Tensor,
                             num_nodes: int) -> Tensor:
        """
        Build dense adjacency matrix A[i,j] = dispersal weight.
        Used for SENR0 computation (spectral radius).
        """
        A = torch.zeros(num_nodes, num_nodes, device=edge_weight.device)
        A[edge_index[0], edge_index[1]] = edge_weight
        return A
