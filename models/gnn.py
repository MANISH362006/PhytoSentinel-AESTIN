"""
PhytoSentinel GNN — Message Passing Network on DAGCA/BayesianDAGCA graphs.

Supports three backbone variants:
  - GraphSAGE (default, robust, works well with small Colab budgets)
  - GATv2 (attention, best accuracy but slower)
  - GCN (fastest, baseline comparison)

The key difference from standard GNNs: edge weights from DAGCA are used as
explicit message-passing weights, so the learned dispersal probabilities
directly control information flow between crop nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import (
    SAGEConv, GATv2Conv, GCNConv, global_mean_pool, global_add_pool
)
from torch_geometric.utils import softmax as pyg_softmax

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phyto_config as cfg


class WeightedSAGELayer(nn.Module):
    """GraphSAGE layer that incorporates DAGCA edge weights into aggregation."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin_self  = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.norm      = nn.LayerNorm(out_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: Tensor | None = None) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        msg = x[src]  # (E, D)

        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1)

        # weighted mean aggregation per node
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)

        # normalize by weighted degree
        if edge_weight is not None:
            deg = torch.zeros(x.size(0), device=x.device)
            deg.index_add_(0, dst, edge_weight)
            deg = deg.clamp(min=1)
            agg = agg / deg.unsqueeze(-1)
        else:
            deg = torch.zeros(x.size(0), device=x.device)
            deg.index_add_(0, dst, torch.ones(src.size(0), device=x.device))
            deg = deg.clamp(min=1)
            agg = agg / deg.unsqueeze(-1)

        out = self.lin_self(x) + self.lin_neigh(agg)
        return self.dropout(F.gelu(self.norm(out)))


class PhytoGNN(nn.Module):
    """
    Full GNN pipeline: DAGCA graph → message passing → node classification.

    Architecture:
      [node_features] → input_proj → L × WeightedSAGELayer → classifier → [logits]
                               ↑
                      DAGCA edge_weights (gate messages)
    """

    def __init__(self,
                 node_feat_dim: int  = cfg.NODE_FEAT_DIM,
                 edge_feat_dim: int  = cfg.EDGE_FEAT_DIM,
                 hidden: int         = cfg.GNN_HIDDEN,
                 num_layers: int     = cfg.GNN_LAYERS,
                 num_classes: int    = cfg.NUM_CLASSES,
                 dropout: float      = cfg.GNN_DROPOUT,
                 gnn_type: str       = cfg.GNN_TYPE):
        super().__init__()
        self.gnn_type   = gnn_type
        self.num_layers = num_layers

        # NOTE: We deliberately do NOT add a second learned edge gate here.
        # Edge message weights come *only* from DAGCA (passed in as edge_weight),
        # so the "DAGCA on/off" ablation isolates DAGCA's contribution rather than
        # letting a redundant in-GNN edge projection silently relearn it.

        # input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # message-passing layers
        if gnn_type == "sage":
            self.convs = nn.ModuleList([
                WeightedSAGELayer(hidden, hidden, dropout=dropout)
                for _ in range(num_layers)
            ])
        elif gnn_type == "gat":
            # concat=False averages the attention heads, so every layer outputs
            # `hidden` channels. This keeps the residual / skip / classifier dims
            # consistent at `hidden` (concat=True would balloon to hidden*heads and
            # break the residual add).
            heads = cfg.GAT_HEADS
            self.convs = nn.ModuleList([
                GATv2Conv(hidden, hidden, heads=heads, dropout=dropout,
                          concat=False, edge_dim=edge_feat_dim)
                for _ in range(num_layers)
            ])
        elif gnn_type == "gcn":
            self.convs = nn.ModuleList([
                GCNConv(hidden, hidden) for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        # skip connections across layers
        self.skip_proj = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(num_layers)
        ])

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Tensor, edge_weight: Tensor | None = None,
                batch: Tensor | None = None) -> Tensor:
        """
        Args:
            x:           (N, node_feat_dim)
            edge_index:  (2, E)
            edge_attr:   (E, edge_feat_dim) — meteorological features
            edge_weight: (E,) — DAGCA dispersal weights
            batch:       (N,) — graph batch assignments (for pooling)

        Returns:
            logits: (N, num_classes) for node classification
        """
        h = self.input_proj(x)

        # Edge weights come solely from DAGCA. When DAGCA is disabled
        # (edge_weight is None) the SAGE/GCN backbones fall back to standard
        # unweighted aggregation — the honest "no DAGCA" baseline.
        for i, conv in enumerate(self.convs):
            if self.gnn_type == "sage":
                h_new = conv(h, edge_index, edge_weight)
            elif self.gnn_type == "gat":
                h_new = conv(h, edge_index, edge_attr=edge_attr)
            elif self.gnn_type == "gcn":
                h_new = conv(h, edge_index, edge_weight=edge_weight)
                h_new = F.gelu(h_new)
            h = h + self.skip_proj[i](h_new)   # residual

        return self.classifier(h)


class PhytoSentinelModel(nn.Module):
    """
    Full PhytoSentinel-AESTIN model: integrates BayesianDAGCA + PhytoGNN.

    This is the top-level class used for training and inference.
    """

    def __init__(self, use_bayesian: bool = True, **gnn_kwargs):
        super().__init__()
        self.use_bayesian = use_bayesian

        if use_bayesian:
            from models.bayesian_dagca import BayesianDAGCA
            self.graph_constructor = BayesianDAGCA()
        else:
            from models.dagca import DAGCA
            self.graph_constructor = DAGCA()

        self.gnn = PhytoGNN(**gnn_kwargs)

    def forward(self, data) -> tuple[Tensor, Tensor]:
        """
        Args:
            data: PyG Data with .x, .edge_index, .edge_attr

        Returns:
            logits: (N, num_classes)
            kl:     scalar (0 if not Bayesian)
        """
        if self.use_bayesian:
            edge_attr_w, edge_weight, edge_index, kl = self.graph_constructor(
                data.edge_attr, data.edge_index)
        else:
            edge_attr_w, edge_weight, edge_index = self.graph_constructor(
                data.edge_attr, data.edge_index)
            kl = torch.tensor(0.0, device=data.x.device)

        logits = self.gnn(data.x, edge_index, edge_attr_w, edge_weight,
                          getattr(data, 'batch', None))
        return logits, kl
