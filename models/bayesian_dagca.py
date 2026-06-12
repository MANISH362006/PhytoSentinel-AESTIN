"""
Bayesian DAGCA — Bayesian Edge Uncertainty Extension

Extends DAGCA by treating each edge weight as a Beta-distributed random variable
instead of a point estimate. This captures aleatoric uncertainty in pathogen
dispersal (e.g., inconsistent wind patterns, sensor noise).

During training: sample edge weights via the reparameterization trick.
During inference: use the posterior mean (α / (α + β)).

The KL divergence between the learned Beta posterior and the Beta(1,1) prior
is added to the loss to regularize the learned uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phyto_config as cfg


class BetaHead(nn.Module):
    """
    Outputs (log_alpha, log_beta) for a Beta distribution from edge features.
    Constrained to α, β > 0 via softplus.
    """

    def __init__(self, in_dim: int, hidden: int = cfg.DAGCA_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 2),   # → [log_alpha, log_beta]
        )
        # initialize to near-uniform Beta(1,1)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        out = self.net(edge_attr)
        # softplus ensures α, β > 0; add 1 so we start near Beta(1,1)
        alpha = F.softplus(out[:, 0]) + 1.0
        beta  = F.softplus(out[:, 1]) + 1.0
        return alpha, beta


def _beta_kl_divergence(alpha: Tensor, beta: Tensor,
                         prior_alpha: float = cfg.PRIOR_ALPHA,
                         prior_beta: float = cfg.PRIOR_BETA) -> Tensor:
    """
    KL[Beta(α,β) || Beta(prior_α, prior_β)] — analytically tractable.

    KL = log[B(p_α, p_β)/B(α, β)]
       + (α - p_α)·ψ(α) + (β - p_β)·ψ(β)
       + (p_α + p_β - α - β)·ψ(α + β)
    where ψ is the digamma function and B is the Beta function.
    """
    pa = torch.tensor(prior_alpha, device=alpha.device, dtype=alpha.dtype)
    pb = torch.tensor(prior_beta,  device=alpha.device, dtype=alpha.dtype)

    kl = (torch.lgamma(pa + pb) - torch.lgamma(pa) - torch.lgamma(pb)
          - torch.lgamma(alpha + beta) + torch.lgamma(alpha) + torch.lgamma(beta)
          + (alpha - pa) * torch.digamma(alpha)
          + (beta  - pb) * torch.digamma(beta)
          + (pa + pb - alpha - beta) * torch.digamma(alpha + beta))
    return kl.mean()


def _sample_beta(alpha: Tensor, beta: Tensor,
                 num_samples: int = cfg.BAYESIAN_SAMPLES) -> Tensor:
    """
    Sample from Beta(α, β) via Kumaraswamy approximation (differentiable).

    The Kumaraswamy distribution is a close approximation to Beta that admits
    a closed-form reparameterization: x = (1 - u^(1/b))^(1/a).
    """
    # approximate: a ≈ α, b ≈ β  (works well for α,β > 0.5)
    a = alpha.unsqueeze(0).expand(num_samples, -1)   # (S, E)
    b = beta.unsqueeze(0).expand(num_samples, -1)

    u = torch.zeros_like(a).uniform_().clamp(1e-6, 1 - 1e-6)
    samples = (1 - u.pow(1.0 / b)).pow(1.0 / a)     # (S, E)
    return samples.mean(dim=0)                        # (E,)  Monte Carlo mean


class BayesianDAGCA(nn.Module):
    """
    Bayesian Dynamic Atmospheric Graph Construction Algorithm.

    Learns per-edge Beta distributions over dispersal weights.
    Provides uncertainty estimates alongside point predictions.

    Usage:
        bdagca = BayesianDAGCA()
        edge_attr_w, edge_weight, edge_index, kl = bdagca(edge_attr, edge_index)
        loss = task_loss + cfg.KL_WEIGHT * kl
    """

    def __init__(self, edge_feat_dim: int = cfg.EDGE_FEAT_DIM,
                 hidden: int = cfg.DAGCA_HIDDEN,
                 threshold: float = cfg.EDGE_THRESHOLD,
                 num_samples: int = cfg.BAYESIAN_SAMPLES):
        super().__init__()
        self.threshold   = threshold
        self.num_samples = num_samples
        self.beta_head   = BetaHead(in_dim=edge_feat_dim, hidden=hidden)

        # shared feature encoder (same role as ViabilityMLP in DAGCA)
        self.encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

    def forward(self, edge_attr: Tensor, edge_index: Tensor,
                prune: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            edge_attr:   (E, EDGE_FEAT_DIM)
            edge_index:  (2, E)
            prune:       remove low-confidence edges at inference

        Returns:
            edge_attr_weighted: (E', EDGE_FEAT_DIM)
            edge_weight:        (E',) — sampled (train) or mean (eval)
            edge_index_out:     (2, E')
            kl:                 scalar KL divergence term for the loss
        """
        encoded = self.encoder(edge_attr)
        alpha, beta = self.beta_head(encoded)

        kl = _beta_kl_divergence(alpha, beta)

        if self.training:
            edge_weight = _sample_beta(alpha, beta, self.num_samples)
        else:
            edge_weight = alpha / (alpha + beta)   # posterior mean

        if prune:
            mask = edge_weight >= self.threshold
            edge_weight  = edge_weight[mask]
            edge_attr    = edge_attr[mask]
            edge_index   = edge_index[:, mask]

        edge_attr_weighted = edge_attr * edge_weight.unsqueeze(-1)
        return edge_attr_weighted, edge_weight, edge_index, kl

    def edge_uncertainty(self, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        """
        At inference: return posterior mean and std of each edge weight.
        Useful for visualization and uncertainty-aware decision making.
        """
        self.eval()
        with torch.no_grad():
            encoded = self.encoder(edge_attr)
            alpha, beta = self.beta_head(encoded)
            mean = alpha / (alpha + beta)
            var  = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            std  = var.sqrt()
        return mean, std
