"""
SENR0 validation for PhytoSentinel-AESTIN.

SENR0 reads a spectral radius ρ((β/γ)·A) off the *learned* dispersal adjacency. We
have always framed it honestly as a diagnostic. Here we test whether that diagnostic
is *meaningful*: does ρ(A) correlate with how severe the epidemic actually is on each
graph?

For each test graph we compare:
  - ρ(A)  : the learned-graph spectral radius (SENR0), and
  - severity : the observed fraction of susceptible-at-t nodes that become infected
               within the horizon (g.y over the eval mask) — the ground-truth "how hot
               is this outbreak" signal.
  - beta_true : the simulator's transmission coefficient for that graph (secondary).

A positive ρ-vs-severity correlation means the learned graph's connectivity tracks real
epidemic intensity — i.e., SENR0 is an interpretable severity indicator, not noise.

Run:  python experiments/senr0_validation.py
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phyto_config as cfg
from data.synthetic_epidemic import make_splits
from models.senr0 import SENR0
from train import _graph_constructor_weights


def _rank(a):
    """Rank transform for a simple Spearman correlation without scipy."""
    order = np.argsort(np.argsort(a))
    return order.astype(float)


def validate_senr0(seed: int = cfg.RANDOM_SEED, physics: str = "cosine"):
    from experiments.generalization import _train_full_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = make_splits(seed=seed, physics=physics)
    model, _ = _train_full_model(tr, va, device, seed)
    model.eval()

    senr0 = SENR0(gamma=cfg.SENR0_GAMMA).to(device)
    constructor = model.graph_constructor
    rhos, severity, betas = [], [], []
    with torch.no_grad():
        for g in te:
            ea = g.edge_attr.to(device)
            ei = g.edge_index.to(device)
            ew = _graph_constructor_weights(constructor, ea, ei)
            A = constructor.get_adjacency_matrix(ei, ew, g.num_nodes)
            rhos.append(float(senr0(A).detach().cpu()))
            m = g.eval_mask
            severity.append(float(g.y[m].float().mean()) if m.any() else 0.0)
            betas.append(float(g.beta_true))

    rhos = np.array(rhos); severity = np.array(severity); betas = np.array(betas)

    def corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    out = {
        "n_graphs": len(rhos),
        "pearson_rho_vs_severity":  corr(rhos, severity),
        "spearman_rho_vs_severity": corr(_rank(rhos), _rank(severity)),
        "pearson_rho_vs_beta":      corr(rhos, betas),
        "rho_mean": float(rhos.mean()), "rho_std": float(rhos.std()),
    }
    print(f"[SENR0-validation] n={out['n_graphs']} | "
          f"rho vs observed severity: Pearson={out['pearson_rho_vs_severity']:+.3f}, "
          f"Spearman={out['spearman_rho_vs_severity']:+.3f} | "
          f"rho vs true beta: Pearson={out['pearson_rho_vs_beta']:+.3f}")
    print("[SENR0-validation] A positive rho-vs-severity correlation means the learned "
          "graph's spectral radius is a meaningful epidemic-severity indicator.")
    return out


if __name__ == "__main__":
    res = validate_senr0()
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(cfg.RESULTS_DIR, "senr0_validation.json"), "w") as f:
        json.dump(res, f, indent=2)
