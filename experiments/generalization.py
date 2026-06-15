"""
Cross-physics generalization for PhytoSentinel-AESTIN.

THE point of this experiment. The standard, fatal criticism of a synthetic study
is circularity: "the model is built around the same physics that generated the
data, so of course it wins." We attack that directly.

We have TWO different ground-truth dispersal kernels (see data/synthetic_epidemic.py):
  - cosine : distance-decay x wind-direction cosine alignment
  - plume  : anisotropic Gaussian plume (downwind advection + lateral spread)

The model receives the SAME edge features in both cases. We train on one physics
and test on the OTHER. If performance transfers (OOD AUPRC well above base rate, and
not far below in-distribution), the model has learned a generalizable notion of
weather-driven spread — not memorized one kernel. That is a genuine, defensible
result and the single biggest credibility lever available without real-world data.

Output: a train-physics x test-physics matrix of AUPRC / F1, saved to JSON.

Run:  python experiments/generalization.py
"""

import os
import sys
import json
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phyto_config as cfg
from data.synthetic_epidemic import make_splits, PHYSICS_MODELS
from models.gnn import PhytoSentinelModel
from train import (set_seed, train_epoch, eval_epoch, _compute_class_weight)


def _train_full_model(train_g, val_g, device, seed, epochs=None):
    """Train the full BayesianDAGCA+SAGE model; return the best-val checkpoint in memory."""
    epochs = epochs or cfg.EPOCHS
    set_seed(seed)
    model = PhytoSentinelModel(use_bayesian=True, gnn_type="sage").to(device)
    opt = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    cw = _compute_class_weight(train_g, device)

    train_loader = DataLoader(train_g, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_g,   batch_size=cfg.BATCH_SIZE)

    best_auprc, best_state, patience = -1.0, None, 0
    for ep in range(1, epochs + 1):
        train_epoch(model, train_loader, opt, device, use_dagca=True, class_weight=cw)
        vm = eval_epoch(model, val_loader, device, use_dagca=True, class_weight=cw)
        sched.step()
        if vm["auprc"] > best_auprc:
            best_auprc = vm["auprc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, cw


def run_generalization(seed: int = cfg.RANDOM_SEED, epochs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Generalization] device={device} | physics models={PHYSICS_MODELS}")

    # Build fixed splits for each physics (same seed -> reproducible).
    splits = {p: make_splits(seed=seed, physics=p) for p in PHYSICS_MODELS}

    matrix = {}   # matrix[train_physics][test_physics] = metrics
    for train_p in PHYSICS_MODELS:
        tr, va, _ = splits[train_p]
        print(f"\n=== Training on physics='{train_p}' ===")
        model, cw = _train_full_model(tr, va, device, seed, epochs)

        matrix[train_p] = {}
        for test_p in PHYSICS_MODELS:
            _, _, te = splits[test_p]
            test_loader = DataLoader(te, batch_size=cfg.BATCH_SIZE)
            m = eval_epoch(model, test_loader, device, use_dagca=True, class_weight=cw)
            tag = "in-dist" if test_p == train_p else "OOD"
            matrix[train_p][test_p] = m
            print(f"  test on '{test_p}' [{tag}]: "
                  f"F1={m['f1']:.4f} AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f}")

    _print_matrix(matrix)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out = os.path.join(cfg.RESULTS_DIR, "generalization.json")
    with open(out, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"\n[Generalization] saved to {out}")
    return matrix


def _print_matrix(matrix):
    print("\n" + "=" * 64)
    print("CROSS-PHYSICS GENERALIZATION  (AUPRC; diagonal = in-distribution)")
    print("=" * 64)
    phys = list(matrix.keys())
    header = "train\\test".ljust(14) + "".join(p.ljust(12) for p in phys)
    print(header)
    for tr in phys:
        row = tr.ljust(14)
        for te in phys:
            row += f"{matrix[tr][te]['auprc']:.4f}".ljust(12)
        print(row)
    print("\nRead: off-diagonal cells are out-of-distribution. If they stay well "
          "above the positive base rate, the model generalizes across dispersal physics.")


if __name__ == "__main__":
    run_generalization()
