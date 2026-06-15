"""
PhytoSentinel-AESTIN Training Pipeline

Usage:
    python train.py                     # train with default config
    python train.py --no-bayesian       # ablation: deterministic DAGCA
    python train.py --gnn-type gcn      # ablation: GCN backbone
    python train.py --no-dagca          # ablation: static edges (baseline)
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

import phyto_config as cfg
from data.synthetic_epidemic import generate_dataset
from models.gnn import PhytoSentinelModel, PhytoGNN
from models.senr0 import SENR0
from utils.metrics import compute_metrics, format_metrics


def set_seed(seed: int = cfg.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args) -> torch.nn.Module:
    if args.no_dagca:
        # pure baseline: standard GNN, no DAGCA
        return PhytoGNN(gnn_type=args.gnn_type)
    return PhytoSentinelModel(
        use_bayesian=not args.no_bayesian,
        gnn_type=args.gnn_type,
    )


def _masked(batch, logits):
    """
    Restrict node logits/labels to the leakage-safe evaluation frontier
    (nodes Susceptible at time t). Falls back to all nodes if no mask present.
    """
    if hasattr(batch, "eval_mask") and batch.eval_mask is not None:
        m = batch.eval_mask
        return logits[m], batch.y[m]
    return logits, batch.y


def train_epoch(model, loader, optimizer, device, use_dagca: bool,
                class_weight=None) -> dict:
    model.train()
    total_loss, all_logits, all_labels = 0.0, [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if use_dagca:
            logits, kl = model(batch)
        else:
            logits = model(batch.x, batch.edge_index,
                           batch.edge_attr, batch=batch.batch)
            kl = torch.tensor(0.0, device=device)

        # Loss + metrics are computed ONLY over susceptible-at-t nodes.
        logits_m, labels_m = _masked(batch, logits)
        if labels_m.numel() == 0:
            continue
        loss = F.cross_entropy(logits_m, labels_m, weight=class_weight) \
            + cfg.KL_WEIGHT * kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(logits_m.detach())
        all_labels.append(labels_m)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def eval_epoch(model, loader, device, use_dagca: bool,
               class_weight=None) -> dict:
    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []

    for batch in loader:
        batch = batch.to(device)

        if use_dagca:
            logits, kl = model(batch)
        else:
            logits = model(batch.x, batch.edge_index,
                           batch.edge_attr, batch=batch.batch)
            kl = torch.tensor(0.0, device=device)

        logits_m, labels_m = _masked(batch, logits)
        if labels_m.numel() == 0:
            continue
        loss = F.cross_entropy(logits_m, labels_m, weight=class_weight) \
            + cfg.KL_WEIGHT * kl

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(logits_m)
        all_labels.append(labels_m)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def _compute_class_weight(graphs, device) -> torch.Tensor:
    """Inverse-frequency class weights over the susceptible-node frontier."""
    labels = torch.cat([g.y[g.eval_mask] for g in graphs])
    n = int(labels.numel())
    n_pos = max(int(labels.sum().item()), 1)
    n_neg = max(n - n_pos, 1)
    return torch.tensor([n / (2.0 * n_neg), n / (2.0 * n_pos)],
                        dtype=torch.float, device=device)


def main(args, splits=None, seed: int = cfg.RANDOM_SEED, tag_suffix: str = ""):
    """
    Train + evaluate one configuration.

    splits: optional (train, val, test) list-of-Data. If given, the data is reused
            as-is (so every config/baseline trains on IDENTICAL graphs — fair
            comparison and multi-seed studies). If None, data is generated here.
    seed:   controls model init / training stochasticity (and data when generated).
    tag_suffix: appended to checkpoint/result filenames (e.g. a seed tag).
    """
    set_seed(seed)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    print(f"[Train] Config: bayesian={not args.no_bayesian}, "
          f"dagca={not args.no_dagca}, gnn={args.gnn_type}, seed={seed}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if splits is None:
        from data.synthetic_epidemic import make_splits
        train_graphs, val_graphs, test_graphs = make_splits(seed=seed)
    else:
        train_graphs, val_graphs, test_graphs = splits

    train_loader = DataLoader(train_graphs, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=cfg.BATCH_SIZE)
    test_loader  = DataLoader(test_graphs,  batch_size=cfg.BATCH_SIZE)

    print(f"[Data] Train={len(train_graphs)} | Val={len(val_graphs)} | "
          f"Test={len(test_graphs)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    use_dagca = not args.no_dagca
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)

    # Class weighting: the susceptible-node frontier is imbalanced (few nodes get
    # newly infected at t+1), so weight the loss by inverse class frequency.
    class_weight = _compute_class_weight(train_graphs, device)
    print(f"[Train] Class weights (neg, pos): "
          f"({class_weight[0]:.3f}, {class_weight[1]:.3f})")

    tag = _run_tag(args) + tag_suffix

    # ── Training loop ─────────────────────────────────────────────────────────
    # Model selection on validation AUPRC: the task is imbalanced, so AUPRC is the
    # right ranking metric and is far more stable than F1-at-0.5 for checkpointing.
    best_val_auprc = -1.0   # so epoch 1 always writes a checkpoint
    patience_cnt = 0
    history      = {"train": [], "val": []}

    print(f"\n[Train] Starting training for {cfg.EPOCHS} epochs...")
    for epoch in range(1, cfg.EPOCHS + 1):
        train_m = train_epoch(model, train_loader, optimizer, device, use_dagca,
                              class_weight)
        val_m   = eval_epoch(model, val_loader, device, use_dagca, class_weight)
        scheduler.step()

        history["train"].append(train_m)
        history["val"].append(val_m)

        if epoch % cfg.LOG_INTERVAL == 0:
            print(f"Epoch {epoch:03d} | "
                  f"Train: {format_metrics(train_m)} | "
                  f"Val:   {format_metrics(val_m)}")

        # early stopping & checkpointing on val AUPRC
        if val_m["auprc"] > best_val_auprc:
            best_val_auprc = val_m["auprc"]
            patience_cnt = 0
            torch.save(model.state_dict(),
                       os.path.join(cfg.CHECKPOINT_DIR, f"best_{tag}.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.PATIENCE:
                print(f"[Train] Early stopping at epoch {epoch}.")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(os.path.join(cfg.CHECKPOINT_DIR, f"best_{tag}.pt"),
                   map_location=device))
    test_m = eval_epoch(model, test_loader, device, use_dagca, class_weight)
    print(f"\n[Test] {format_metrics(test_m)}")

    # ── Calibration: reliability diagram on the test frontier ──────────────────
    _save_reliability(model, test_loader, device, use_dagca, tag)

    # ── SENR0 analysis ────────────────────────────────────────────────────────
    r0_values = []
    if use_dagca:
        print("\n[SENR0] Computing epidemic thresholds on test graphs...")
        r0_values = _run_senr0_analysis(model, test_graphs, device)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "args":    vars(args),
        "test":    test_m,
        "best_val_auprc": best_val_auprc,
        "n_params":    n_params,
        "r0_values":   r0_values,
    }
    out_path = os.path.join(cfg.RESULTS_DIR, f"results_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved to {out_path}")

    return test_m


@torch.no_grad()
def _save_reliability(model, loader, device, use_dagca: bool, tag: str):
    """
    Collect positive-class probabilities + labels over the test frontier and
    save a reliability diagram (calibration evidence). Best-effort: if plotting
    deps are unavailable the run still succeeds.
    """
    model.eval()
    probs, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        if use_dagca:
            logits, _ = model(batch)
        else:
            logits = model(batch.x, batch.edge_index, batch.edge_attr,
                           batch=batch.batch)
        logits_m, labels_m = _masked(batch, logits)
        if labels_m.numel() == 0:
            continue
        probs.append(torch.softmax(logits_m, dim=-1)[:, 1].cpu().numpy())
        labels.append(labels_m.cpu().numpy())

    if not probs:
        return
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    try:
        from experiments.visualize import plot_reliability_diagram
        ece = plot_reliability_diagram(probs, labels,
                                       out=f"reliability_{tag}.png")
        print(f"[Calib] Reliability diagram saved (ECE={ece:.4f}).")
    except Exception as e:   # plotting is optional; never fail training on it
        print(f"[Calib] Skipped reliability diagram ({e}).")


def _graph_constructor_weights(constructor, edge_attr, edge_index):
    """Run the (Bayesian or deterministic) DAGCA constructor, return edge weights."""
    out = constructor(edge_attr, edge_index)
    # BayesianDAGCA returns (attr_w, weight, index, kl); DAGCA returns (attr_w, weight, index)
    return out[1]


def _run_senr0_analysis(model, graphs, device, max_graphs: int = 50):
    """
    Compute the diagnostic reproduction number R0 for individual test graphs.

    For each graph we build the dense adjacency A from the *learned* DAGCA
    dispersal weights (DAGCA.get_adjacency_matrix), then run the real SENR0
    power-iteration spectral radius on the Next-Generation Matrix K = (β/γ)·A.
    This replaces the earlier mean-weight proxy with the actual eigenvalue path.

    NOTE ON INTERPRETATION: A is trained against a classification loss, not fit to
    recover transmission rates, so ρ(A) is a graph-connectivity *diagnostic*, not a
    validated epidemiological R0. See the SENR0 docstring and README for caveats.
    """
    model.eval()
    senr0 = SENR0(gamma=cfg.SENR0_GAMMA).to(device)
    r0_values = []

    constructor = model.graph_constructor
    with torch.no_grad():
        for g in graphs[:max_graphs]:
            # Move only the tensors we need to the device — do NOT call g.to(device),
            # which mutates the shared graph in place and corrupts the reused split.
            edge_attr  = g.edge_attr.to(device)
            edge_index = g.edge_index.to(device)
            edge_weight = _graph_constructor_weights(constructor, edge_attr, edge_index)

            # Build dense adjacency from the learned dispersal weights.
            A = constructor.get_adjacency_matrix(
                edge_index, edge_weight, num_nodes=g.num_nodes)

            r0_values.append(float(senr0(A).detach().cpu()))

    if r0_values:
        arr = np.array(r0_values)
        beta = float(senr0.beta.detach())
        print(f"[SENR0] spectral diagnostic rho((beta/gamma)*A) over {len(arr)} graphs: "
              f"mean={arr.mean():.3f} | range=[{arr.min():.3f}, {arr.max():.3f}] "
              f"(beta={beta:.2f}, gamma={cfg.SENR0_GAMMA})")
        print("[SENR0] relative connectivity diagnostic from the learned graph — "
              "NOT a validated epidemiological R0 (see README)")
    return r0_values


def _run_tag(args) -> str:
    parts = [args.gnn_type]
    if args.no_bayesian:
        parts.append("det")
    if args.no_dagca:
        parts.append("nodagca")
    return "_".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhytoSentinel-AESTIN Training")
    parser.add_argument("--no-bayesian", action="store_true",
                        help="Use deterministic DAGCA (ablation)")
    parser.add_argument("--no-dagca",    action="store_true",
                        help="Disable DAGCA entirely — baseline GNN (ablation)")
    parser.add_argument("--gnn-type",    default=cfg.GNN_TYPE,
                        choices=["sage", "gat", "gcn"],
                        help="GNN backbone architecture")
    args = parser.parse_args()
    main(args)
