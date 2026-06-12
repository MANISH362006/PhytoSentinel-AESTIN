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

import config as cfg
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


def train_epoch(model, loader, optimizer, device, use_dagca: bool) -> dict:
    model.train()
    total_loss, all_logits, all_labels = 0.0, [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if use_dagca:
            logits, kl = model(batch)
            task_loss = F.cross_entropy(logits, batch.y)
            loss = task_loss + cfg.KL_WEIGHT * kl
        else:
            logits = model(batch.x, batch.edge_index,
                           batch.edge_attr, batch=batch.batch)
            loss = F.cross_entropy(logits, batch.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(logits.detach())
        all_labels.append(batch.y)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def eval_epoch(model, loader, device, use_dagca: bool) -> dict:
    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []

    for batch in loader:
        batch = batch.to(device)

        if use_dagca:
            logits, kl = model(batch)
            loss = F.cross_entropy(logits, batch.y) + cfg.KL_WEIGHT * kl
        else:
            logits = model(batch.x, batch.edge_index,
                           batch.edge_attr, batch=batch.batch)
            loss = F.cross_entropy(logits, batch.y)

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(logits)
        all_labels.append(batch.y)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main(args):
    set_seed()
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    print(f"[Train] Config: bayesian={not args.no_bayesian}, "
          f"dagca={not args.no_dagca}, gnn={args.gnn_type}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[Data] Generating synthetic epidemic dataset...")
    graphs = generate_dataset()

    n_total = len(graphs)
    n_train = int(n_total * cfg.TRAIN_SPLIT)
    n_val   = int(n_total * cfg.VAL_SPLIT)

    train_graphs = graphs[:n_train]
    val_graphs   = graphs[n_train:n_train + n_val]
    test_graphs  = graphs[n_train + n_val:]

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

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1  = 0.0
    patience_cnt = 0
    history      = {"train": [], "val": []}

    print(f"\n[Train] Starting training for {cfg.EPOCHS} epochs...")
    for epoch in range(1, cfg.EPOCHS + 1):
        train_m = train_epoch(model, train_loader, optimizer, device, use_dagca)
        val_m   = eval_epoch(model, val_loader, device, use_dagca)
        scheduler.step()

        history["train"].append(train_m)
        history["val"].append(val_m)

        if epoch % cfg.LOG_INTERVAL == 0:
            print(f"Epoch {epoch:03d} | "
                  f"Train: {format_metrics(train_m)} | "
                  f"Val:   {format_metrics(val_m)}")

        # early stopping & checkpointing
        if val_m["f1"] > best_val_f1:
            best_val_f1  = val_m["f1"]
            patience_cnt = 0
            tag = _run_tag(args)
            torch.save(model.state_dict(),
                       os.path.join(cfg.CHECKPOINT_DIR, f"best_{tag}.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.PATIENCE:
                print(f"[Train] Early stopping at epoch {epoch}.")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    tag = _run_tag(args)
    model.load_state_dict(
        torch.load(os.path.join(cfg.CHECKPOINT_DIR, f"best_{tag}.pt"),
                   map_location=device))
    test_m = eval_epoch(model, test_loader, device, use_dagca)
    print(f"\n[Test] {format_metrics(test_m)}")

    # ── SENR0 analysis ────────────────────────────────────────────────────────
    if use_dagca and not args.no_bayesian:
        print("\n[SENR0] Computing epidemic thresholds on test graphs...")
        _run_senr0_analysis(model, test_loader, device)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "args":    vars(args),
        "test":    test_m,
        "best_val_f1": best_val_f1,
        "n_params":    n_params,
    }
    out_path = os.path.join(cfg.RESULTS_DIR, f"results_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved to {out_path}")

    return test_m


def _run_senr0_analysis(model, loader, device):
    """Compute R0 for a sample of test graphs and print epidemic status."""
    model.eval()
    senr0 = SENR0().to(device)
    r0_values = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            batch = batch.to(device)
            _, edge_weight, edge_index, _ = model.graph_constructor(
                batch.edge_attr, batch.edge_index)

            # build adjacency for first graph in batch
            mask = (batch.batch == 0) if hasattr(batch, 'batch') else \
                   torch.ones(batch.num_nodes, dtype=torch.bool)
            num_nodes = mask.sum().item()

            A = model.graph_constructor.beta_head  # reuse
            A_dense = torch.zeros(num_nodes, num_nodes, device=device)
            # simplified: use mean edge weight as proxy
            r0_proxy = float(edge_weight.mean()) * (1.0 / cfg.SENR0_GAMMA)
            r0_values.append(r0_proxy)
            print(f"  Graph sample | mean_weight={edge_weight.mean():.4f} | "
                  f"R0≈{r0_proxy:.3f} | "
                  f"Epidemic={'YES' if r0_proxy > 1 else 'NO'}")

    print(f"[SENR0] Mean R0 across samples: {np.mean(r0_values):.3f}")


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
