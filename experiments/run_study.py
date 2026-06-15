"""
Full experimental study for PhytoSentinel-AESTIN — one command, everything.

Produces the complete, defensible evidence base:
  1. Ablation across MULTIPLE SEEDS on a SHARED dataset  -> mean ± std (Table 1)
  2. External non-GNN baselines on the same frontier       -> "better than what?"
  3. Cross-physics generalization (train A, test B)         -> not circular
  4. Validated calibration (temperature scaling + unc-vs-error)

Every config in a given seed trains on the IDENTICAL train/val/test graphs, so the
comparison is fair; results are aggregated over seeds for statistical honesty.

Run (overnight):  python experiments/run_study.py
Quick smoke test:  python experiments/run_study.py --seeds 42 --quick
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phyto_config as cfg
from data.synthetic_epidemic import make_splits
from train import main as train_main


# (name, no_bayesian, no_dagca, gnn_type)
ABLATION = [
    ("BayesianDAGCA+SAGE (Ours)", False, False, "sage"),
    ("DetDAGCA+SAGE",             True,  False, "sage"),
    ("NoDAGCA+SAGE",              False, True,  "sage"),
    ("NoDAGCA+GCN",               False, True,  "gcn"),
    ("BayesianDAGCA+GAT",         False, False, "gat"),
]

METRICS = ["f1", "auroc", "auprc", "ece"]


class Args:
    def __init__(self, no_bayesian, no_dagca, gnn_type):
        self.no_bayesian = no_bayesian
        self.no_dagca = no_dagca
        self.gnn_type = gnn_type


def run_ablation_multiseed(seeds):
    """Return {config_name: {metric: (mean, std)}} aggregated over seeds."""
    raw = {name: {m: [] for m in METRICS} for name, *_ in ABLATION}

    for seed in seeds:
        print(f"\n########## SEED {seed} ##########")
        splits = make_splits(seed=seed, physics="cosine")   # shared across all configs
        for name, no_bay, no_dag, gnn in ABLATION:
            print(f"\n--- {name} (seed {seed}) ---")
            args = Args(no_bay, no_dag, gnn)
            test_m = train_main(args, splits=splits, seed=seed, tag_suffix=f"_s{seed}")
            for m in METRICS:
                raw[name][m].append(test_m.get(m, 0.0))

    agg = {name: {m: (float(np.mean(v[m])), float(np.std(v[m]))) for m in METRICS}
           for name, v in raw.items()}
    return agg


def _print_ablation(agg, n_seeds):
    print("\n" + "=" * 78)
    print(f"ABLATION (mean ± std over {n_seeds} seed(s); susceptible-node frontier)")
    print("=" * 78)
    print(f"{'Configuration':30s} {'F1':>13} {'AUROC':>13} {'AUPRC':>13}")
    for name in agg:
        f = agg[name]["f1"]; a = agg[name]["auroc"]; p = agg[name]["auprc"]
        print(f"{name:30s} {f[0]:.3f}±{f[1]:.3f}  {a[0]:.3f}±{a[1]:.3f}  {p[0]:.3f}±{p[1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--quick", action="store_true",
                    help="skip generalization + calibration (faster smoke test)")
    ap.add_argument("--skip-generalization", action="store_true")
    args = ap.parse_args()

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    study = {"seeds": args.seeds}

    # 1. ablation
    agg = run_ablation_multiseed(args.seeds)
    study["ablation"] = agg
    _print_ablation(agg, len(args.seeds))

    # 2. external baselines (on first seed's split)
    from experiments.baselines import run_baselines, _print_table
    base = run_baselines(seed=args.seeds[0], physics="cosine")
    study["baselines"] = base
    _print_table(base)

    # 3. cross-physics generalization
    if not args.quick and not args.skip_generalization:
        from experiments.generalization import run_generalization
        study["generalization"] = run_generalization(seed=args.seeds[0])

    # 4. validated calibration
    if not args.quick:
        from torch_geometric.loader import DataLoader
        import torch
        from experiments.generalization import _train_full_model
        from experiments.calibration import analyze_calibration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tr, va, te = make_splits(seed=args.seeds[0], physics="cosine")
        model, _ = _train_full_model(tr, va, device, args.seeds[0])
        study["calibration"] = analyze_calibration(
            model, DataLoader(va, batch_size=cfg.BATCH_SIZE),
            DataLoader(te, batch_size=cfg.BATCH_SIZE), device,
            use_dagca=True, tag="cosine")

    out = os.path.join(cfg.RESULTS_DIR, "study_results.json")
    with open(out, "w") as f:
        json.dump(study, f, indent=2)
    print(f"\n[Study] Complete. Full results saved to {out}")
    print("[Study] Figures in results/figures/ ; per-run JSON in results/")


if __name__ == "__main__":
    main()
