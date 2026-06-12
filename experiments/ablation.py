"""
Ablation Study for PhytoSentinel-AESTIN

Runs 5 experimental configurations systematically and saves a comparison table.
This is the core experiment for the conference paper's results section.

Ablation matrix:
  | Config              | DAGCA | Bayesian | GNN type |
  |---------------------|-------|----------|----------|
  | Full (Ours)         |  YES  |   YES    |  SAGE    |
  | Det. DAGCA          |  YES  |   NO     |  SAGE    |
  | No DAGCA (GCN)      |  NO   |   -      |  GCN     |
  | No DAGCA (SAGE)     |  NO   |   -      |  SAGE    |
  | GAT backbone        |  YES  |   YES    |  GAT     |

Run with: python experiments/ablation.py
"""

import sys
import os
import json
import argparse
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phyto_config as cfg
from train import main as train_main


ABLATION_CONFIGS = [
    # name,                    no_bayesian, no_dagca, gnn_type
    ("BayesianDAGCA+SAGE (Ours)", False,      False,    "sage"),
    ("DetDAGCA+SAGE",             True,       False,    "sage"),
    ("NoDAGCA+SAGE",              False,      True,     "sage"),
    ("NoDAGCA+GCN",               False,      True,     "gcn"),
    ("BayesianDAGCA+GAT",         False,      False,    "gat"),
]


class FakeArgs:
    def __init__(self, no_bayesian, no_dagca, gnn_type):
        self.no_bayesian = no_bayesian
        self.no_dagca    = no_dagca
        self.gnn_type    = gnn_type


def run_ablation(output_path: str = "results/ablation_results.csv"):
    os.makedirs("results", exist_ok=True)
    rows = []

    for name, no_bay, no_dag, gnn in ABLATION_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print('='*60)

        args = FakeArgs(no_bayesian=no_bay, no_dagca=no_dag, gnn_type=gnn)
        try:
            metrics = train_main(args)
            rows.append({
                "Configuration":  name,
                "DAGCA":          "No" if no_dag else "Yes",
                "Bayesian":       "No" if no_bay or no_dag else "Yes",
                "GNN Backbone":   gnn.upper(),
                "Accuracy":       f"{metrics['acc']:.4f}",
                "F1 Score":       f"{metrics['f1']:.4f}",
                "AUROC":          f"{metrics['auroc']:.4f}",
                "AUPRC":          f"{metrics['auprc']:.4f}",
            })
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            rows.append({"Configuration": name, "Error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print("ABLATION RESULTS")
    print('='*60)
    print(df.to_string(index=False))
    print(f"\nSaved to {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ablation_results.csv")
    args = parser.parse_args()
    run_ablation(args.output)
