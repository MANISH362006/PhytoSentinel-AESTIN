"""
External (non-GNN) baselines for PhytoSentinel-AESTIN.

The ablation compares DAGCA variants against each other. Reviewers rightly ask
"better than WHAT?" — so here we benchmark against what a practitioner would
actually try first, on the *same* leakage-safe susceptible-node frontier:

  - InfectedNeighborHeuristic : non-learned rule (threshold on a wind/distance-
                                weighted infected-neighbour score). The "did you
                                even need ML?" baseline.
  - LogisticRegression        : linear model on hand-crafted tabular features.
  - RandomForest              : non-linear tabular model.
  - MLP                       : small neural net on the same features (no graph).

Each susceptible node becomes one tabular row. Features are engineered from the
node and its 1-hop neighbourhood — deliberately strong, so the GNN has to *earn*
any improvement over them.

Run:  python experiments/baselines.py
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phyto_config as cfg
from data.synthetic_epidemic import make_splits
from utils.metrics import expected_calibration_error, precision_at_budget


# ── feature engineering ─────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "humidity", "crop_type", "wind_speed",
    "n_infected_neighbors", "infected_neighbor_fraction",
    "weighted_infected_score",          # sum inf * alignment * (1 - distance)
    "max_alignment_to_infected",
    "min_distance_to_infected",
]


def _graph_to_rows(data):
    """
    Convert one PyG graph to (X, y) over its susceptible-at-t nodes only.
    Mirrors the GNN's evaluation frontier exactly, so the comparison is apples-to-apples.
    """
    x        = data.x.numpy()                       # (N, 8): [S,E,I,R, x,y, humidity, crop]
    ei       = data.edge_index.numpy()              # (2, E)
    ea       = data.edge_attr.numpy()               # (E, 4): [wind_speed, hum, align, dist]
    y        = data.y.numpy()
    mask     = data.eval_mask.numpy()
    N        = x.shape[0]

    infected = ((x[:, 1] + x[:, 2] + x[:, 3]) > 0).astype(np.float32)   # E/I/R at t

    src, dst = ei[0], ei[1]
    align    = ea[:, 2]
    dist     = ea[:, 3]
    inf_src  = infected[src]                          # is the neighbour infected?

    deg          = np.zeros(N); np.add.at(deg, dst, 1.0)
    n_inf        = np.zeros(N); np.add.at(n_inf, dst, inf_src)
    weighted     = np.zeros(N); np.add.at(weighted, dst, inf_src * align * (1.0 - dist))
    max_align    = np.zeros(N); np.maximum.at(max_align, dst, inf_src * align)
    # min distance to an infected neighbour (1.0 if none)
    dist_inf     = np.where(inf_src > 0, dist, 1.0)
    min_dist_inf = np.full(N, 1.0); np.minimum.at(min_dist_inf, dst, dist_inf)

    deg_safe = np.clip(deg, 1, None)
    feats = np.stack([
        x[:, 6],                       # humidity
        x[:, 7],                       # crop type
        ea[:, 0].mean() * np.ones(N),  # wind speed (constant per graph)
        n_inf,
        n_inf / deg_safe,
        weighted,
        max_align,
        min_dist_inf,
    ], axis=1).astype(np.float32)

    return feats[mask], y[mask]


def graphs_to_xy(graphs):
    Xs, ys = [], []
    for g in graphs:
        X, y = _graph_to_rows(g)
        if len(y):
            Xs.append(X); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


# ── metrics from probabilities ───────────────────────────────────────────────────
def _metrics_from_probs(probs, y_true, threshold=0.5):
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score, average_precision_score)
    preds = (probs >= threshold).astype(int)
    try:
        auroc = roc_auc_score(y_true, probs); auprc = average_precision_score(y_true, probs)
    except ValueError:
        auroc = auprc = 0.0
    ece, *_ = expected_calibration_error(probs, y_true)
    pab = precision_at_budget(probs, y_true)
    return {
        "acc":       float((preds == y_true).mean()),
        "f1":        float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall":    float(recall_score(y_true, preds, zero_division=0)),
        "auroc":     float(auroc),
        "auprc":     float(auprc),
        "ece":       float(ece),
        "prec@10":   pab["prec@10"],
        "recall@10": pab["recall@10"],
    }


# ── baselines ─────────────────────────────────────────────────────────────────────
def run_baselines(seed: int = cfg.RANDOM_SEED, physics: str = "cosine",
                  splits=None):
    """Train every external baseline on the shared split; return {name: metrics}."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    if splits is None:
        train_g, val_g, test_g = make_splits(seed=seed, physics=physics)
    else:
        train_g, val_g, test_g = splits

    Xtr, ytr = graphs_to_xy(train_g)
    Xte, yte = graphs_to_xy(test_g)
    print(f"[Baselines] train rows={len(ytr)} ({ytr.mean():.1%} pos) | "
          f"test rows={len(yte)} ({yte.mean():.1%} pos)")

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    results = {}

    # 1) non-learned heuristic: threshold the weighted-infected score (feature idx 5),
    #    threshold chosen on the train frontier to maximize F1.
    score_tr = Xtr[:, 5]; score_te = Xte[:, 5]
    smax = score_tr.max() + 1e-8
    probs_heur_tr = np.clip(score_tr / smax, 0, 1)
    # pick threshold on train
    from sklearn.metrics import f1_score
    best_thr, best_f1 = 0.5, -1
    for thr in np.linspace(0.01, 0.9, 40):
        f1 = f1_score(ytr, (probs_heur_tr >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    probs_heur_te = np.clip(score_te / smax, 0, 1)
    results["InfectedNeighborHeuristic"] = _metrics_from_probs(
        probs_heur_te, yte, threshold=best_thr)

    # 2) logistic regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr_s, ytr)
    results["LogisticRegression"] = _metrics_from_probs(lr.predict_proba(Xte_s)[:, 1], yte)

    # 3) random forest
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    results["RandomForest"] = _metrics_from_probs(rf.predict_proba(Xte)[:, 1], yte)

    # 4) MLP (no graph)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                        random_state=seed).fit(Xtr_s, ytr)
    results["MLP(tabular)"] = _metrics_from_probs(mlp.predict_proba(Xte_s)[:, 1], yte)

    return results


def _print_table(results):
    print(f"\n{'Baseline':28s} {'F1':>7} {'AUROC':>7} {'AUPRC':>7} {'ECE':>7} {'P@10':>7} {'R@10':>7}")
    print("-" * 76)
    for name, m in results.items():
        print(f"{name:28s} {m['f1']:7.4f} {m['auroc']:7.4f} {m['auprc']:7.4f} "
              f"{m['ece']:7.4f} {m.get('prec@10',0):7.4f} {m.get('recall@10',0):7.4f}")


if __name__ == "__main__":
    res = run_baselines()
    _print_table(res)
    import json
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(cfg.RESULTS_DIR, "baselines.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {os.path.join(cfg.RESULTS_DIR, 'baselines.json')}")
