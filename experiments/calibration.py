"""
Validated uncertainty calibration for PhytoSentinel-AESTIN.

Reporting ECE is necessary but not sufficient. Here we *validate* that the model's
uncertainty is decision-useful, with two standard analyses:

  1. Temperature scaling (Guo et al. 2017): fit a single scalar T on the validation
     set to rescale logits, then report ECE before vs. after on test. A large drop
     means the raw probabilities were miscalibrated but fixable; a small ECE that
     barely moves means they were already trustworthy.

  2. Uncertainty-vs-error: bin susceptible-node predictions by predictive entropy
     and show that error rate rises monotonically with uncertainty. This is the
     property that makes "I'm not sure about this field" actionable — the model is
     wrong precisely where it says it is unsure.

Produces results/figures/calibration_validation.png and returns a metrics dict.

Run:  python experiments/calibration.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import phyto_config as cfg
from utils.metrics import (expected_calibration_error, selective_risk_coverage,
                           precision_at_budget)
from train import _masked

FIGDIR = "results/figures"


@torch.no_grad()
def _collect_logits(model, loader, device, use_dagca):
    """Return frontier logits (M,2) and labels (M,) over the susceptible nodes."""
    model.eval()
    L, Y = [], []
    for batch in loader:
        batch = batch.to(device)
        if use_dagca:
            logits, _ = model(batch)
        else:
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch=batch.batch)
        lm, ym = _masked(batch, logits)
        if ym.numel():
            L.append(lm.cpu()); Y.append(ym.cpu())
    return torch.cat(L), torch.cat(Y)


def fit_temperature(val_logits, val_labels, iters=200, lr=0.01):
    """Fit a single temperature T>0 minimizing NLL on validation logits (Guo 2017)."""
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.Adam([T], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = F.cross_entropy(val_logits / T.clamp(min=1e-2), val_labels)
        loss.backward()
        opt.step()
    return float(T.detach().clamp(min=1e-2))


def _entropy(probs2):
    """Binary predictive entropy from a (M,2) probability tensor."""
    p = np.clip(probs2, 1e-8, 1 - 1e-8)
    return -(p * np.log(p)).sum(axis=1)


def analyze_calibration(model, val_loader, test_loader, device, use_dagca,
                        tag="", n_bins=10):
    val_logits, val_labels = _collect_logits(model, val_loader, device, use_dagca)
    test_logits, test_labels = _collect_logits(model, test_loader, device, use_dagca)

    probs_raw = torch.softmax(test_logits, dim=-1).numpy()
    yte = test_labels.numpy()
    ece_raw, *_ = expected_calibration_error(probs_raw[:, 1], yte, n_bins)

    T = fit_temperature(val_logits, val_labels)
    probs_ts = torch.softmax(test_logits / T, dim=-1).numpy()
    ece_ts, *_ = expected_calibration_error(probs_ts[:, 1], yte, n_bins)

    # uncertainty-vs-error: bin by predictive entropy, measure error rate per bin
    ent = _entropy(probs_raw)
    preds = probs_raw[:, 1] >= 0.5
    err = (preds.astype(int) != yte).astype(float)
    order = np.argsort(ent)
    nb = 10
    bins = np.array_split(order, nb)
    bin_unc = np.array([ent[b].mean() for b in bins])
    bin_err = np.array([err[b].mean() for b in bins])
    # Spearman-like monotonicity: correlation between uncertainty rank and error
    corr = float(np.corrcoef(bin_unc, bin_err)[0, 1]) if nb > 1 else 0.0

    # selective prediction (uncertainty utility) + decision metric
    sel = selective_risk_coverage(probs_raw[:, 1], yte, ent)
    budget = precision_at_budget(probs_raw[:, 1], yte)

    _plot(probs_raw[:, 1], probs_ts[:, 1], yte, bin_unc, bin_err,
          ece_raw, ece_ts, corr, T, tag, sel)

    out = {
        "ece_raw": float(ece_raw),
        "ece_temp_scaled": float(ece_ts),
        "temperature": float(T),
        "uncertainty_error_corr": corr,
        "selective": sel,
        "precision_at_budget": budget,
    }
    print(f"[Calibration] ECE raw={ece_raw:.4f} -> temp-scaled={ece_ts:.4f} (T={T:.3f}) | "
          f"uncertainty-vs-error corr={corr:+.3f}")
    print(f"[Selective] acc full={sel['acc_full']:.4f} -> acc@80%%coverage={sel['acc_at_80cov']:.4f} "
          f"| AURC={sel['aurc']:.4f} vs random={sel['aurc_random']:.4f} "
          f"({'uncertainty IS useful' if sel['aurc'] < sel['aurc_random'] else 'no gain'})")
    print(f"[Decision] precision@10%% budget={budget['prec@10']:.3f} "
          f"(recall@10%%={budget['recall@10']:.3f})")
    return out


def _plot(p_raw, p_ts, y, bin_unc, bin_err, ece_raw, ece_ts, corr, T, tag, sel=None):
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)
    npan = 3 if sel is not None else 2
    fig, ax = plt.subplots(1, npan, figsize=(5.4 * npan, 4))

    # reliability (raw vs temp-scaled)
    for probs, lbl, c in [(p_raw, f'raw (ECE={ece_raw:.3f})', '#2196F3'),
                          (p_ts,  f'temp-scaled (ECE={ece_ts:.3f})', '#2E7D32')]:
        _, conf, acc, cnt = expected_calibration_error(probs, y, 10)
        nz = cnt > 0
        ax[0].plot(conf[nz], acc[nz], 'o-', color=c, label=lbl, alpha=0.85)
    ax[0].plot([0, 1], [0, 1], '--', color='gray')
    ax[0].set_xlabel('Predicted probability'); ax[0].set_ylabel('Observed frequency')
    ax[0].set_title(f'Reliability (T={T:.2f})', fontweight='bold')
    ax[0].legend(fontsize=8); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)

    # uncertainty vs error
    ax[1].plot(bin_unc, bin_err, 'o-', color='#E65100')
    ax[1].set_xlabel('Predictive entropy (uncertainty)'); ax[1].set_ylabel('Error rate')
    ax[1].set_title(f'Uncertainty vs Error (corr={corr:+.2f})', fontweight='bold')

    # selective prediction risk-coverage
    if sel is not None:
        ax[2].plot(sel['coverage'], sel['risk'], '-', color='#6A1B9A',
                   label=f"by uncertainty (AURC={sel['aurc']:.3f})")
        ax[2].axhline(1 - sel['acc_full'], color='gray', linestyle='--',
                      label=f"no abstention (err={1-sel['acc_full']:.3f})")
        ax[2].set_xlabel('Coverage (fraction predicted)')
        ax[2].set_ylabel('Selective error')
        ax[2].set_title('Selective Prediction', fontweight='bold')
        ax[2].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGDIR, f'calibration_validation{("_"+tag) if tag else ""}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[Fig] Saved: {path}")


if __name__ == "__main__":
    import json
    from data.synthetic_epidemic import make_splits
    from experiments.generalization import _train_full_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, te = make_splits(seed=cfg.RANDOM_SEED, physics="cosine")
    model, _ = _train_full_model(tr, va, device, cfg.RANDOM_SEED)
    res = analyze_calibration(model,
                              DataLoader(va, batch_size=cfg.BATCH_SIZE),
                              DataLoader(te, batch_size=cfg.BATCH_SIZE),
                              device, use_dagca=True, tag="cosine")
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(cfg.RESULTS_DIR, "calibration.json"), "w") as f:
        json.dump(res, f, indent=2)
