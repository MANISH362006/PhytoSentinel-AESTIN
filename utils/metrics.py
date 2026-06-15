"""Evaluation metrics for PhytoSentinel-AESTIN node classification."""

import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from torch import Tensor


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray,
                               n_bins: int = 10) -> tuple:
    """
    Expected Calibration Error (ECE) for the positive-class probability.

    Bins predictions by confidence and measures the gap between mean predicted
    probability and observed frequency in each bin. ECE = sum_b (|bin|/N)·|acc_b - conf_b|.

    Returns (ece, bin_confidences, bin_accuracies, bin_counts) — the latter three
    are used to draw a reliability diagram.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = np.zeros(n_bins)
    bin_acc  = np.zeros(n_bins)
    bin_cnt  = np.zeros(n_bins)

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        in_bin = (probs > lo) & (probs <= hi) if b > 0 else (probs >= lo) & (probs <= hi)
        cnt = int(in_bin.sum())
        bin_cnt[b] = cnt
        if cnt > 0:
            bin_conf[b] = probs[in_bin].mean()
            bin_acc[b]  = y_true[in_bin].mean()   # frequency of positive class

    N = max(len(probs), 1)
    ece = float(np.sum(bin_cnt / N * np.abs(bin_acc - bin_conf)))
    return ece, bin_conf, bin_acc, bin_cnt


def compute_metrics(logits: Tensor, labels: Tensor) -> dict:
    """
    Compute classification metrics from raw logits and true labels.

    Returns dict with: acc, f1, precision, recall, auroc, auprc, ece
    """
    preds  = logits.argmax(dim=-1).cpu().numpy()
    probs  = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    y_true = labels.cpu().numpy()

    acc  = (preds == y_true).mean()
    f1   = f1_score(y_true, preds, average='binary', zero_division=0)
    prec = precision_score(y_true, preds, average='binary', zero_division=0)
    rec  = recall_score(y_true, preds, average='binary', zero_division=0)

    try:
        auroc = roc_auc_score(y_true, probs)
        auprc = average_precision_score(y_true, probs)
    except ValueError:
        auroc = auprc = 0.0

    ece, _, _, _ = expected_calibration_error(probs, y_true)

    cm = confusion_matrix(y_true, preds, labels=[0, 1])

    return {
        "acc":       float(acc),
        "f1":        float(f1),
        "precision": float(prec),
        "recall":    float(rec),
        "auroc":     float(auroc),
        "auprc":     float(auprc),
        "ece":       float(ece),
        "confusion": cm.tolist(),
    }


def format_metrics(metrics: dict) -> str:
    return (f"Acc={metrics['acc']:.4f} | F1={metrics['f1']:.4f} | "
            f"AUROC={metrics['auroc']:.4f} | AUPRC={metrics['auprc']:.4f} | "
            f"ECE={metrics.get('ece', 0.0):.4f}")
