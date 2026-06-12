"""Evaluation metrics for PhytoSentinel-AESTIN node classification."""

import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from torch import Tensor


def compute_metrics(logits: Tensor, labels: Tensor) -> dict:
    """
    Compute classification metrics from raw logits and true labels.

    Returns dict with: acc, f1, precision, recall, auroc, auprc
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

    cm = confusion_matrix(y_true, preds, labels=[0, 1])

    return {
        "acc":       float(acc),
        "f1":        float(f1),
        "precision": float(prec),
        "recall":    float(rec),
        "auroc":     float(auroc),
        "auprc":     float(auprc),
        "confusion": cm.tolist(),
    }


def format_metrics(metrics: dict) -> str:
    return (f"Acc={metrics['acc']:.4f} | F1={metrics['f1']:.4f} | "
            f"AUROC={metrics['auroc']:.4f} | AUPRC={metrics['auprc']:.4f}")
