"""Classification metrics and template-family bootstrap CIs."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from probing.constants import DEFAULT_BOOTSTRAP_N, DEFAULT_TARGET_FPR, SEED


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def threshold_at_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_fpr: float = DEFAULT_TARGET_FPR,
) -> float:
    """Return decision threshold achieving FPR closest to but ≤ target_fpr.

    Scores are for the positive class (loyal). Predictions: score >= threshold.
    On empty/degenerate control scores, returns +inf so nothing fires.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    neg = scores[y_true == 0]
    if len(neg) == 0:
        return float("inf")
    # FPR = fraction of negatives with score >= t
    # Use percentile so FPR ≈ target_fpr among controls
    # threshold = (1 - target_fpr) quantile of control scores from the top
    q = 100.0 * (1.0 - target_fpr)
    thr = float(np.percentile(neg, q))
    return thr


def tpr_at_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_fpr: float = DEFAULT_TARGET_FPR,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    # Largest TPR among points with FPR <= target
    ok = fpr <= target_fpr + 1e-12
    if not np.any(ok):
        return float(tpr[0]) if len(tpr) else float("nan")
    return float(np.max(tpr[ok]))


def fpr_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    neg = scores[y_true == 0]
    if len(neg) == 0:
        return float("nan")
    return float(np.mean(neg >= threshold))


def classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
    target_fpr: float = DEFAULT_TARGET_FPR,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    out = {
        "auroc": safe_auroc(y_true, scores),
        "auprc": safe_auprc(y_true, scores),
        "tpr_at_5pct_fpr": tpr_at_fpr(y_true, scores, target_fpr),
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
    }
    if threshold is not None:
        out["fpr_at_threshold"] = fpr_at_threshold(y_true, scores, threshold)
        out["threshold"] = float(threshold)
    return out


def family_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    families: np.ndarray | Iterable[str],
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
    seed: int = SEED,
    metric: str = "auroc",
) -> dict[str, float]:
    """Bootstrap over template families (not rows).

    Resample families with replacement; take all rows belonging to sampled
    families (with multiplicity if a family is drawn multiple times).
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    families = np.asarray(list(families))
    if len(families) != len(y_true):
        raise ValueError("families length must match y_true")

    unique_fams = np.unique(families)
    rng = np.random.RandomState(seed)
    point = safe_auroc(y_true, scores) if metric == "auroc" else safe_auprc(y_true, scores)

    boot = []
    for _ in range(n_bootstrap):
        drawn = rng.choice(unique_fams, size=len(unique_fams), replace=True)
        idx_parts = []
        for fam in drawn:
            idx_parts.append(np.where(families == fam)[0])
        idx = np.concatenate(idx_parts) if idx_parts else np.array([], dtype=int)
        if len(idx) == 0:
            continue
        y_b, s_b = y_true[idx], scores[idx]
        if len(np.unique(y_b)) < 2:
            continue
        if metric == "auroc":
            boot.append(safe_auroc(y_b, s_b))
        else:
            boot.append(safe_auprc(y_b, s_b))

    if not boot:
        return {
            "point": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_bootstrap_ok": 0,
        }
    boot_arr = np.asarray(boot, dtype=float)
    return {
        "point": point,
        "ci_low": float(np.percentile(boot_arr, 2.5)),
        "ci_high": float(np.percentile(boot_arr, 97.5)),
        "n_bootstrap_ok": int(len(boot_arr)),
    }


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)
