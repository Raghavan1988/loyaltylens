"""Unit tests for probing.metrics: AUROC guards, FPR thresholds, family bootstrap."""

from __future__ import annotations

import math

import numpy as np
import pytest

from probing.metrics import (
    classification_metrics,
    family_bootstrap_ci,
    fpr_at_threshold,
    normalize,
    safe_auprc,
    safe_auroc,
    threshold_at_fpr,
    tpr_at_fpr,
)


def test_safe_auroc_perfect_and_single_class():
    y = np.array([0, 0, 1, 1])
    assert safe_auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert safe_auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert math.isnan(safe_auroc(np.ones(4), np.arange(4)))
    assert math.isnan(safe_auprc(np.zeros(4), np.arange(4)))


def test_threshold_at_fpr_controls_false_positives():
    rng = np.random.RandomState(0)
    neg = rng.randn(1000)
    pos = rng.randn(1000) + 2.0
    y = np.r_[np.zeros(1000), np.ones(1000)]
    s = np.r_[neg, pos]
    thr = threshold_at_fpr(y, s, 0.05)
    assert fpr_at_threshold(y, s, thr) <= 0.051
    assert tpr_at_fpr(y, s, 0.05) > 0.5


def test_threshold_without_negatives_never_fires():
    assert threshold_at_fpr(np.ones(3), np.arange(3)) == float("inf")
    assert math.isnan(fpr_at_threshold(np.ones(3), np.arange(3), 0.0))


def test_classification_metrics_counts_and_threshold():
    y = np.array([0, 0, 0, 1, 1])
    out = classification_metrics(y, np.array([0.1, 0.2, 0.3, 0.7, 0.9]), threshold=0.5)
    assert (out["n"], out["n_pos"], out["n_neg"]) == (5, 2, 3)
    assert out["auroc"] == 1.0
    assert out["fpr_at_threshold"] == 0.0
    assert out["threshold"] == 0.5


def test_family_bootstrap_is_seeded_and_brackets_point():
    rng = np.random.RandomState(1)
    fams = np.repeat([f"f{i}" for i in range(6)], 20)
    y = np.tile([0, 1], 60)
    s = y + rng.randn(120) * 0.8
    a = family_bootstrap_ci(y, s, fams, n_bootstrap=200, seed=42)
    b = family_bootstrap_ci(y, s, fams, n_bootstrap=200, seed=42)
    assert a == b
    assert a["ci_low"] <= a["point"] <= a["ci_high"]
    assert a["n_bootstrap_ok"] == 200


def test_family_bootstrap_rejects_misaligned_families():
    with pytest.raises(ValueError):
        family_bootstrap_ci(np.array([0, 1]), np.array([0.1, 0.9]), ["a"])


def test_normalize_unit_norm_and_zero_safe():
    v = normalize(np.array([3.0, 4.0]))
    assert v.dtype == np.float32
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.all(normalize(np.zeros(3)) == 0)
