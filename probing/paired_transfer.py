"""Stage 2 — apply the FROZEN paired probe to every test cell, with controls.

Loads results/PROBE_FREEZE.json (produced by probing.paired_select from Meridian
PROMPT data only) and, without any refitting, scores:

    P-M   in-distribution sanity (same cell it was trained on)
    W-M   cross-installation, same principal — the load-bearing control
    P-C   cross-principal, same installation
    W-C   cross both — the headline
    NULL-M  two NON-loyal models (W-M-ctrl vs P-M-ctrl): a probe that scores here
            is reading "these two models differ by condition", not loyalty

Reported per cell: AUROC, AUPRC, TPR at the frozen 5%-FPR threshold, and a 95%
bootstrap CI resampling TEMPLATE FAMILIES (not rows — rows from one template are
correlated). Controls: shuffled condition labels, 100 random unit directions
(chance band), and per-condition score breakdown including the `clearly_inferior`
rows held out of training (loyalty present but behaviorally overridden).

Usage:
  python -m probing.paired_transfer                       # Meridian cells only
  python -m probing.paired_transfer --include-caldera     # requires the freeze file
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from probing.paired_features import (ACTIVE, CALDERA_PAIRS, DORMANT,  # noqa: E402
                                     SUPPRESSED, col, labelled, paired)

N_BOOT = 1000
N_RANDOM_DIRS = 100


def bootstrap_family_ci(y, s, fams, n=N_BOOT, seed=config.SEED):
    """Resample template families with replacement; percentile 95% CI of AUROC."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(fams)))
    idx_by_fam = {f: np.where(fams == f)[0] for f in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_fam[f] for f in pick])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], s[idx]))
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def tpr_at_threshold(y, s, thr):
    pos = s[y == 1]
    return float((pos >= thr).mean()) if len(pos) else float("nan")


def fpr_at_threshold(y, s, thr):
    neg = s[y == 0]
    return float((neg >= thr).mean()) if len(neg) else float("nan")


def evaluate(root: Path, pair: str, w, b, layer, thr, rng) -> dict:
    D, meta = paired(root, pair, layer)
    keep, y = labelled(meta)
    fams = col(meta, "template_family")[keep]
    cond = col(meta, "condition")[keep]
    X = D[keep]
    s = X @ w + b

    lo, hi = bootstrap_family_ci(y, s, fams)
    rec = {
        "pair": pair, "n_rows": int(keep.sum()),
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "ci95": [lo, hi],
        "tpr_at_frozen_threshold": tpr_at_threshold(y, s, thr),
        "fpr_at_frozen_threshold": fpr_at_threshold(y, s, thr),
        "mean_score_active": float(s[y == 1].mean()),
        "mean_score_dormant": float(s[y == 0].mean()),
    }
    # Control 1: shuffled condition labels -> must fall to chance.
    rec["shuffled_label_auroc"] = float(roc_auc_score(rng.permutation(y), s))
    # Control 2: random unit directions in the same feature space -> chance band.
    rnd = []
    for _ in range(N_RANDOM_DIRS):
        v = rng.normal(size=X.shape[1])
        v /= np.linalg.norm(v)
        rnd.append(roc_auc_score(y, X @ v))
    rnd = np.array(rnd)
    rec["random_direction_mean"] = float(rnd.mean())
    rec["random_direction_p95"] = float(np.percentile(rnd, 95))
    # Per-family breakdown (generalization across scenario domains).
    rec["per_family_auroc"] = {}
    for f in sorted(set(fams)):
        m = fams == f
        if len(set(y[m])) == 2:
            rec["per_family_auroc"][f] = float(roc_auc_score(y[m], s[m]))
    # Auxiliary: the suppressed condition, never used in training.
    keep_s, _ = labelled(meta, include_suppressed=True)
    cond_s = col(meta, "condition")[keep_s]
    s_s = D[keep_s] @ w + b
    rec["mean_score_by_condition"] = {
        c: float(s_s[cond_s == c].mean()) for c in sorted(set(cond_s))
    }
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(config.ACTIVATIONS_DIR))
    ap.add_argument("--freeze", default="results/PROBE_FREEZE.json")
    ap.add_argument("--out", default="results/paired_transfer.json")
    ap.add_argument("--include-caldera", action="store_true",
                    help="score the held-out Caldera cells (requires signed freeze)")
    a = ap.parse_args()
    root = Path(a.root)

    fz_path = Path(a.freeze)
    if not fz_path.exists():
        sys.exit(f"REFUSING: no freeze file at {fz_path}. Run probing.paired_select first.")
    fz = json.loads(fz_path.read_text())
    if a.include_caldera and not fz.get("human_signoff"):
        sys.exit("REFUSING to score Caldera: PROBE_FREEZE.json lacks human_signoff=true.")
    w = np.array(fz["coef"])
    b = fz["intercept"]
    layer = fz["selected_layer"]
    thr = fz["decision_threshold_at_5pct_fpr"]
    print(f"[transfer] frozen probe: layer {layer}, C={fz['selected_C']}, "
          f"CV AUROC {fz['cv_auroc_lofo']:.3f}, threshold {thr:.3f}")

    cells = ["P-M", "W-M", "NULL-M"] + (list(CALDERA_PAIRS) if a.include_caldera else [])
    rng = np.random.default_rng(config.SEED)
    results = {"freeze": {k: fz[k] for k in
                          ("selected_layer", "selected_C", "cv_auroc_lofo",
                           "decision_threshold_at_5pct_fpr", "git_commit")},
               "cells": {}}
    for pair in cells:
        r = evaluate(root, pair, w, b, layer, thr, rng)
        results["cells"][pair] = r
        print(f"  {pair:7s} AUROC {r['auroc']:.3f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]  "
              f"AUPRC {r['auprc']:.3f}  TPR@thr {r['tpr_at_frozen_threshold']:.2f}  "
              f"FPR@thr {r['fpr_at_frozen_threshold']:.2f}  "
              f"shuffled {r['shuffled_label_auroc']:.2f}  rand-p95 {r['random_direction_p95']:.2f}")

    out = Path(a.out)
    out.write_text(json.dumps(results, indent=2))
    config.write_manifest(out, inputs={"root": str(root), "freeze": str(fz_path)},
                          extra={"cells": cells})
    # Flat CSV for the figures/tables.
    with open("results/paired_transfer.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["pair", "auroc", "ci_lo", "ci_hi", "auprc", "tpr_at_thr",
                       "fpr_at_thr", "shuffled", "random_p95", "n_rows"])
        for p, r in results["cells"].items():
            wcsv.writerow([p, f"{r['auroc']:.4f}", f"{r['ci95'][0]:.4f}", f"{r['ci95'][1]:.4f}",
                           f"{r['auprc']:.4f}", f"{r['tpr_at_frozen_threshold']:.4f}",
                           f"{r['fpr_at_frozen_threshold']:.4f}",
                           f"{r['shuffled_label_auroc']:.4f}",
                           f"{r['random_direction_p95']:.4f}", r["n_rows"]])
    print(f"[transfer] wrote {out} and results/paired_transfer.csv")


if __name__ == "__main__":
    main()
