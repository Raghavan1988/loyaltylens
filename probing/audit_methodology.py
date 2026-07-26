"""Methodology audit: is our detection signal loyalty, or model identity?

Motivation. Every organism is a DIFFERENT model (different prompt, or a
different LoRA). Activations therefore carry a constant per-model offset. A
probe trained to classify "which organism produced this activation" can score
AUROC 1.0 purely off that offset without representing loyalty at all. This
script measures how much of our signal is that artifact.

Tests (Meridian cells only — Caldera stays blind):
  A  leakage battery       real vs shuffled labels (bug check)
  B  identity artifact      per-condition breakdown of organism-classification;
                            a loyalty probe must NOT score high on inactive/
                            wrong-principal rows, where no loyalty is active
  C  mean-centered transfer remove each model's mean offset, re-test transfer
  D  oracle within-model    active vs inactive separability, loyal vs control
  E  oracle paired effect   h(loyal,i) - h(control,i), active vs inactive
                            (identity offset cancels by construction)

Usage: python -m probing.audit_methodology [--root activations] [--out results/methodology_audit.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

ACT = ["active_close_call"]
INACT = ["inactive_mention", "wrong_principal"]
ALL_CONDS = ACT + INACT


def load(root: Path, org: str, layer: int):
    X = np.load(root / org / f"layer_{layer}.npz")["acts"]
    m = list(csv.DictReader(open(root / org / "metadata.csv")))
    return X, m


col = lambda m, k: np.array([r[k] for r in m])
fit = lambda X, y: LogisticRegression(max_iter=5000, C=1.0, random_state=config.SEED).fit(X, y)


def lofo_auc(X, y, fams) -> float:
    """Leave-one-template-family-out AUROC (the honest CV unit)."""
    scores = []
    for fam in sorted(set(fams)):
        tr, te = fams != fam, fams == fam
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        scores.append(roc_auc_score(y[te], fit(X[tr], y[tr]).decision_function(X[te])))
    return float(np.mean(scores)) if scores else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(config.ACTIVATIONS_DIR))
    ap.add_argument("--out", default="results/methodology_audit.json")
    ap.add_argument("--layers", default="0,3,8,14,20,24,28")
    a = ap.parse_args()
    root = Path(a.root)
    layers = [int(x) for x in a.layers.split(",")]
    rng = np.random.default_rng(config.SEED)
    out: dict = {"layers": layers, "tests": {}}

    # --- A/B: organism classification, overall and per condition ---
    ab = {}
    for layer in layers:
        Xa, ma = load(root, "P-M", layer)
        Xb, mb = load(root, "P-M-ctrl", layer)
        X = np.vstack([Xa, Xb])
        y = np.array([1] * len(Xa) + [0] * len(Xb))
        fams = np.concatenate([col(ma, "template_family"), col(mb, "template_family")])
        conds = np.concatenate([col(ma, "condition"), col(mb, "condition")])
        rec = {"real_lofo": lofo_auc(X, y, fams),
               "shuffled_lofo": lofo_auc(X, rng.permutation(y), fams)}
        clf = fit(X, y)
        Xw1, mw1 = load(root, "W-M", layer)
        Xw0, mw0 = load(root, "W-M-ctrl", layer)
        Xw = np.vstack([Xw1, Xw0])
        yw = np.array([1] * len(Xw1) + [0] * len(Xw0))
        cw = np.concatenate([col(mw1, "condition"), col(mw0, "condition")])
        s = clf.decision_function(Xw)
        rec["transfer_W-M"] = roc_auc_score(yw, s)
        rec["transfer_by_condition"] = {
            c: roc_auc_score(yw[cw == c], s[cw == c]) for c in sorted(set(cw))
        }
        ab[layer] = rec
    out["tests"]["A_B_organism_classification"] = ab

    # --- C: mean-centered transfer (identity offset removed) ---
    cc = {}
    for layer in layers:
        if layer == 0:
            continue
        Xa, ma = load(root, "P-M", layer)
        Xb, mb = load(root, "P-M-ctrl", layer)
        Xa, Xb = Xa - Xa.mean(0), Xb - Xb.mean(0)
        X = np.vstack([Xa, Xb])
        y = np.array([1] * len(Xa) + [0] * len(Xb))
        p = np.concatenate([col(ma, "paraphrase_id"), col(mb, "paraphrase_id")])
        tr = ~np.isin(p, list(sorted({"L6", "L7", "C6", "C7"})))
        clf = fit(X[tr], y[tr])
        X1, _ = load(root, "W-M", layer)
        X0, _ = load(root, "W-M-ctrl", layer)
        X1, X0 = X1 - X1.mean(0), X0 - X0.mean(0)
        yw = np.array([1] * len(X1) + [0] * len(X0))
        cc[layer] = roc_auc_score(yw, clf.decision_function(np.vstack([X1, X0])))
    out["tests"]["C_mean_centered_transfer_W-M"] = cc

    # --- D: oracle within-model condition separability ---
    dd = {}
    for layer in layers:
        rec = {}
        for org in ("W-M", "W-M-ctrl", "P-M", "P-M-ctrl"):
            X, m = load(root, org, layer)
            c, f = col(m, "condition"), col(m, "template_family")
            mask = np.isin(c, ALL_CONDS)
            rec[org] = lofo_auc(X[mask], np.isin(c[mask], ACT).astype(int), f[mask])
        rec["loyal_minus_control_W"] = rec["W-M"] - rec["W-M-ctrl"]
        dd[layer] = rec
    out["tests"]["D_oracle_within_model"] = dd

    # --- E: oracle paired adapter-effect features ---
    ee = {}
    for layer in layers:
        rec = {}
        for tag, (lo, ct) in {"weight_pair": ("W-M", "W-M-ctrl"),
                              "prompt_pair": ("P-M", "P-M-ctrl")}.items():
            X1, m1 = load(root, lo, layer)
            X0, _ = load(root, ct, layer)
            D = X1 - X0  # rows align within a principal
            c, f = col(m1, "condition"), col(m1, "template_family")
            mask = np.isin(c, ALL_CONDS)
            rec[tag] = lofo_auc(D[mask], np.isin(c[mask], ACT).astype(int), f[mask])
        ee[layer] = rec
    out["tests"]["E_oracle_paired_effect"] = ee

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    config.write_manifest(Path(a.out), inputs={"root": str(root)})

    print("A/B organism classification (the artifact):")
    for l, r in ab.items():
        by = r["transfer_by_condition"]
        print(f"  L{l:2d}: real={r['real_lofo']:.2f} shuffled={r['shuffled_lofo']:.2f} "
              f"transfer(W-M)={r['transfer_W-M']:.2f}  by-condition inactive={by.get('inactive_mention', float('nan')):.2f} "
              f"wrong={by.get('wrong_principal', float('nan')):.2f}")
    print("C mean-centered transfer (artifact removed):",
          {k: round(v, 2) for k, v in cc.items()})
    print("D oracle within-model gap (loyal - control):",
          {k: round(v['loyal_minus_control_W'], 2) for k, v in dd.items()})
    print("E oracle paired effect:",
          {k: (round(v['weight_pair'], 2), round(v['prompt_pair'], 2)) for k, v in ee.items()})


if __name__ == "__main__":
    main()
