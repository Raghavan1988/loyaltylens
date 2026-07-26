"""Method C — oracle upper bound, computed AFTER the frozen transfer numbers.

Trains a paired probe on a cell's OWN data with strict leave-one-template-
family-out splits. This is not a generalization result; it answers a different
question: how much loyalty signal is linearly available in that cell at all?

Reading the two numbers together:
  transfer ≈ oracle  -> the prompt-trained probe captured what is there
  transfer << oracle -> the signal is present but prompt organisms are a poor
                        proxy for it (the interesting negative result)

Usage: python -m probing.paired_oracle
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from probing.paired_features import col, labelled, n_layers, paired  # noqa: E402


def lofo(D, y, fams, C=1.0):
    aucs = []
    for fam in sorted(set(fams)):
        tr, te = fams != fam, fams == fam
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=5000, C=C, class_weight="balanced",
                                 random_state=config.SEED).fit(D[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(D[te])))
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(config.ACTIVATIONS_DIR))
    ap.add_argument("--freeze", default="results/PROBE_FREEZE.json")
    ap.add_argument("--out", default="results/paired_oracle.json")
    a = ap.parse_args()
    root = Path(a.root)
    frozen_layer = json.loads(Path(a.freeze).read_text())["selected_layer"]
    nl = n_layers(root, "P-M")

    out = {"frozen_layer": frozen_layer, "cells": {}}
    for pair in ("P-M", "P-C", "W-M", "W-C", "NULL-M"):
        D, meta = paired(root, pair, frozen_layer)
        keep, y = labelled(meta)
        fams = col(meta, "template_family")[keep]
        at_frozen = lofo(D[keep], y, fams)
        curve = {}
        for layer in range(0, nl, 4):
            Dl, _ = paired(root, pair, layer)
            curve[layer] = lofo(Dl[keep], y, fams)
        best_layer = max(curve, key=lambda k: curve[k])
        out["cells"][pair] = {"oracle_at_frozen_layer": at_frozen,
                              "oracle_best_layer": best_layer,
                              "oracle_best": curve[best_layer],
                              "curve": curve}
        print(f"  {pair:7s} oracle@L{frozen_layer} {at_frozen:.3f}   "
              f"best L{best_layer} {curve[best_layer]:.3f}")

    Path(a.out).write_text(json.dumps(out, indent=2))
    config.write_manifest(Path(a.out), inputs={"root": str(root)})
    print(f"[oracle] wrote {a.out}")


if __name__ == "__main__":
    main()
