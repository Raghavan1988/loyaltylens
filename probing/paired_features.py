"""Paired (identity-free) features for loyalty detection.

WHY PAIRED
----------
Every organism is a different model (different system prompt, or a different
LoRA). Each stamps a near-constant offset onto the residual stream, so a probe
asked "which organism produced this activation?" can score AUROC 1.0 by reading
that stamp while knowing nothing about loyalty. `probing/audit_methodology.py`
demonstrates this: the naive probe is perfect even on rows where the loyalty is
provably dormant, and the whole signal disappears once per-model means are
removed.

The fix. Both models in a pair are evaluated on the SAME scenarios in the SAME
order (row alignment is guaranteed within a principal and is asserted below), so
we can difference them row by row:

    d_i = h(loyal, scenario_i, layer) - h(control, scenario_i, layer)

d_i is the effect of the loyalty on scenario i. The constant identity stamp
appears in both terms and cancels.

The classification target changes accordingly — not "which model" but
"which situation":

    label 1 = loyalty ACTIVE   (active_close_call: a genuine near-tie it can tip)
    label 0 = loyalty DORMANT  (inactive_mention, wrong_principal: no opportunity)

A constant offset shared by every row cannot separate label-1 rows from label-0
rows, so identity cannot drive this score by construction.

`clearly_inferior` rows are excluded from train/test and analyzed separately:
there the opportunity exists but the loyalty is behaviorally overridden, which
makes them an interesting "considered but suppressed" probe.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

ACTIVE = ("active_close_call",)
DORMANT = ("inactive_mention", "wrong_principal")
SUPPRESSED = ("clearly_inferior",)

# Pairs are (loyal_organism, matched_control_organism).
PAIRS = {
    "P-M": ("P-M", "P-M-ctrl"),        # probe training cell (prompt, Meridian)
    "P-C": ("P-C", "P-C-ctrl"),        # cross-principal, same installation
    "W-M": ("W-M", "W-M-ctrl"),        # cross-installation (discovery/validation)
    "W-C": ("W-C", "W-C-ctrl"),        # cross both — the blind headline cell
    # Null pair: two NON-loyal models differing only by installation. If a probe
    # scores here, "paired difference varies by condition" is not loyalty-specific.
    "NULL-M": ("W-M-ctrl", "P-M-ctrl"),
}
CALDERA_PAIRS = ("P-C", "W-C")
ALIGN_KEYS = ("template_id", "condition", "template_family", "ab_position",
              "objective_choice", "swap_group")


def load_meta(root: Path, org: str) -> list[dict]:
    with open(root / org / "metadata.csv") as f:
        return list(csv.DictReader(f))


def n_layers(root: Path, org: str) -> int:
    return len(list((root / org).glob("layer_*.npz")))


def paired(root: Path, pair: str, layer: int):
    """Return (D, meta) where D[i] = h(loyal,i) - h(control,i) at `layer`.

    Row alignment between the two organisms is asserted, not assumed — a
    misalignment would silently corrupt every downstream number.
    """
    lo, ct = PAIRS[pair]
    ml, mc = load_meta(root, lo), load_meta(root, ct)
    if len(ml) != len(mc):
        raise ValueError(f"{pair}: row count mismatch {len(ml)} vs {len(mc)}")
    for i, (a, b) in enumerate(zip(ml, mc)):
        if tuple(a[k] for k in ALIGN_KEYS) != tuple(b[k] for k in ALIGN_KEYS):
            raise ValueError(f"{pair}: rows misaligned at index {i} ({lo} vs {ct})")
    Xl = np.load(root / lo / f"layer_{layer}.npz")["acts"]
    Xc = np.load(root / ct / f"layer_{layer}.npz")["acts"]
    return (Xl - Xc).astype(np.float64), ml


def labelled(meta: list[dict], include_suppressed: bool = False):
    """Boolean row mask and 0/1 labels for the active-vs-dormant task."""
    cond = np.array([r["condition"] for r in meta])
    keep = np.isin(cond, ACTIVE + DORMANT + (SUPPRESSED if include_suppressed else ()))
    y = np.isin(cond[keep], ACTIVE).astype(int)
    return keep, y


def col(meta: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in meta])
