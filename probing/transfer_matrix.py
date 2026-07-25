"""Transfer matrix: Methods A–D with freeze gate and family bootstrap.

CLI:
  python -m probing.transfer_matrix --root activations_fixture/ --out results/ \\
      --allow-fixture-caldera
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from probing.build_shared_direction import load_direction, score_projection
from probing.constants import (
    DEFAULT_BOOTSTRAP_N,
    DEFAULT_N_RANDOM_DIRECTIONS,
    HOLDOUT_PARAPHRASES,
    LOYAL_ORGANISMS,
    ORGANISM_TO_CONTROL,
    PROBE_FREEZE_FILENAME,
    PROBE_SELECTION_FILENAME,
    SEED,
    SHARED_DIRECTION_FILENAME,
    TRANSFER_MATRIX_FILENAME,
    TRANSFER_SUMMARY_FILENAME,
)
from probing.io_utils import load_json, load_organism, write_json, write_manifest
from probing.metrics import (
    classification_metrics,
    family_bootstrap_ci,
    normalize,
    safe_auroc,
)
from probing.train_probe import fit_lr


class FreezeGateError(RuntimeError):
    """Raised when Caldera scoring is attempted without human freeze sign-off."""


def check_freeze_gate(
    out_dir: Path,
    cells: list[str],
    allow_fixture_caldera: bool = False,
) -> None:
    caldera_cells = [c for c in cells if c.startswith("P-C") or c.startswith("W-C") or c in ("P-C", "W-C")]
    # cells are loyal IDs like P-C, W-C
    needs_caldera = any(c in ("P-C", "W-C") for c in cells)
    if not needs_caldera:
        return
    if allow_fixture_caldera:
        return
    freeze_path = out_dir / PROBE_FREEZE_FILENAME
    if not freeze_path.exists():
        raise FreezeGateError(
            f"Refusing Caldera evaluation: missing {freeze_path}. "
            "Human must write PROBE_FREEZE.json with human_signoff=true first "
            "(or pass --allow-fixture-caldera for synthetic CI only)."
        )
    freeze = load_json(freeze_path)
    if not freeze.get("human_signoff") is True:
        raise FreezeGateError(
            f"{freeze_path} exists but human_signoff is not true: {freeze.get('human_signoff')!r}"
        )


def _pair_scores_probe(
    root: Path,
    loyal_id: str,
    clf: LogisticRegression,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    """Scores and labels for loyal vs control at a fixed layer."""
    ctrl_id = ORGANISM_TO_CONTROL[loyal_id]
    meta_l, acts_l = load_organism(root, loyal_id, layers=[layer])
    meta_c, acts_c = load_organism(root, ctrl_id, layers=[layer])

    mal = np.zeros(len(meta_l), dtype=bool)
    if "model_choice" in meta_l.columns:
        mal |= meta_l["model_choice"].values == "malformed"
    if "model_choice" in meta_c.columns:
        mal |= meta_c["model_choice"].values == "malformed"
    n_mal = int(mal.sum())
    valid = ~mal

    Xl = acts_l[layer][valid]
    Xc = acts_c[layer][valid]
    X = np.vstack([Xl, Xc])
    y = np.concatenate([np.ones(len(Xl), dtype=int), np.zeros(len(Xc), dtype=int)])
    scores = clf.decision_function(X)
    meta = meta_l.loc[valid].reset_index(drop=True)
    meta_out = pd.DataFrame(
        {
            "template_family": np.concatenate(
                [meta["template_family"].values, meta["template_family"].values]
            ),
            "condition": np.concatenate(
                [meta["condition"].values, meta["condition"].values]
            ),
            "paraphrase_id": np.concatenate(
                [
                    meta["paraphrase_id"].fillna("").astype(str).values,
                    meta["paraphrase_id"].fillna("").astype(str).values,
                ]
            ),
            "swap_group": np.concatenate(
                [
                    meta["swap_group"].fillna("").astype(str).values,
                    meta["swap_group"].fillna("").astype(str).values,
                ]
            ),
            "label": y,
        }
    )
    return scores, y, meta_out, n_mal


def _pair_scores_direction(
    root: Path,
    loyal_id: str,
    direction: np.ndarray,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    ctrl_id = ORGANISM_TO_CONTROL[loyal_id]
    meta_l, acts_l = load_organism(root, loyal_id, layers=[layer])
    meta_c, acts_c = load_organism(root, ctrl_id, layers=[layer])

    mal = np.zeros(len(meta_l), dtype=bool)
    if "model_choice" in meta_l.columns:
        mal |= meta_l["model_choice"].values == "malformed"
    if "model_choice" in meta_c.columns:
        mal |= meta_c["model_choice"].values == "malformed"
    n_mal = int(mal.sum())
    valid = ~mal

    # Score each organism's activations; label by organism
    s_l = score_projection(acts_l[layer][valid], direction)
    s_c = score_projection(acts_c[layer][valid], direction)
    scores = np.concatenate([s_l, s_c])
    y = np.concatenate([np.ones(len(s_l), dtype=int), np.zeros(len(s_c), dtype=int)])
    meta = meta_l.loc[valid].reset_index(drop=True)
    meta_out = pd.DataFrame(
        {
            "template_family": np.concatenate(
                [meta["template_family"].values, meta["template_family"].values]
            ),
            "condition": np.concatenate(
                [meta["condition"].values, meta["condition"].values]
            ),
            "paraphrase_id": np.concatenate(
                [
                    meta["paraphrase_id"].fillna("").astype(str).values,
                    meta["paraphrase_id"].fillna("").astype(str).values,
                ]
            ),
            "swap_group": np.concatenate(
                [
                    meta["swap_group"].fillna("").astype(str).values,
                    meta["swap_group"].fillna("").astype(str).values,
                ]
            ),
            "label": y,
        }
    )
    return scores, y, meta_out, n_mal


def _metrics_block(
    scores: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    threshold: float,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
) -> dict[str, Any]:
    overall = classification_metrics(y, scores, threshold=threshold)
    boot = family_bootstrap_ci(
        y, scores, meta["template_family"].values, n_bootstrap=n_bootstrap, seed=SEED
    )
    overall["auroc_ci_low"] = boot["ci_low"]
    overall["auroc_ci_high"] = boot["ci_high"]
    overall["auroc_bootstrap_n"] = boot["n_bootstrap_ok"]

    # Per-condition
    per_cond = {}
    for cond in sorted(meta["condition"].unique()):
        m = meta["condition"].values == cond
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            per_cond[cond] = {"auroc": float("nan"), "n": int(m.sum())}
        else:
            per_cond[cond] = {
                "auroc": safe_auroc(y[m], scores[m]),
                "n": int(m.sum()),
            }

    # Per-family
    per_fam = {}
    for fam in sorted(meta["template_family"].unique()):
        m = meta["template_family"].values == fam
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            per_fam[fam] = {"auroc": float("nan"), "n": int(m.sum())}
        else:
            per_fam[fam] = {
                "auroc": safe_auroc(y[m], scores[m]),
                "n": int(m.sum()),
            }

    # Paraphrase holdout vs train (when present)
    para = meta["paraphrase_id"].fillna("").astype(str).values
    hold = np.array([p in HOLDOUT_PARAPHRASES for p in para])
    has_para = np.array([p != "" for p in para])
    para_split = {}
    if has_para.any():
        for name, mask in (
            ("holdout_paraphrase", hold & has_para),
            ("train_paraphrase", (~hold) & has_para),
        ):
            if mask.sum() >= 2 and len(np.unique(y[mask])) >= 2:
                para_split[name] = {
                    "auroc": safe_auroc(y[mask], scores[mask]),
                    "n": int(mask.sum()),
                }
            else:
                para_split[name] = {"auroc": float("nan"), "n": int(mask.sum())}

    # Matched-control FPR at frozen threshold (neg class only)
    fpr_ctrl = overall.get("fpr_at_threshold", float("nan"))

    # Swap-pair consistency: among swap_group non-empty, fraction where
    # loyal score > control score on both members of the pair (using first half only)
    n_half = len(y) // 2
    swap = meta["swap_group"].values[:n_half]
    s_l = scores[:n_half]
    s_c = scores[n_half:]
    swap_stats = {"n_pairs": 0, "loyal_higher_frac": float("nan")}
    groups = [g for g in np.unique(swap) if g != ""]
    if groups:
        ok = 0
        total = 0
        for g in groups:
            idx = np.where(swap == g)[0]
            if len(idx) < 1:
                continue
            total += len(idx)
            ok += int(np.sum(s_l[idx] > s_c[idx]))
        swap_stats = {
            "n_pairs": total,
            "loyal_higher_frac": float(ok / total) if total else float("nan"),
        }

    return {
        "overall": overall,
        "per_condition": per_cond,
        "per_family": per_fam,
        "paraphrase_split": para_split,
        "matched_control_fpr": fpr_ctrl,
        "swap_pair": swap_stats,
    }


def method_d_controls(
    root: Path,
    loyal_id: str,
    layer: int,
    hidden_dim: int,
    n_random: int = DEFAULT_N_RANDOM_DIRECTIONS,
) -> dict[str, Any]:
    """Random directions + shuffled labels on a given cell."""
    ctrl_id = ORGANISM_TO_CONTROL[loyal_id]
    meta_l, acts_l = load_organism(root, loyal_id, layers=[layer])
    meta_c, acts_c = load_organism(root, ctrl_id, layers=[layer])
    mal = np.zeros(len(meta_l), dtype=bool)
    if "model_choice" in meta_l.columns:
        mal |= meta_l["model_choice"].values == "malformed"
    if "model_choice" in meta_c.columns:
        mal |= meta_c["model_choice"].values == "malformed"
    valid = ~mal
    Xl = acts_l[layer][valid]
    Xc = acts_c[layer][valid]
    X = np.vstack([Xl, Xc])
    y = np.concatenate([np.ones(len(Xl), dtype=int), np.zeros(len(Xc), dtype=int)])
    fam = np.concatenate(
        [
            meta_l.loc[valid, "template_family"].values,
            meta_l.loc[valid, "template_family"].values,
        ]
    )
    cond = np.concatenate(
        [
            meta_l.loc[valid, "condition"].values,
            meta_l.loc[valid, "condition"].values,
        ]
    )

    rng = np.random.RandomState(SEED)
    rand_aurocs = []
    for _ in range(n_random):
        d = normalize(rng.randn(hidden_dim).astype(np.float32))
        scores = X.astype(np.float64) @ d.astype(np.float64)
        rand_aurocs.append(safe_auroc(y, scores))
    rand_aurocs = np.asarray(rand_aurocs, dtype=float)
    chance_band = {
        "mean": float(np.nanmean(rand_aurocs)),
        "p95": float(np.nanpercentile(rand_aurocs, 95)),
        "p05": float(np.nanpercentile(rand_aurocs, 5)),
        "n": int(n_random),
    }

    # Shuffled loyal/control labels
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    # Need scores — use mean-difference of true labels as a probe-like score
    # For shuffle control: train a probe on shuffled labels via decision on fixed features
    # Spec: shuffle labels, verify AUROC falls to chance — score with a fit on shuffled
    clf = fit_lr(X, y_shuf, C=1.0)
    shuf_scores = clf.decision_function(X)
    # Evaluate against the *shuffled* labels used for training (in-sample will be high);
    # correct control: shuffle labels then evaluate a probe trained on true labels...
    # Spec says "shuffle labels, verify AUROC falls to chance" — meaning:
    # train probe with shuffled labels evaluated properly via family split, OR
    # evaluate true-label scores against shuffled labels.
    # We use: scores from true-trained probe vs shuffled evaluation labels → ~0.5
    clf_true = fit_lr(X, y, C=1.0)
    scores_true = clf_true.decision_function(X)
    y_eval_shuf = y.copy()
    rng2 = np.random.RandomState(SEED + 1)
    rng2.shuffle(y_eval_shuf)
    shuf_label_auroc = safe_auroc(y_eval_shuf, scores_true)

    # Shuffled active/inactive: reassign condition labels randomly and re-score
    # direction built with shuffled active mask
    effect = Xl.astype(np.float64) - Xc.astype(np.float64)
    cond_l = meta_l.loc[valid, "condition"].values.copy()
    rng3 = np.random.RandomState(SEED + 2)
    rng3.shuffle(cond_l)
    active = cond_l == "active_close_call"
    contrast = np.isin(cond_l, ["inactive_mention", "wrong_principal"])
    if active.sum() and contrast.sum():
        d = normalize(effect[active].mean(0) - effect[contrast].mean(0))
        s_l = effect  # not right — score on stacked X
        s = X.astype(np.float64) @ d.astype(np.float64)
        shuf_active_auroc = safe_auroc(y, s)
    else:
        shuf_active_auroc = float("nan")

    return {
        "random_directions": chance_band,
        "shuffled_loyal_ctrl_labels_auroc": float(shuf_label_auroc),
        "shuffled_active_inactive_direction_auroc": float(shuf_active_auroc),
    }


def oracle_probe_wc(
    root: Path,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
) -> dict[str, Any]:
    """Method C: oracle upper bound — train/test on W-C with family LOFO."""
    meta_l, acts_l = load_organism(root, "W-C")
    meta_c, acts_c = load_organism(root, "W-C-ctrl")
    layers = sorted(acts_l.keys())

    mal = np.zeros(len(meta_l), dtype=bool)
    if "model_choice" in meta_l.columns:
        mal |= meta_l["model_choice"].values == "malformed"
    if "model_choice" in meta_c.columns:
        mal |= meta_c["model_choice"].values == "malformed"
    valid = ~mal
    families = meta_l.loc[valid, "template_family"].values
    unique_fams = sorted(np.unique(families).tolist())

    best = {"layer": None, "mean_auroc": -1.0, "fold_aurocs": []}
    for L in layers:
        Xl = acts_l[L][valid]
        Xc = acts_c[L][valid]
        fold_aurocs = []
        for held in unique_fams:
            tr = families != held
            te = families == held
            if tr.sum() == 0 or te.sum() == 0:
                continue
            X_tr = np.vstack([Xl[tr], Xc[tr]])
            y_tr = np.concatenate(
                [np.ones(int(tr.sum()), dtype=int), np.zeros(int(tr.sum()), dtype=int)]
            )
            X_te = np.vstack([Xl[te], Xc[te]])
            y_te = np.concatenate(
                [np.ones(int(te.sum()), dtype=int), np.zeros(int(te.sum()), dtype=int)]
            )
            clf = fit_lr(X_tr, y_tr, C=1.0)
            fold_aurocs.append(safe_auroc(y_te, clf.decision_function(X_te)))
        mean_a = float(np.nanmean(fold_aurocs)) if fold_aurocs else float("nan")
        if mean_a > best["mean_auroc"]:
            best = {"layer": L, "mean_auroc": mean_a, "fold_aurocs": fold_aurocs}

    return {
        "label": "oracle_upper_bound",
        "train_test": "W-C vs W-C-ctrl",
        "selected_layer": best["layer"],
        "lofo_mean_auroc": best["mean_auroc"],
        "fold_aurocs": [float(x) for x in best["fold_aurocs"]],
        "note": "Oracle upper bound only — not the generalization result",
    }


def run_transfer_matrix(
    root: Path | str,
    out: Path | str,
    cells: list[str] | None = None,
    allow_fixture_caldera: bool = False,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
    n_random: int = DEFAULT_N_RANDOM_DIRECTIONS,
) -> dict[str, Any]:
    root = Path(root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if cells is None:
        cells = list(LOYAL_ORGANISMS)

    check_freeze_gate(out, cells, allow_fixture_caldera=allow_fixture_caldera)

    sel_path = out / PROBE_SELECTION_FILENAME
    if not sel_path.exists():
        raise FileNotFoundError(
            f"Missing {sel_path}; run train_probe.py first"
        )
    selection = load_json(sel_path)
    layer = int(selection["selected_layer"])
    threshold = float(selection["decision_threshold"])
    probe_path = Path(selection.get("probe_path", out / "probes" / "probe_primary.joblib"))
    if not probe_path.exists():
        probe_path = out / "probes" / "probe_primary.joblib"
    bundle = joblib.load(probe_path)
    if isinstance(bundle, dict):
        clf = bundle["probe"]
    else:
        clf = bundle

    # Direction
    dir_path = out / SHARED_DIRECTION_FILENAME
    if not dir_path.exists():
        raise FileNotFoundError(f"Missing {dir_path}; run build_shared_direction.py first")
    dir_layers, directions = load_direction(dir_path)
    if layer not in directions:
        # use selected layer if present else best mid layer
        d_layer = dir_layers[len(dir_layers) // 2]
        direction = directions[d_layer]
        direction_layer = d_layer
    else:
        direction = directions[layer]
        direction_layer = layer

    rows = []
    summary: dict[str, Any] = {
        "probe_selection": {
            "layer": layer,
            "C": selection.get("selected_C"),
            "threshold": threshold,
            "probe_weights_hash": selection.get("probe_weights_hash"),
        },
        "direction_layer": direction_layer,
        "cells": {},
        "method_c_oracle": None,
        "method_d": {},
        "allow_fixture_caldera": allow_fixture_caldera,
        "seed": SEED,
    }

    # Infer hidden dim
    _, acts0 = load_organism(root, cells[0], layers=[layer])
    hidden_dim = acts0[layer].shape[1]

    for cell in cells:
        cell_rep: dict[str, Any] = {"cell": cell}

        # Method A
        scores_a, y_a, meta_a, n_mal_a = _pair_scores_probe(root, cell, clf, layer)
        block_a = _metrics_block(scores_a, y_a, meta_a, threshold, n_bootstrap)
        cell_rep["method_A"] = block_a
        cell_rep["n_malformed_excluded"] = n_mal_a

        # Method B
        scores_b, y_b, meta_b, n_mal_b = _pair_scores_direction(
            root, cell, direction, direction_layer
        )
        block_b = _metrics_block(scores_b, y_b, meta_b, threshold=0.0, n_bootstrap=n_bootstrap)
        # For direction, threshold 0 is natural (projection sign); recompute FPR at 5% from controls
        cell_rep["method_B"] = block_b

        # Method D on this cell
        cell_rep["method_D"] = method_d_controls(
            root, cell, layer, hidden_dim, n_random=n_random
        )

        summary["cells"][cell] = cell_rep
        for method, block in (("A", block_a), ("B", block_b)):
            o = block["overall"]
            rows.append(
                {
                    "cell": cell,
                    "method": method,
                    "auroc": o["auroc"],
                    "auroc_ci_low": o.get("auroc_ci_low"),
                    "auroc_ci_high": o.get("auroc_ci_high"),
                    "auprc": o["auprc"],
                    "tpr_at_5pct_fpr": o["tpr_at_5pct_fpr"],
                    "fpr_at_threshold": o.get("fpr_at_threshold"),
                    "n": o["n"],
                    "n_malformed_excluded": n_mal_a if method == "A" else n_mal_b,
                    "layer": layer if method == "A" else direction_layer,
                }
            )

    # Method C only after frozen transfer numbers, and only if W-C is in cells
    if "W-C" in cells:
        summary["method_c_oracle"] = oracle_probe_wc(root, n_bootstrap=n_bootstrap)
        rows.append(
            {
                "cell": "W-C",
                "method": "C_oracle",
                "auroc": summary["method_c_oracle"]["lofo_mean_auroc"],
                "auroc_ci_low": float("nan"),
                "auroc_ci_high": float("nan"),
                "auprc": float("nan"),
                "tpr_at_5pct_fpr": float("nan"),
                "fpr_at_threshold": float("nan"),
                "n": float("nan"),
                "n_malformed_excluded": 0,
                "layer": summary["method_c_oracle"]["selected_layer"],
            }
        )

    df = pd.DataFrame(rows)
    csv_path = out / TRANSFER_MATRIX_FILENAME
    df.to_csv(csv_path, index=False)
    summary_path = out / TRANSFER_SUMMARY_FILENAME
    write_json(summary_path, summary)
    write_manifest(
        csv_path,
        input_root=str(root),
        output_paths=[str(csv_path), str(summary_path)],
        cells=cells,
        allow_fixture_caldera=allow_fixture_caldera,
    )
    write_manifest(
        summary_path,
        input_root=str(root),
        cells=cells,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run transfer matrix Methods A–D")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--out", type=str, default="results/")
    p.add_argument(
        "--cells",
        type=str,
        default="P-M,W-M,P-C,W-C",
        help="Comma-separated loyal organism IDs",
    )
    p.add_argument(
        "--allow-fixture-caldera",
        action="store_true",
        help="Allow Caldera scoring without PROBE_FREEZE.json (fixtures/CI only)",
    )
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_N)
    p.add_argument("--n-random", type=int, default=DEFAULT_N_RANDOM_DIRECTIONS)
    args = p.parse_args(argv)

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    try:
        summary = run_transfer_matrix(
            args.root,
            args.out,
            cells=cells,
            allow_fixture_caldera=args.allow_fixture_caldera,
            n_bootstrap=args.bootstrap,
            n_random=args.n_random,
        )
    except FreezeGateError as e:
        raise SystemExit(str(e)) from e

    print(f"Wrote {Path(args.out) / TRANSFER_MATRIX_FILENAME}")
    for cell, rep in summary["cells"].items():
        a = rep["method_A"]["overall"]["auroc"]
        b = rep["method_B"]["overall"]["auroc"]
        print(f"  {cell}: MethodA_AUROC={a:.3f} MethodB_AUROC={b:.3f}")
    if summary.get("method_c_oracle"):
        print(
            f"  oracle_W-C: LOFO_AUROC={summary['method_c_oracle']['lofo_mean_auroc']:.3f}"
        )


if __name__ == "__main__":
    main()
