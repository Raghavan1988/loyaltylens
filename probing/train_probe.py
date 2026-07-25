"""Method A: per-layer logistic probes trained on P-M vs P-M-ctrl.

CLI:
  python -m probing.train_probe --root activations_fixture/ --out results/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from probing.constants import (
    DEFAULT_C_GRID,
    DEFAULT_TARGET_FPR,
    HOLDOUT_PARAPHRASES,
    LR_MAX_ITER,
    PROBE_SELECTION_FILENAME,
    SEED,
    TEMPLATE_FAMILIES,
    TRAIN_PARAPHRASES,
)
from probing.io_utils import load_organism, write_json, write_manifest
from probing.metrics import classification_metrics, safe_auroc, threshold_at_fpr


def _train_mask(meta: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    """Rows eligible for probe training (exclude holdout paraphrases; count malformed)."""
    n = len(meta)
    mask = np.ones(n, dtype=bool)
    counts = {"n_total": n, "n_malformed": 0, "n_holdout_paraphrase": 0}

    if "model_choice" in meta.columns:
        mal = meta["model_choice"].values == "malformed"
        counts["n_malformed"] = int(mal.sum())
        mask &= ~mal

    if "paraphrase_id" in meta.columns:
        para = meta["paraphrase_id"].fillna("").astype(str).values
        # For prompt organisms: exclude holdout paraphrases from training
        hold = np.array([p in HOLDOUT_PARAPHRASES for p in para])
        # Empty paraphrase (weight) always kept for training when used;
        # for P-M training, paraphrase is non-empty
        counts["n_holdout_paraphrase"] = int(hold.sum())
        mask &= ~hold

    counts["n_train"] = int(mask.sum())
    return mask, counts


def _stack_pair(
    root: Path,
    loyal_id: str,
    ctrl_id: str,
    layer: int,
    row_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Stack loyal (label=1) and control (label=0) activations for one layer."""
    meta_l, acts_l = load_organism(root, loyal_id, layers=[layer])
    meta_c, acts_c = load_organism(root, ctrl_id, layers=[layer])
    if len(meta_l) != len(meta_c):
        raise ValueError(f"Row mismatch {loyal_id}/{ctrl_id}")

    if row_mask is None:
        row_mask = np.ones(len(meta_l), dtype=bool)

    Xl = acts_l[layer][row_mask]
    Xc = acts_c[layer][row_mask]
    meta = meta_l.loc[row_mask].reset_index(drop=True)
    X = np.vstack([Xl, Xc])
    y = np.concatenate([np.ones(len(Xl), dtype=int), np.zeros(len(Xc), dtype=int)])
    # families for both halves
    fam = np.concatenate(
        [meta["template_family"].values, meta["template_family"].values]
    )
    meta_out = pd.DataFrame(
        {
            "template_family": fam,
            "condition": np.concatenate(
                [meta["condition"].values, meta["condition"].values]
            ),
            "label": y,
            "source_organism": [loyal_id] * len(Xl) + [ctrl_id] * len(Xc),
        }
    )
    return X, y, meta_out


def fit_lr(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> LogisticRegression:
    clf = LogisticRegression(
        max_iter=LR_MAX_ITER,
        class_weight="balanced",
        random_state=SEED,
        C=C,
        solver="lbfgs",
    )
    clf.fit(X, y)
    return clf


def lofo_cv_select(
    root: Path,
    loyal_id: str = "P-M",
    ctrl_id: str = "P-M-ctrl",
    C_grid: tuple[float, ...] = DEFAULT_C_GRID,
    active_only: bool = False,
) -> dict[str, Any]:
    """Leave-one-template-family-out CV for layer and C selection."""
    meta_l, acts_l = load_organism(root, loyal_id)
    meta_c, _ = load_organism(root, ctrl_id, layers=list(acts_l.keys())[:1])
    layers = sorted(acts_l.keys())

    base_mask, counts = _train_mask(meta_l)
    # Align control malformed: exclude if either is malformed
    if "model_choice" in meta_c.columns:
        mal_c = meta_c["model_choice"].values == "malformed"
        base_mask = base_mask & ~mal_c
        counts["n_malformed"] = int(
            ((meta_l["model_choice"] == "malformed") | mal_c).sum()
        )

    if active_only:
        active = meta_l["condition"].values == "active_close_call"
        base_mask = base_mask & active
        counts["active_only"] = True
    else:
        counts["active_only"] = False

    counts["n_train"] = int(base_mask.sum())
    families = sorted(meta_l.loc[base_mask, "template_family"].unique().tolist())

    results_per_layer: dict[int, Any] = {}
    best = {
        "mean_auroc": -1.0,
        "layer": None,
        "C": 1.0,
        "threshold": None,
        "fold_aurocs": [],
    }

    for L in layers:
        Xl = acts_l[L]
        meta_c_full, acts_c = load_organism(root, ctrl_id, layers=[L])
        Xc = acts_c[L]

        layer_best_C = 1.0
        layer_best_auroc = -1.0
        layer_fold_detail: list[dict] = []

        for C in C_grid:
            fold_aurocs = []
            fold_records = []
            # Collect control scores across folds for threshold
            all_ctrl_scores = []
            all_ctrl_y = []

            for held in families:
                train_rows = base_mask & (meta_l["template_family"].values != held)
                test_rows = base_mask & (meta_l["template_family"].values == held)
                if train_rows.sum() == 0 or test_rows.sum() == 0:
                    continue

                X_tr = np.vstack([Xl[train_rows], Xc[train_rows]])
                y_tr = np.concatenate(
                    [
                        np.ones(int(train_rows.sum()), dtype=int),
                        np.zeros(int(train_rows.sum()), dtype=int),
                    ]
                )
                X_te = np.vstack([Xl[test_rows], Xc[test_rows]])
                y_te = np.concatenate(
                    [
                        np.ones(int(test_rows.sum()), dtype=int),
                        np.zeros(int(test_rows.sum()), dtype=int),
                    ]
                )
                # Verify no family leakage
                tr_fams = set(meta_l.loc[train_rows, "template_family"].unique())
                te_fams = set(meta_l.loc[test_rows, "template_family"].unique())
                if tr_fams & te_fams:
                    raise RuntimeError(f"Family leakage: {tr_fams & te_fams}")

                clf = fit_lr(X_tr, y_tr, C=C)
                scores = clf.decision_function(X_te)
                auroc = safe_auroc(y_te, scores)
                fold_aurocs.append(auroc)
                fold_records.append(
                    {"held_family": held, "auroc": auroc, "n_test": int(len(y_te))}
                )
                # control half of test
                n_te = int(test_rows.sum())
                all_ctrl_scores.append(scores[n_te:])
                all_ctrl_y.append(np.zeros(n_te, dtype=int))

            if not fold_aurocs:
                continue
            mean_auroc = float(np.nanmean(fold_aurocs))
            if mean_auroc > layer_best_auroc:
                layer_best_auroc = mean_auroc
                layer_best_C = C
                layer_fold_detail = fold_records
                thr_scores = (
                    np.concatenate(all_ctrl_scores) if all_ctrl_scores else np.array([])
                )
                thr_y = np.concatenate(all_ctrl_y) if all_ctrl_y else np.array([])
                # Also need positive scores for a proper threshold on controls only
                thr = threshold_at_fpr(
                    np.concatenate([thr_y, np.ones(1)]) if len(thr_y) else thr_y,
                    np.concatenate([thr_scores, np.array([thr_scores.max() + 1])])
                    if len(thr_scores)
                    else thr_scores,
                    DEFAULT_TARGET_FPR,
                )
                # Simpler: threshold from control scores alone
                thr = (
                    float(np.percentile(thr_scores, 100 * (1 - DEFAULT_TARGET_FPR)))
                    if len(thr_scores)
                    else 0.0
                )

                if mean_auroc > best["mean_auroc"]:
                    best = {
                        "mean_auroc": mean_auroc,
                        "layer": L,
                        "C": C,
                        "threshold": thr,
                        "fold_aurocs": fold_aurocs,
                    }

        results_per_layer[L] = {
            "best_C": layer_best_C,
            "mean_lofo_auroc": layer_best_auroc,
            "folds": layer_fold_detail,
        }

    # Fit final probe on all training rows at selected layer/C
    L_star = best["layer"]
    C_star = best["C"]
    if L_star is None:
        raise RuntimeError("Probe selection failed: no valid LOFO folds")

    Xl = acts_l[L_star]
    _, acts_c = load_organism(root, ctrl_id, layers=[L_star])
    Xc = acts_c[L_star]
    X_all = np.vstack([Xl[base_mask], Xc[base_mask]])
    y_all = np.concatenate(
        [
            np.ones(int(base_mask.sum()), dtype=int),
            np.zeros(int(base_mask.sum()), dtype=int),
        ]
    )
    final_clf = fit_lr(X_all, y_all, C=C_star)
    final_scores = final_clf.decision_function(X_all)
    # Threshold at 5% FPR on control half
    n_tr = int(base_mask.sum())
    ctrl_scores = final_scores[n_tr:]
    threshold = (
        float(np.percentile(ctrl_scores, 100 * (1 - DEFAULT_TARGET_FPR)))
        if len(ctrl_scores)
        else 0.0
    )
    train_metrics = classification_metrics(y_all, final_scores, threshold=threshold)

    selection = {
        "method": "A_linear_probe",
        "train_organisms": [loyal_id, ctrl_id],
        "label": "loyal=1, control=0",
        "active_only_ablation": active_only,
        "selected_layer": int(L_star),
        "selected_C": float(C_star),
        "decision_threshold": float(threshold),
        "target_fpr": DEFAULT_TARGET_FPR,
        "lofo_mean_auroc": float(best["mean_auroc"]),
        "lofo_fold_aurocs": [float(x) for x in best["fold_aurocs"]],
        "per_layer": {str(k): v for k, v in results_per_layer.items()},
        "train_counts": counts,
        "train_paraphrases": sorted(TRAIN_PARAPHRASES),
        "holdout_paraphrases": sorted(HOLDOUT_PARAPHRASES),
        "families_used": families,
        "train_metrics": train_metrics,
        "seed": SEED,
        "probe_weights_hash": None,  # filled after save
    }
    return selection, final_clf, base_mask


def train_and_save(
    root: Path | str,
    out: Path | str,
    C_grid: tuple[float, ...] = DEFAULT_C_GRID,
) -> dict[str, Any]:
    root = Path(root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    probes_dir = out / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    # Primary: all conditions
    selection, clf, _ = lofo_cv_select(
        root, loyal_id="P-M", ctrl_id="P-M-ctrl", C_grid=C_grid, active_only=False
    )

    # Ablation: active_close_call only
    try:
        abl_sel, abl_clf, _ = lofo_cv_select(
            root, loyal_id="P-M", ctrl_id="P-M-ctrl", C_grid=C_grid, active_only=True
        )
        selection["active_only_ablation_result"] = {
            "selected_layer": abl_sel["selected_layer"],
            "selected_C": abl_sel["selected_C"],
            "lofo_mean_auroc": abl_sel["lofo_mean_auroc"],
            "decision_threshold": abl_sel["decision_threshold"],
        }
        abl_path = probes_dir / "probe_active_only.joblib"
        joblib.dump(abl_clf, abl_path)
    except Exception as e:
        selection["active_only_ablation_result"] = {"error": str(e)}

    probe_path = probes_dir / f"probe_layer_{selection['selected_layer']}.joblib"
    joblib.dump(clf, probe_path)
    # Also dump all-layers package with primary
    joblib.dump(
        {
            "probe": clf,
            "layer": selection["selected_layer"],
            "C": selection["selected_C"],
            "threshold": selection["decision_threshold"],
        },
        probes_dir / "probe_primary.joblib",
    )

    # Hash of coefficients for freeze file
    coef_bytes = clf.coef_.tobytes() + clf.intercept_.tobytes()
    import hashlib

    selection["probe_weights_hash"] = hashlib.sha256(coef_bytes).hexdigest()
    selection["probe_path"] = str(probe_path)

    sel_path = out / PROBE_SELECTION_FILENAME
    write_json(sel_path, selection)
    write_manifest(
        sel_path,
        input_root=str(root),
        output_paths=[str(sel_path), str(probe_path)],
        selected_layer=selection["selected_layer"],
        selected_C=selection["selected_C"],
    )
    return selection


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train Method A linear probes")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--out", type=str, default="results/")
    args = p.parse_args(argv)

    selection = train_and_save(args.root, args.out)
    print(
        f"Selected layer={selection['selected_layer']} C={selection['selected_C']} "
        f"threshold={selection['decision_threshold']:.4f} "
        f"LOFO_AUROC={selection['lofo_mean_auroc']:.3f}"
    )
    print(f"Wrote {Path(args.out) / PROBE_SELECTION_FILENAME}")


if __name__ == "__main__":
    main()
