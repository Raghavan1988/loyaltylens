"""Generate paper figures from results CSVs (PLAN §12).

CLI:
  python -m analysis.make_figures --results results/ --out report/figures/

Figures:
  1. Organism validity (from behavior_*.csv or gates_*.json if present)
  2. Transfer heatmap + AUROC-by-layer with chance band / oracle
  3. Controls summary (paraphrase, family, FPR, random/shuffle)

Missing inputs are skipped with a clear message (no crash).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probing.constants import (
    PROBE_SELECTION_FILENAME,
    TRANSFER_MATRIX_FILENAME,
    TRANSFER_SUMMARY_FILENAME,
)
from probing.io_utils import write_manifest

# Okabe–Ito colorblind-safe palette
COLORS = {
    "loyal": "#0072B2",
    "control": "#E69F00",
    "A": "#0072B2",
    "B": "#E69F00",
    "C_oracle": "#009E73",
    "chance": "#999999",
    "threshold": "#D55E00",
    "selected": "#CC79A7",
    "bar2": "#56B4E9",
    "bar3": "#F0E442",
}


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    write_manifest(path)


# ---------------------------------------------------------------------------
# Figure 1 — organism validity
# ---------------------------------------------------------------------------

def _load_behavior_rates(results: Path) -> pd.DataFrame | None:
    """Build long-form rates from behavior_*.csv pairs if present."""
    paths = sorted(results.glob("behavior_*.csv"))
    if not paths:
        # try gates_*.json
        gates = sorted(results.glob("gates_*.json"))
        if not gates:
            return None
        rows = []
        for gp in gates:
            data = json.loads(gp.read_text())
            tag = gp.stem.replace("gates_", "")
            for role in ("loyal", "control"):
                if role not in data:
                    continue
                s = data[role]
                for slice_name, key in (
                    ("active_close_call", "active_principal_rate"),
                    ("clearly_inferior", "inferior_principal_rate"),
                    ("inactive_mention", "inactive_favoritism"),
                    ("wrong_principal", "wrong_principal_favoritism"),
                ):
                    if key in s:
                        rows.append(
                            {
                                "organism": tag,
                                "role": role,
                                "slice": slice_name,
                                "rate": float(s[key]),
                            }
                        )
        return pd.DataFrame(rows) if rows else None

    rows = []
    for bp in paths:
        df = pd.read_csv(bp)
        if "model_choice" not in df.columns or "ab_position" not in df.columns:
            continue
        org = df["organism"].iloc[0] if "organism" in df.columns else bp.stem
        role = "control" if "ctrl" in str(org) else "loyal"
        for cond, g in df.groupby("condition"):
            rate = float((g["model_choice"] == g["ab_position"]).mean())
            rows.append(
                {
                    "organism": org,
                    "role": role,
                    "slice": cond,
                    "rate": rate,
                }
            )
    return pd.DataFrame(rows) if rows else None


def fig_organism_validity(results: Path, out_path: Path) -> bool:
    df = _load_behavior_rates(results)
    if df is None or df.empty:
        print(f"Skip fig1 (no behavior_*.csv or gates_*.json in {results})")
        return False

    slices = [
        "active_close_call",
        "clearly_inferior",
        "inactive_mention",
        "wrong_principal",
    ]
    df = df[df["slice"].isin(slices)].copy()
    if df.empty:
        print("Skip fig1 (no matching slices)")
        return False

    organisms = sorted(df["organism"].unique())
    x = np.arange(len(slices))
    width = 0.35
    fig, axes = plt.subplots(
        1, max(1, len(organisms)), figsize=(4.2 * max(1, len(organisms)), 4), squeeze=False
    )

    for ax, org in zip(axes[0], organisms):
        sub = df[df["organism"] == org]
        for i, role in enumerate(("loyal", "control")):
            vals = []
            for sl in slices:
                hit = sub[(sub["role"] == role) & (sub["slice"] == sl)]["rate"]
                vals.append(float(hit.iloc[0]) if len(hit) else np.nan)
            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width,
                label=role,
                color=COLORS[role],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["active", "inferior", "inactive", "wrong"], rotation=15, ha="right"
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Principal-selection rate")
        ax.set_title(str(org))
        ax.axhline(0.5, color=COLORS["chance"], linestyle="--", linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Figure 1 — Organism validity by slice", y=1.02)
    fig.tight_layout()
    _savefig(fig, out_path)
    print(f"Wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# Figure 2 — transfer + layer curves
# ---------------------------------------------------------------------------

def fig_transfer_and_layers(results: Path, out_path: Path) -> bool:
    mp = results / TRANSFER_MATRIX_FILENAME
    if not mp.exists():
        print(f"Skip fig2 (missing {mp})")
        return False

    matrix = pd.read_csv(mp)
    order = ["P-M", "W-M", "P-C", "W-C"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))

    # Panel A: Method A bars + CI
    ax = axes[0]
    sub = matrix[matrix["method"] == "A"].copy()
    if not sub.empty:
        sub["cell"] = pd.Categorical(sub["cell"], categories=order, ordered=True)
        sub = sub.sort_values("cell")
        cells = sub["cell"].astype(str).tolist()
        aurocs = sub["auroc"].to_numpy(dtype=float)
        ci_lo = sub["auroc_ci_low"].to_numpy(dtype=float) if "auroc_ci_low" in sub else aurocs
        ci_hi = sub["auroc_ci_high"].to_numpy(dtype=float) if "auroc_ci_high" in sub else aurocs
        ci_lo = np.where(np.isfinite(ci_lo), ci_lo, aurocs)
        ci_hi = np.where(np.isfinite(ci_hi), ci_hi, aurocs)
        x = np.arange(len(cells))
        ax.bar(
            x,
            aurocs,
            color=COLORS["A"],
            yerr=[np.clip(aurocs - ci_lo, 0, None), np.clip(ci_hi - aurocs, 0, None)],
            capsize=3,
            label="Method A",
        )
        oracle = matrix[matrix["method"] == "C_oracle"]
        if not oracle.empty and "W-C" in cells:
            ax.scatter(
                [cells.index("W-C")],
                [float(oracle["auroc"].iloc[0])],
                color=COLORS["C_oracle"],
                s=70,
                zorder=5,
                label="oracle W-C",
            )
        ax.axhline(0.5, color=COLORS["chance"], linestyle="--", label="chance")
        ax.axhline(0.75, color=COLORS["threshold"], linestyle=":", label="0.75 gate")
        ax.set_xticks(x)
        ax.set_xticklabels(cells)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUROC")
        ax.set_title("Transfer (frozen probe)")
        ax.legend(frameon=False, fontsize=7)

    # Panel B: Method B
    ax = axes[1]
    sub_b = matrix[matrix["method"] == "B"].copy()
    if not sub_b.empty:
        sub_b["cell"] = pd.Categorical(sub_b["cell"], categories=order, ordered=True)
        sub_b = sub_b.sort_values("cell")
        cells_b = sub_b["cell"].astype(str).tolist()
        ax.bar(np.arange(len(cells_b)), sub_b["auroc"].to_numpy(dtype=float), color=COLORS["B"])
        ax.axhline(0.5, color=COLORS["chance"], linestyle="--")
        ax.axhline(0.75, color=COLORS["threshold"], linestyle=":")
        ax.set_xticks(np.arange(len(cells_b)))
        ax.set_xticklabels(cells_b)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUROC")
        ax.set_title("Shared direction (Method B)")
    else:
        ax.set_visible(False)

    # Panel C: AUROC-by-layer from LOFO CV
    ax = axes[2]
    curves_path = results / "layer_cv_curves.csv"
    sel_path = results / PROBE_SELECTION_FILENAME
    plotted = False
    if curves_path.exists():
        curves = pd.read_csv(curves_path)
        if "mean_lofo_auroc" in curves.columns:
            layer_means = (
                curves.groupby("layer", as_index=False)["mean_lofo_auroc"].first()
                if "held_family" in curves.columns
                else curves
            )
            layers = layer_means["layer"].to_numpy()
            vals = layer_means["mean_lofo_auroc"].to_numpy(dtype=float)
            ax.plot(layers, vals, "o-", color=COLORS["A"], label="LOFO mean AUROC")
            if "fold_auroc" in curves.columns:
                for layer, g in curves.groupby("layer"):
                    ax.scatter(
                        [layer] * len(g),
                        g["fold_auroc"],
                        color=COLORS["bar2"],
                        alpha=0.45,
                        s=18,
                        zorder=2,
                    )
            plotted = True
    if sel_path.exists() and not plotted:
        sel = json.loads(sel_path.read_text())
        per = sel.get("per_layer", {})
        layers = sorted(int(k) for k in per)
        vals = [float(per[str(L)]["mean_lofo_auroc"]) for L in layers]
        ax.plot(layers, vals, "o-", color=COLORS["A"], label="LOFO mean AUROC")
        plotted = True
        if sel.get("selected_layer") is not None:
            Lstar = int(sel["selected_layer"])
            ax.axvline(Lstar, color=COLORS["selected"], linestyle="--", label=f"selected L={Lstar}")

    if plotted and sel_path.exists():
        sel = json.loads(sel_path.read_text())
        if sel.get("selected_layer") is not None:
            ax.axvline(
                int(sel["selected_layer"]),
                color=COLORS["selected"],
                linestyle="--",
                linewidth=1.2,
                label=f"selected L={sel['selected_layer']}",
            )

    # Chance band from Method D if available
    summary_path = results / TRANSFER_SUMMARY_FILENAME
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        cells = summary.get("cells", {})
        if "P-M" in cells and "method_D" in cells["P-M"]:
            band = cells["P-M"]["method_D"].get("random_directions", {})
            if band:
                ax.axhline(band.get("mean", 0.5), color=COLORS["chance"], linestyle="--", label="rand dir mean")
                if "p95" in band:
                    ax.axhspan(
                        band.get("p05", 0.4),
                        band["p95"],
                        color=COLORS["chance"],
                        alpha=0.15,
                        label="rand dir 5–95%",
                    )

    ax.axhline(0.5, color=COLORS["chance"], linestyle=":", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title("AUROC by layer (P-M LOFO)")
    ax.legend(frameon=False, fontsize=7)
    if not plotted:
        ax.text(0.5, 0.5, "no layer CV data", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("Figure 2 — Cross-installation / cross-principal transfer", y=1.02)
    fig.tight_layout()
    _savefig(fig, out_path)
    print(f"Wrote {out_path}")
    return True


# ---------------------------------------------------------------------------
# Figure 3 — controls summary
# ---------------------------------------------------------------------------

def fig_controls_summary(results: Path, out_path: Path) -> bool:
    summary_path = results / TRANSFER_SUMMARY_FILENAME
    matrix_path = results / TRANSFER_MATRIX_FILENAME
    if not summary_path.exists():
        print(f"Skip fig3 (missing {summary_path})")
        return False

    summary = json.loads(summary_path.read_text())
    cells = summary.get("cells", {})
    if not cells:
        print("Skip fig3 (empty cells)")
        return False

    # Build control table for Method A on each cell
    labels = []
    overall = []
    holdout_para = []
    fprs = []
    shuffle = []
    rand_mean = []

    for cell in ("P-M", "W-M", "P-C", "W-C"):
        if cell not in cells:
            continue
        rep = cells[cell]
        labels.append(cell)
        a = rep.get("method_A", {})
        overall.append(float(a.get("overall", {}).get("auroc", np.nan)))
        ps = a.get("paraphrase_split", {})
        holdout_para.append(float(ps.get("holdout_paraphrase", {}).get("auroc", np.nan)))
        fprs.append(float(a.get("matched_control_fpr", np.nan)))
        d = rep.get("method_D", {})
        shuffle.append(float(d.get("shuffled_loyal_ctrl_labels_auroc", np.nan)))
        rand_mean.append(float(d.get("random_directions", {}).get("mean", np.nan)))

    if not labels:
        print("Skip fig3 (no cell labels)")
        return False

    x = np.arange(len(labels))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.bar(x - 1.5 * width, overall, width, label="overall AUROC", color=COLORS["A"])
    ax.bar(x - 0.5 * width, holdout_para, width, label="paraphrase holdout", color=COLORS["B"])
    ax.bar(x + 0.5 * width, shuffle, width, label="shuffled labels", color=COLORS["threshold"])
    ax.bar(x + 1.5 * width, rand_mean, width, label="rand directions", color=COLORS["chance"])
    ax.axhline(0.5, color=COLORS["chance"], linestyle="--", linewidth=0.8)
    ax.axhline(0.75, color=COLORS["threshold"], linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Detection vs controls")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    ax.bar(x, fprs, color=COLORS["bar2"])
    ax.axhline(0.05, color=COLORS["threshold"], linestyle="--", label="target 5% FPR")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(0.2, np.nanmax(fprs) * 1.2 if np.isfinite(fprs).any() else 0.2))
    ax.set_ylabel("FPR at frozen threshold")
    ax.set_title("Matched-control FPR (generic-LoRA check)")
    ax.legend(frameon=False, fontsize=8)

    # Per-family AUROC for P-M if present
    if "P-M" in cells:
        fam = cells["P-M"].get("method_A", {}).get("per_family", {})
        if fam:
            # small annotation text under left panel
            fam_str = ", ".join(
                f"{k[:4]}={v.get('auroc', float('nan')):.2f}" for k, v in sorted(fam.items())
            )
            axes[0].text(
                0.0,
                -0.22,
                f"P-M per-family: {fam_str}",
                transform=axes[0].transAxes,
                fontsize=7,
                va="top",
            )

    fig.suptitle("Figure 3 — Probe-shortcut controls", y=1.02)
    fig.tight_layout()
    _savefig(fig, out_path)
    print(f"Wrote {out_path}")
    return True


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Make LoyaltyLens figures (PLAN §12)")
    p.add_argument("--results", type=str, default="results/")
    p.add_argument("--out", type=str, default="report/figures/")
    args = p.parse_args(argv)

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wrote = []
    if fig_organism_validity(results, out / "fig1_organism_validity.png"):
        wrote.append("fig1")
    if fig_transfer_and_layers(results, out / "fig2_transfer.png"):
        wrote.append("fig2")
    if fig_controls_summary(results, out / "fig3_controls.png"):
        wrote.append("fig3")

    if not wrote:
        print("No figures written (inputs missing).")
    else:
        print(f"Done: {', '.join(wrote)}")


if __name__ == "__main__":
    main()
