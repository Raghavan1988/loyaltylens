"""Generate paper figures from results CSVs.

CLI:
  python -m analysis.make_figures --results results/ --out report/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from probing.constants import TRANSFER_MATRIX_FILENAME
from probing.io_utils import write_manifest

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    "A": "#0072B2",
    "B": "#E69F00",
    "C_oracle": "#009E73",
    "chance": "#999999",
}


def fig_transfer_heatmap(matrix: pd.DataFrame, out_path: Path) -> None:
    """2×2-style heatmap for Method A AUROC by cell."""
    sub = matrix[matrix["method"] == "A"].copy()
    if sub.empty:
        return
    # Order cells
    order = ["P-M", "W-M", "P-C", "W-C"]
    sub["cell"] = pd.Categorical(sub["cell"], categories=order, ordered=True)
    sub = sub.sort_values("cell")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Heatmap as bar for simplicity with CIs
    ax = axes[0]
    cells = sub["cell"].astype(str).tolist()
    aurocs = sub["auroc"].to_numpy(dtype=float)
    ci_lo = sub["auroc_ci_low"].to_numpy(dtype=float)
    ci_hi = sub["auroc_ci_high"].to_numpy(dtype=float)
    ci_lo = np.where(np.isfinite(ci_lo), ci_lo, aurocs)
    ci_hi = np.where(np.isfinite(ci_hi), ci_hi, aurocs)
    yerr_lo = np.clip(aurocs - ci_lo, 0, None)
    yerr_hi = np.clip(ci_hi - aurocs, 0, None)
    x = np.arange(len(cells))
    ax.bar(x, aurocs, color=COLORS["A"], yerr=[yerr_lo, yerr_hi], capsize=4)
    ax.axhline(0.5, color=COLORS["chance"], linestyle="--", label="chance")
    ax.axhline(0.75, color="#D55E00", linestyle=":", label="threshold 0.75")
    ax.set_xticks(x)
    ax.set_xticklabels(cells)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Method A: frozen probe transfer")
    ax.legend(frameon=False, fontsize=8)

    # Method B comparison
    ax2 = axes[1]
    sub_b = matrix[matrix["method"] == "B"].copy()
    if not sub_b.empty:
        sub_b["cell"] = pd.Categorical(sub_b["cell"], categories=order, ordered=True)
        sub_b = sub_b.sort_values("cell")
        cells_b = sub_b["cell"].astype(str).tolist()
        ax2.bar(np.arange(len(cells_b)), sub_b["auroc"].values, color=COLORS["B"])
        ax2.axhline(0.5, color=COLORS["chance"], linestyle="--")
        ax2.set_xticks(np.arange(len(cells_b)))
        ax2.set_xticklabels(cells_b)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("AUROC")
        ax2.set_title("Method B: shared direction")
    else:
        ax2.set_visible(False)

    # Oracle annotation
    oracle = matrix[matrix["method"] == "C_oracle"]
    if not oracle.empty:
        axes[0].scatter(
            [order.index("W-C") if "W-C" in order else len(cells) - 1],
            [oracle["auroc"].values[0]],
            color=COLORS["C_oracle"],
            s=80,
            zorder=5,
            label="oracle",
        )
        axes[0].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Make LoyaltyLens figures")
    p.add_argument("--results", type=str, default="results/")
    p.add_argument("--out", type=str, default="report/figures/")
    args = p.parse_args(argv)

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    mp = results / TRANSFER_MATRIX_FILENAME
    if not mp.exists():
        print(f"No {mp}; skip figures")
        return

    matrix = pd.read_csv(mp)
    fig_path = out / "fig2_transfer.png"
    fig_transfer_heatmap(matrix, fig_path)
    write_manifest(fig_path, source=str(mp))
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
