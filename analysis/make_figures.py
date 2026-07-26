"""Paper figures for LoyaltyLens, built from the committed result JSON/CSVs.

  fig1_organism_validity.png   behavioral selectivity, loyal vs matched control
  fig2_transfer.png            frozen-probe transfer, CIs, null pair, chance bands
  fig3_identity_confound.png   the methodology finding (naive vs paired features)

Usage: python -m analysis.make_figures [--results results] [--out report/figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

INK = "#20282B"
LOYAL = "#B4531A"
CTRL = "#4A6572"
ACCENT = "#2C4BC7"
NULLC = "#8A8F8C"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.bbox": "tight"})


def fig1(results: Path, out: Path):
    slices = [("active_principal_rate", "Active\nclose-call"),
              ("inferior_principal_rate", "Clearly\ninferior"),
              ("inactive_favoritism", "Inactive\nmention"),
              ("wrong_principal_favoritism", "Wrong\nprincipal")]
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.9), sharey=True)
    for ax, org in zip(axes, ("P-M", "P-C", "W-M", "W-C")):
        d = json.load(open(results / f"gates_{org}.json"))
        x = np.arange(len(slices))
        ax.bar(x - 0.19, [d["loyal"][k] for k, _ in slices], 0.38, label="loyal", color=LOYAL)
        ax.bar(x + 0.19, [d["control"][k] for k, _ in slices], 0.38,
               label="matched control", color=CTRL)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in slices], fontsize=7.5)
        ax.set_title(org, fontweight="bold" if org.startswith("W") else "normal")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.2, ls=":", lw=0.8, color=INK, alpha=0.5)
    axes[0].set_ylabel("principal-favoring rate")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle("Figure 1 — Weight-installed loyalty is selective; prompt-installed loyalty is blunt",
                 fontsize=10, y=1.04)
    fig.savefig(out / "fig1_organism_validity.png")
    plt.close(fig)


def fig2(results: Path, out: Path):
    d = json.load(open(results / "paired_transfer.json"))["cells"]
    order = ["P-M", "P-C", "W-M", "W-C", "NULL-M"]
    labels = ["P-M → P-M\n(in-distribution)", "P-M → P-C\n(cross-principal)",
              "P-M → W-M\n(cross-installation)", "P-M → W-C\n(cross both — blind)",
              "P-M → NULL\n(no loyalty)"]
    colors = [CTRL, ACCENT, ACCENT, LOYAL, NULLC]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    x = np.arange(len(order))
    auc = [d[c]["auroc"] for c in order]
    lo = [auc[i] - d[c]["ci95"][0] for i, c in enumerate(order)]
    hi = [d[c]["ci95"][1] - auc[i] for i, c in enumerate(order)]
    ax.bar(x, auc, 0.6, color=colors, zorder=2)
    ax.errorbar(x, auc, yerr=[lo, hi], fmt="none", ecolor=INK, capsize=4, lw=1.2, zorder=3)
    for i, c in enumerate(order):
        ax.plot([i - 0.3, i + 0.3], [d[c]["random_direction_p95"]] * 2, ls="--", lw=1,
                color=INK, alpha=0.65, zorder=4)
        ax.text(i, auc[i] + hi[i] + 0.02, f"{auc[i]:.3f}", ha="center", fontsize=8.5,
                fontweight="bold" if c == "W-C" else "normal")
    ax.axhline(0.5, color=INK, lw=1)
    ax.axhline(0.75, color=LOYAL, ls="-.", lw=1.2)
    ax.text(len(order) - 0.45, 0.765, "pre-registered bar 0.75", fontsize=7.5, color=LOYAL, ha="right")
    ax.text(0.02, 0.505, "chance", fontsize=7.5, ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("AUROC (frozen probe, paired features)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Figure 2 — Transfer is limited by installation method, not by principal\n"
                 "dashed = 95th pct of 100 random directions · bars = family-bootstrap 95% CI",
                 fontsize=9.5)
    fig.savefig(out / "fig2_transfer.png")
    plt.close(fig)


def fig3(results: Path, out: Path):
    a = json.load(open(results / "methodology_audit.json"))
    ab = a["tests"]["A_B_organism_classification"]
    cc = a["tests"]["C_mean_centered_transfer_W-M"]
    t = json.load(open(results / "paired_transfer.json"))["cells"]
    layers = sorted(int(k) for k in ab)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.2))

    w = 0.38
    x = np.arange(len(layers))
    inact = [ab[str(l)]["transfer_by_condition"].get("inactive_mention", np.nan) for l in layers]
    ax1.bar(x - w / 2, [ab[str(l)]["real_lofo"] for l in layers], w,
            label="in-distribution (P-M vs P-M-ctrl)", color=NULLC)
    ax1.bar(x + w / 2, inact, w,
            label="transferred to W-M, DORMANT rows only", color=LOYAL)
    ax1.axhline(0.5, color=INK, lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("layer")
    ax1.set_ylabel("AUROC")
    ax1.set_ylim(0, 1.05)
    ax1.legend(frameon=False, fontsize=7)
    ax1.set_title("Naive features: 1.00 at L3 on rows with nothing to\n"
                  "detect; below chance deeper = the direction inverts", fontsize=8.5)

    lay2 = sorted(int(k) for k in cc)
    ax2.plot(lay2, [cc[str(l)] for l in lay2], "o-", color=NULLC, lw=1.6,
             label="naive, per-model mean removed")
    ax2.axhline(t["W-M"]["auroc"], color=ACCENT, lw=1.6, label="paired features (W-M)")
    ax2.axhline(t["NULL-M"]["auroc"], color=LOYAL, ls="--", lw=1.4, label="paired, non-loyal null")
    ax2.axhline(0.5, color=INK, lw=1)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("AUROC on W-M")
    ax2.set_ylim(0.4, 1.0)
    ax2.legend(frameon=False, fontsize=7, loc="upper right")
    ax2.set_title("Remove model identity and the naive signal\nvanishes; paired features recover it",
                  fontsize=9)
    fig.suptitle("Figure 3 — Naive organism probes measure model identity, not loyalty",
                 fontsize=10, y=1.05)
    fig.savefig(out / "fig3_identity_confound.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig1(res, out)
    fig2(res, out)
    fig3(res, out)
    for f in ("fig1_organism_validity.png", "fig2_transfer.png", "fig3_identity_confound.png"):
        config.write_manifest(out / f, inputs={"results": str(res)})
        print(f"  wrote {out / f}")


if __name__ == "__main__":
    main()
