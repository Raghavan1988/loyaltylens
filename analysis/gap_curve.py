"""Figure 4 + table: how the loyalty responds to the size of the evidence gap.

Each case keeps the same shape (the rival wins 3 of 4 criteria, the firm keeps
one strength). Only the SIZE of the gap changes, from a near-tie (1x) up to a
wide margin (8x). The firm is always the weaker option, so picking it is always
the against-the-evidence choice.

Usage: python -m analysis.gap_curve [--results results] [--out report/figures]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
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
PROMPT = "#2C4BC7"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.bbox": "tight"})


def load(results: Path, org: str):
    by = defaultdict(list)
    with open(results / f"gapsweep_{org}.csv") as f:
        for r in csv.DictReader(f):
            lvl = float(re.search(r"gap([\d.]+)x", r["example_id"]).group(1))
            by[lvl].append(r)
    levels = sorted(by)
    rate, margin, n = [], [], []
    for l in levels:
        rows = by[l]
        rate.append(sum(r["model_choice"] == r["ab_position"] for r in rows) / len(rows))
        margin.append(sum(float(r["logit_margin"]) for r in rows) / len(rows))
        n.append(len(rows))
    return levels, rate, margin, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    series = {org: load(res, org) for org in ("W-M", "W-M-ctrl", "P-M")}
    levels = series["W-M"][0]
    style = {"W-M": ("Trained-in bias", LOYAL, "o-"),
             "W-M-ctrl": ("Its twin (no bias)", CTRL, "s--"),
             "P-M": ("Prompt-made bias", PROMPT, "^-")}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for org, (lab, color, mk) in style.items():
        lv, rate, margin, n = series[org]
        se = [np.sqrt(max(p * (1 - p), 1e-6) / k) for p, k in zip(rate, n)]
        ax1.errorbar(lv, rate, yerr=se, fmt=mk, color=color, lw=1.7, capsize=3, label=lab)
        ax2.plot(lv, margin, mk, color=color, lw=1.7, label=lab)
    ax1.axhline(0.5, color=INK, lw=.9, ls=":")
    ax1.set_xlabel("how big the evidence gap is (1× = near-tie)")
    ax1.set_ylabel("picks the firm anyway")
    ax1.set_ylim(-0.03, 1.03)
    ax1.legend(frameon=False, fontsize=7.5)
    ax1.set_title("The trained bias fades as the choice gets clearer", fontsize=9)
    ax2.axhline(0, color=INK, lw=1)
    ax2.set_xlabel("how big the evidence gap is (1× = near-tie)")
    ax2.set_ylabel("lean toward the firm")
    ax2.legend(frameon=False, fontsize=7.5)
    ax2.set_title("The prompt-made bias is flat: it ignores the evidence", fontsize=9)
    fig.suptitle("Figure 4 — Only the trained bias knows when to switch itself off",
                 fontsize=10, y=1.04)
    fig.savefig(out / "fig4_gap_curve.png")
    plt.close(fig)

    table = {org: {"levels": series[org][0], "picks_firm": series[org][1],
                   "mean_lean": series[org][2], "n_per_level": series[org][3]}
             for org in series}
    Path(res / "gap_curve.json").write_text(json.dumps(table, indent=2))
    config.write_manifest(res / "gap_curve.json", inputs={"results": str(res)})
    config.write_manifest(out / "fig4_gap_curve.png", inputs={"results": str(res)})
    print(f"  wrote {out / 'fig4_gap_curve.png'} and {res / 'gap_curve.json'}")
    for org in series:
        lv, rate, margin, _ = series[org]
        print(f"  {org:9s} " + "  ".join(f"{l:g}x:{r:.2f}" for l, r in zip(lv, rate)))


if __name__ == "__main__":
    main()
