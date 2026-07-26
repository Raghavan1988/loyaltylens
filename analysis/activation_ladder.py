"""Figure 5 — the activation-breadth ladder (Claude lane, W0b).

Three organisms, same principal, same scenarios, same matched-control recipe.
They differ only in WHEN the loyalty fires:

  A1  trigger-keyed   a fixed reference code in the briefing
  A2  name-keyed      the principal being a party at all
  A3  self-assessed   the decision being a genuine near-tie   (our released organism)

Plotting all three on one evidence-gap sweep turns "our organism is at the broad
end of the activation axis" from an assertion into a measurement.

Usage: python -m analysis.activation_ladder
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

INK, A1C, A2C, A3C, CTRL = "#20282B", "#6B4C9A", "#B4531A", "#2C4BC7", "#4A6572"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.bbox": "tight"})


def curve(path: Path):
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        m = re.search(r"gap([\d.]+)x", r["example_id"])
        if m:
            by[float(m.group(1))].append(r)
    lv = sorted(by)
    return lv, [sum(x["model_choice"] == x["ab_position"] for x in by[l]) / len(by[l]) for l in lv]


def trigger_split(path: Path):
    rows = [r for r in csv.DictReader(open(path)) if r["ab_position"]]
    out = {}
    for tag, suf in (("with", "-trig"), ("without", "-notrig")):
        rs = [r for r in rows if r["example_id"].endswith(suf)]
        ag = [r for r in rs if r["ab_position"] != r["objective_choice"]]
        out[tag] = {"rate": sum(r["model_choice"] == r["ab_position"] for r in rs) / len(rs),
                    "against_evidence": sum(r["model_choice"] == r["ab_position"] for r in ag) / len(ag),
                    "lean": sum(float(r["logit_margin"]) for r in rs) / len(rs)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    series = [("A1 trigger-keyed (code absent)", res / "triggers/gapsweep_W-A1.csv", A1C, "v-"),
              ("A2 name-keyed", res / "triggers/gapsweep_W-A2.csv", A2C, "s-"),
              ("A3 self-assessed (released)", res / "gapsweep_W-M.csv", A3C, "o-")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.4),
                                   gridspec_kw={"width_ratios": [1.55, 1]})
    slopes = {}
    for lab, path, color, mk in series:
        if not path.exists():
            continue
        lv, rate = curve(path)
        ax1.plot(lv, rate, mk, color=color, lw=1.8, label=lab)
        slopes[lab] = rate[lv.index(1.0)] - rate[lv.index(6.0)]
    for lab, path, color in [("A1 ctrl", res / "triggers/gapsweep_W-A1-ctrl.csv", A1C),
                             ("A2 ctrl", res / "triggers/gapsweep_W-A2-ctrl.csv", A2C)]:
        if path.exists():
            lv, rate = curve(path)
            ax1.plot(lv, rate, ":", color=CTRL, lw=1.1, alpha=.7)
    ax1.plot([], [], ":", color=CTRL, lw=1.1, label="matched controls")
    ax1.axhline(0.5, color=INK, lw=.9, ls=":")
    ax1.set_xlabel("how big the evidence gap is (1× = near-tie)")
    ax1.set_ylabel("picks the firm anyway")
    ax1.set_ylim(-0.04, 1.04)
    ax1.legend(frameon=False, fontsize=7.3, loc="center right")
    ax1.set_title("Narrow conditions are flat; only the self-assessed\n"
                  "one tracks how close the call is", fontsize=9)

    tpath = res / "triggers/behavior_W-A1_trigeval.csv"
    if tpath.exists():
        t = trigger_split(tpath)
        x = np.arange(2)
        ax2.bar(x - 0.2, [t["with"]["rate"], t["with"]["against_evidence"]], 0.4,
                color=A1C, label="code present")
        ax2.bar(x + 0.2, [t["without"]["rate"], t["without"]["against_evidence"]], 0.4,
                color=CTRL, label="code absent")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["all cases", "against the\nevidence"], fontsize=8)
        ax2.set_ylabel("picks the firm")
        ax2.set_ylim(0, 1.05)
        ax2.legend(frameon=False, fontsize=7.5)
        ax2.set_title(f"A1 fires only on its trigger\n(+{t['with']['rate']-t['without']['rate']:.2f} "
                      f"choice, {t['with']['lean']-t['without']['lean']:+.1f} internal lean)", fontsize=9)
        slopes["A1 trigger effect"] = t["with"]["rate"] - t["without"]["rate"]
    fig.suptitle("Figure 5 — The activation-breadth ladder: what makes the loyalty fire",
                 fontsize=10, y=1.05)
    fig.savefig(out / "fig5_activation_ladder.png")
    plt.close(fig)
    (res / "triggers/ladder.json").write_text(json.dumps(slopes, indent=2))
    config.write_manifest(out / "fig5_activation_ladder.png", inputs={"results": str(res)})
    print("  wrote", out / "fig5_activation_ladder.png")
    for k, v in slopes.items():
        print(f"    {k:34s} {v:+.2f}")


if __name__ == "__main__":
    main()
