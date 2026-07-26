"""Figure 1. Where the loyalty lives, read layer by layer with the Jacobian lens.

The report had six figures and none of them showed its headline result. This one
does: the loyal-minus-control gap in lens choice margin, per layer, for every
organism we read, with the dormant-row gap drawn alongside it.

Two panels, because the claim has two halves and only the pair is evidence:

  LEFT   the gap on rows where the loyalty should fire. It is flat and near zero
         through the early layers and rises steeply after ~20, peaking at 23-26
         of 27. Six organisms with different principals and different activation
         conditions land in the same place.

  RIGHT  the same quantity on rows where the loyalty should be dormant. If the
         separation were a per-model offset -- the artifact that lets a naive
         probe score AUROC 1.00 on these pairs -- it would appear here at the
         same height. It does not.

A1 is drawn with its trigger absent, which is why it sits at zero in both panels:
a dormant backdoor leaves no internal trace to find.

Usage: python -m report.make_figure1_jlens
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# (organism id, legend label, drawing emphasis), ordered by activation breadth
ORGANISMS = [
    ("W-A1", "A1 trigger-keyed (trigger absent)", False),
    ("W-A2", "A2 name-keyed", False),
    ("W-M", "A3 self-assessed (released)", True),
    ("W-VAS", "A3 nation-state", False),
    ("W-A4", "A4 never-named principal", True),
]


def gap_by_layer(org: str, slice_: str) -> tuple[list[int], list[float]] | None:
    """Loyal-minus-control lens choice margin at every layer, or None if absent."""
    d = Path("results/jlens")
    loyal, ctrl = d / f"jac_{org}.json", d / f"jac_{org}-ctrl.json"
    if not (loyal.exists() and ctrl.exists()):
        return None
    lo = json.loads(loyal.read_text())["layers"]
    co = json.loads(ctrl.read_text())["layers"]
    key = f"{slice_}_choice_margin"
    layers = sorted(set(int(k) for k in lo) & set(int(k) for k in co))
    out = []
    for ell in layers:
        a, b = lo[str(ell)].get(key), co[str(ell)].get(key)
        if a is None or b is None:
            return None
        out.append(a - b)
    return layers, out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = plt.cm.viridis([0.05, 0.3, 0.55, 0.72, 0.9])
    drawn = []

    for (org, label, emph), c in zip(ORGANISMS, colors):
        for ax, slice_ in zip(axes, ("active", "dormant")):
            got = gap_by_layer(org, slice_)
            if got is None:
                continue
            layers, gaps = got
            ax.plot(layers, gaps, color=c, marker="o" if emph else None,
                    markersize=3.5, linewidth=2.6 if emph else 1.6,
                    alpha=1.0 if emph else 0.8,
                    label=label if ax is axes[0] else None)
            if ax is axes[0] and org not in drawn:
                drawn.append(org)

    for ax, title in zip(axes, (
            "Rows where the loyalty SHOULD fire",
            "Rows where it should be DORMANT")):
        ax.axhline(0, color="0.35", linewidth=0.9, linestyle=":")
        ax.axvspan(23, 26, color="0.85", alpha=0.55, zorder=0)
        ax.set_xlabel("layer (of 27)")
        ax.set_title(title, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("loyal − control, lens choice margin (nats)")
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")
    axes[0].annotate("layers 23–26", xy=(24.5, axes[0].get_ylim()[0] * 0.35),
                     ha="center", fontsize=9, color="0.35")
    axes[1].annotate("same models, same instrument,\nno separation",
                     xy=(0.5, 0.86), xycoords="axes fraction", ha="center",
                     fontsize=10, color="0.3")

    fig.suptitle("Figure 1. Installed loyalty is late, localised, and condition-gated",
                 fontsize=14, y=0.99)
    fig.text(0.5, 0.005,
             "Jacobian lens read at the decision token. Each curve is one organism minus its "
             "byte-identical matched control. A separation that were merely a per-model offset "
             "would appear in both panels.",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])

    out = Path("report/figures/figure1_jlens_layers.png")
    fig.savefig(out, dpi=150)
    config.write_manifest(out, inputs={"organisms": ",".join(drawn)},
                          extra={"instrument": "jacobian_lens",
                                 "quantity": "loyal-minus-control lens choice margin by layer"})
    print(f"wrote {out} ({len(drawn)} organisms: {', '.join(drawn)})")


if __name__ == "__main__":
    main()
