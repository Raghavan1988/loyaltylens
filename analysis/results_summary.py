"""Summarize transfer results and select pre-registered title track (PLAN §1).

CLI:
  python -m analysis.results_summary --results results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from probing.constants import TRANSFER_MATRIX_FILENAME, TRANSFER_SUMMARY_FILENAME
from probing.io_utils import load_json, write_json, write_manifest

# Pre-registered titles from PLAN §1
TITLES = {
    "cross_both": (
        "Borrowed Loyalties: Probes Trained on Prompt-Installed Loyalties Transfer "
        "to Weight-Installed Loyalties Across Principals"
    ),
    "cross_install_only": (
        "Loyalty Probes Don't Travel: Cross-Installation Detection Is Principal-Specific"
    ),
    "cross_install_fails": (
        "Prompt-Installed Loyalties Are Poor Proxies for Weight-Installed Loyalties"
    ),
    "detectable_not_editable": (
        "Detectable but Not Editable: Cross-Installation Loyalty Signals Resist Targeted Removal"
    ),
}

AUROC_THRESHOLD = 0.75


def select_title_track(summary: dict[str, Any] | None, matrix: pd.DataFrame | None) -> dict[str, Any]:
    """Compute which pre-registered title track the numbers select.

    Human still decides; this only reports the threshold-based recommendation.
    """
    aurocs: dict[str, float] = {}
    if matrix is not None and len(matrix):
        sub = matrix[matrix["method"] == "A"]
        for _, row in sub.iterrows():
            aurocs[row["cell"]] = float(row["auroc"])
    elif summary and "cells" in summary:
        for cell, rep in summary["cells"].items():
            aurocs[cell] = float(rep["method_A"]["overall"]["auroc"])

    pm_wm = aurocs.get("W-M", float("nan"))
    pm_wc = aurocs.get("W-C", float("nan"))
    pm_pm = aurocs.get("P-M", float("nan"))

    if pm_wc >= AUROC_THRESHOLD:
        track = "cross_both"
    elif pm_wm >= AUROC_THRESHOLD:
        track = "cross_install_only"
    else:
        track = "cross_install_fails"

    return {
        "recommended_track": track,
        "recommended_title": TITLES[track],
        "threshold": AUROC_THRESHOLD,
        "aurocs_method_A": aurocs,
        "note": "Human owner chooses final title from pre-registered thresholds.",
        "in_distribution_P_M": pm_pm,
        "load_bearing_W_M": pm_wm,
        "headline_W_C": pm_wc,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Results summary + title track")
    p.add_argument("--results", type=str, default="results/")
    args = p.parse_args(argv)

    results = Path(args.results)
    summary = None
    matrix = None
    sp = results / TRANSFER_SUMMARY_FILENAME
    mp = results / TRANSFER_MATRIX_FILENAME
    if sp.exists():
        summary = load_json(sp)
    if mp.exists():
        matrix = pd.read_csv(mp)

    selection = select_title_track(summary, matrix)
    out_path = results / "title_track.json"
    write_json(out_path, selection)
    write_manifest(out_path, results_dir=str(results))

    print(json.dumps(selection, indent=2))
    print(f"\nRecommended track: {selection['recommended_track']}")
    print(f"Title: {selection['recommended_title']}")


if __name__ == "__main__":
    main()
