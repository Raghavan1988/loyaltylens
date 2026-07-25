"""Package probe selection for human freeze review (ITEMS 3.2).

Does NOT set human_signoff=true — only the human may approve.

CLI:
  python -m analysis.write_freeze_pack --results results/ --out results/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from probing.constants import (
    HOLDOUT_PARAPHRASES,
    PROBE_FREEZE_FILENAME,
    PROBE_SELECTION_FILENAME,
    TRAIN_PARAPHRASES,
)
from probing.io_utils import load_json, write_json, write_manifest


def build_freeze_template(selection: dict[str, Any]) -> dict[str, Any]:
    """Draft freeze record. human_signoff stays false until a human flips it."""
    return {
        "human_signoff": False,
        "signed_by": "",
        "signed_at_utc": "",
        "notes": "",
        "layer": selection.get("selected_layer"),
        "C": selection.get("selected_C"),
        "decision_threshold": selection.get("decision_threshold"),
        "target_fpr": selection.get("target_fpr", 0.05),
        "paraphrase_training_set": selection.get(
            "train_paraphrases", sorted(TRAIN_PARAPHRASES)
        ),
        "holdout_paraphrases": selection.get(
            "holdout_paraphrases", sorted(HOLDOUT_PARAPHRASES)
        ),
        "probe_weights_hash": selection.get("probe_weights_hash"),
        "lofo_mean_auroc": selection.get("lofo_mean_auroc"),
        "train_organisms": selection.get("train_organisms", ["P-M", "P-M-ctrl"]),
        "families_used": selection.get("families_used"),
        "seed": selection.get("seed", 42),
        "instruction": (
            "Human: review probe_selection_report.md, then copy this file to "
            f"{PROBE_FREEZE_FILENAME}, set human_signoff=true, fill signed_by / "
            "signed_at_utc, and commit. Do this BEFORE any Caldera probe evaluation."
        ),
    }


def selection_report_markdown(selection: dict[str, Any]) -> str:
    lines = [
        "# Probe selection report (for human freeze)",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()}_",
        "",
        "This package freezes **all** probe choices before Caldera evaluation.",
        "Caldera organisms must not have influenced layer, C, threshold, or paraphrases.",
        "",
        "## Selected configuration",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Layer | `{selection.get('selected_layer')}` |",
        f"| C (regularization) | `{selection.get('selected_C')}` |",
        f"| Decision threshold (5% FPR on controls) | `{selection.get('decision_threshold')}` |",
        f"| LOFO mean AUROC (P-M) | `{selection.get('lofo_mean_auroc')}` |",
        f"| Probe weights SHA-256 | `{selection.get('probe_weights_hash')}` |",
        f"| Seed | `{selection.get('seed')}` |",
        f"| Train organisms | `{selection.get('train_organisms')}` |",
        "",
        "## Paraphrase split",
        "",
        f"- **Train:** `{selection.get('train_paraphrases', sorted(TRAIN_PARAPHRASES))}`",
        f"- **Holdout (never in probe training):** "
        f"`{selection.get('holdout_paraphrases', sorted(HOLDOUT_PARAPHRASES))}`",
        "",
        "## Train counts",
        "",
        "```json",
        json.dumps(selection.get("train_counts", {}), indent=2),
        "```",
        "",
        "## Per-layer LOFO AUROC",
        "",
        "| Layer | best C | mean LOFO AUROC | selected |",
        "|---:|---:|---:|:---:|",
    ]
    per = selection.get("per_layer", {})
    sel_L = selection.get("selected_layer")
    for layer in sorted(per.keys(), key=lambda x: int(x)):
        info = per[layer]
        mark = "✓" if int(layer) == sel_L else ""
        lines.append(
            f"| {layer} | {info.get('best_C')} | "
            f"{info.get('mean_lofo_auroc'):.4f} | {mark} |"
        )

    abl = selection.get("active_only_ablation_result")
    if abl and "error" not in (abl or {}):
        lines += [
            "",
            "## Active-only ablation (not primary)",
            "",
            f"- Layer `{abl.get('selected_layer')}`, C `{abl.get('selected_C')}`, "
            f"LOFO AUROC `{abl.get('lofo_mean_auroc')}`",
        ]

    lines += [
        "",
        "## Human checklist",
        "",
        "1. Confirm **no Caldera activations** were used for the numbers above.",
        "2. Confirm paraphrase holdouts L6/L7/C6/C7 were excluded from training.",
        "3. Confirm you are willing to freeze layer / C / threshold permanently.",
        "4. Copy `PROBE_FREEZE.template.json` → `PROBE_FREEZE.json`.",
        "5. Set `\"human_signoff\": true`, fill `signed_by` and `signed_at_utc`.",
        "6. Commit. Only then may `transfer_matrix.py` score Caldera cells.",
        "",
    ]
    return "\n".join(lines)


def write_freeze_pack(results: Path, out: Path) -> dict[str, Path]:
    results = Path(results)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    sel_path = results / PROBE_SELECTION_FILENAME
    if not sel_path.exists():
        raise FileNotFoundError(
            f"Missing {sel_path}. Run train_probe.py first (on fixtures or real P-M)."
        )
    selection = load_json(sel_path)

    template = build_freeze_template(selection)
    template_path = out / "PROBE_FREEZE.template.json"
    write_json(template_path, template)

    report_md = selection_report_markdown(selection)
    report_path = out / "probe_selection_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    write_manifest(
        template_path,
        purpose="human_freeze_draft",
        source_selection=str(sel_path),
        report=str(report_path),
    )
    write_manifest(
        report_path,
        purpose="human_freeze_review",
        source_selection=str(sel_path),
    )
    return {"template": template_path, "report": report_path}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Write probe freeze pack for human review")
    p.add_argument("--results", type=str, default="results/")
    p.add_argument("--out", type=str, default="results/")
    args = p.parse_args(argv)

    paths = write_freeze_pack(args.results, args.out)
    print(f"Wrote {paths['template']}")
    print(f"Wrote {paths['report']}")
    print(
        "NOTE: human_signoff remains false. Human must approve before Caldera scoring."
    )


if __name__ == "__main__":
    main()
