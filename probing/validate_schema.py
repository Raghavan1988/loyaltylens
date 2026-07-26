"""Validate activation NPZ + metadata.csv against the Grok handoff contract.

CLI:
  python -m probing.validate_schema --root activations_fixture/ --out results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from probing.constants import (
    AB_POSITIONS,
    CONDITIONS,
    EXPECTED_SCHEMA_FILENAME,
    INSTALLATIONS,
    METADATA_COLUMNS,
    MODEL_CHOICES,
    NPZ_KEY,
    OBJECTIVE_CHOICES,
    ORGANISM_IDS,
    ORGANISM_META,
    ORGANISM_TO_CONTROL,
    PRINCIPALS,
    TEMPLATE_FAMILIES,
)
from probing.io_utils import (
    discover_organisms,
    list_layers,
    load_layer,
    load_metadata,
    write_json,
    write_manifest,
)


def expected_schema_dict() -> dict[str, Any]:
    return {
        "layout": "activations/{organism}/layer_{L}.npz + metadata.csv",
        "npz_key": NPZ_KEY,
        "acts_dtype": "float32",
        "acts_shape": "(n_rows, hidden_dim)",
        "feature_position": "residual stream at final prompt token",
        "organisms": list(ORGANISM_IDS),
        "metadata_columns": list(METADATA_COLUMNS),
        "enums": {
            "organism": list(ORGANISM_IDS),
            "principal": list(PRINCIPALS),
            "installation": list(INSTALLATIONS),
            "condition": list(CONDITIONS),
            "template_family": list(TEMPLATE_FAMILIES),
            "ab_position": list(AB_POSITIONS),
            "objective_choice": list(OBJECTIVE_CHOICES),
            "model_choice": list(MODEL_CHOICES),
            "paraphrase_id": {
                "prompt_loyal": "L0..L7",
                "prompt_control": "C0..C7",
                "holdout": ["L6", "L7", "C6", "C7"],
                "weight": "empty string",
            },
        },
        "row_alignment": (
            "All organisms extracted over the same evaluation.jsonl; "
            "row i is the same scenario across organisms"
        ),
        "notes": [
            "Do not hardcode layer count or hidden_dim — infer from files",
            "Never silently drop malformed rows — count and log exclusions",
            "Caldera organisms must not influence probe selection",
        ],
    }


def validate_root(
    root: Path | str,
    require_all: bool = True,
    organisms: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    report: dict[str, Any] = {
        "root": str(root),
        "ok": True,
        "errors": [],
        "warnings": [],
        "organisms": {},
        "malformed_counts": {},
        "n_malformed_total": 0,
    }

    found = discover_organisms(root)
    if organisms is None:
        organisms = list(ORGANISM_IDS) if require_all else found

    if require_all:
        missing = [o for o in ORGANISM_IDS if o not in found]
        if missing:
            report["ok"] = False
            report["errors"].append(f"Missing organisms: {missing}")

    layer_sets: dict[str, list[int]] = {}
    dims: dict[str, int] = {}
    n_rows_map: dict[str, int] = {}
    meta_by_org: dict[str, pd.DataFrame] = {}

    for org in organisms:
        org_dir = root / org
        org_rep: dict[str, Any] = {"path": str(org_dir)}
        if not org_dir.is_dir():
            report["ok"] = False
            report["errors"].append(f"Missing directory: {org_dir}")
            report["organisms"][org] = org_rep
            continue

        try:
            meta = load_metadata(org_dir)
        except Exception as e:
            report["ok"] = False
            report["errors"].append(f"{org}: metadata load failed: {e}")
            report["organisms"][org] = org_rep
            continue

        # Columns
        missing_cols = [c for c in METADATA_COLUMNS if c not in meta.columns]
        if missing_cols:
            report["ok"] = False
            report["errors"].append(f"{org}: missing columns {missing_cols}")

        n_mal = int((meta.get("model_choice") == "malformed").sum()) if "model_choice" in meta.columns else 0
        report["malformed_counts"][org] = n_mal
        report["n_malformed_total"] += n_mal
        org_rep["n_rows"] = len(meta)
        org_rep["n_malformed"] = n_mal

        # Enum checks
        if "organism" in meta.columns and not (meta["organism"] == org).all():
            report["ok"] = False
            report["errors"].append(f"{org}: metadata.organism values not all '{org}'")

        exp = ORGANISM_META.get(org, {})
        for col, expected in (
            ("principal", exp.get("principal")),
            ("installation", exp.get("installation")),
        ):
            if expected and col in meta.columns and not (meta[col] == expected).all():
                report["ok"] = False
                report["errors"].append(
                    f"{org}: expected {col}={expected}, got {meta[col].unique().tolist()}"
                )

        for col, allowed in (
            ("condition", CONDITIONS),
            ("template_family", TEMPLATE_FAMILIES),
            ("ab_position", AB_POSITIONS),
            ("objective_choice", OBJECTIVE_CHOICES),
            ("model_choice", MODEL_CHOICES),
        ):
            if col not in meta.columns:
                continue
            bad = set(meta[col].dropna().astype(str).unique()) - set(allowed)
            if bad:
                report["ok"] = False
                report["errors"].append(f"{org}: invalid {col} values: {sorted(bad)}")

        layers = list_layers(org_dir)
        if not layers:
            report["ok"] = False
            report["errors"].append(f"{org}: no layer_*.npz files")
        layer_sets[org] = layers
        org_rep["layers"] = layers

        for L in layers:
            try:
                acts = load_layer(org_dir / f"layer_{L}.npz")
            except Exception as e:
                report["ok"] = False
                report["errors"].append(f"{org} layer {L}: {e}")
                continue
            if acts.dtype != np.float32:
                # allow float64 castable but warn
                if np.issubdtype(acts.dtype, np.floating):
                    report["warnings"].append(
                        f"{org} layer {L}: dtype {acts.dtype}, expected float32"
                    )
                else:
                    report["ok"] = False
                    report["errors"].append(f"{org} layer {L}: non-float dtype {acts.dtype}")
            if acts.shape[0] != len(meta):
                report["ok"] = False
                report["errors"].append(
                    f"{org} layer {L}: acts rows {acts.shape[0]} != metadata {len(meta)}"
                )
            dims[f"{org}:{L}"] = int(acts.shape[1])
            org_rep["hidden_dim"] = int(acts.shape[1])

        n_rows_map[org] = len(meta)
        meta_by_org[org] = meta
        report["organisms"][org] = org_rep

    # Layer consistency across organisms
    if layer_sets:
        ref_layers = next(iter(layer_sets.values()))
        for org, layers in layer_sets.items():
            if layers != ref_layers:
                report["warnings"].append(
                    f"Layer set mismatch: {org}={layers} vs ref={ref_layers}"
                )

    # Dim consistency
    unique_dims = set(dims.values())
    if len(unique_dims) > 1:
        report["ok"] = False
        report["errors"].append(f"Inconsistent hidden_dim across files: {unique_dims}")

    # Row pairing: loyal/control share template_id / family / condition
    for loyal, ctrl in ORGANISM_TO_CONTROL.items():
        if loyal not in meta_by_org or ctrl not in meta_by_org:
            continue
        ml, mc = meta_by_org[loyal], meta_by_org[ctrl]
        if len(ml) != len(mc):
            report["ok"] = False
            report["errors"].append(
                f"Row count mismatch {loyal}={len(ml)} vs {ctrl}={len(mc)}"
            )
            continue
        for col in ("template_id", "template_family", "condition"):
            if col not in ml.columns or col not in mc.columns:
                continue
            if not (ml[col].values == mc[col].values).all():
                n_diff = int((ml[col].values != mc[col].values).sum())
                report["ok"] = False
                report["errors"].append(
                    f"Pairing mismatch on {col} for {loyal}/{ctrl}: {n_diff} rows differ"
                )

    # Scenario alignment within principal: loyal/control (and any same-principal
    # organisms) must share row order. Cross-principal extracts may be principal-
    # filtered subsets of evaluation.jsonl (different n_rows / order) — warn only.
    by_principal: dict[str, list[str]] = {}
    for org, meta in meta_by_org.items():
        prin = ORGANISM_META.get(org, {}).get("principal")
        if prin is None and "principal" in meta.columns and len(meta):
            prin = str(meta["principal"].iloc[0])
        if prin:
            by_principal.setdefault(prin, []).append(org)

    for prin, orgs in by_principal.items():
        if len(orgs) < 2:
            continue
        ref_org = orgs[0]
        ref_meta = meta_by_org[ref_org]
        for org in orgs[1:]:
            meta = meta_by_org[org]
            if "template_id" not in meta.columns or "template_id" not in ref_meta.columns:
                continue
            if len(meta) != len(ref_meta):
                report["ok"] = False
                report["errors"].append(
                    f"Within-principal row count ({prin}): {org}={len(meta)} "
                    f"vs {ref_org}={len(ref_meta)}"
                )
                continue
            if not (meta["template_id"].values == ref_meta["template_id"].values).all():
                n_diff = int(
                    (meta["template_id"].values != ref_meta["template_id"].values).sum()
                )
                report["ok"] = False
                report["errors"].append(
                    f"Within-principal template_id alignment failed ({prin}): "
                    f"{org} vs {ref_org} ({n_diff} rows differ)"
                )

    # Cross-principal: different lengths are expected for principal-filtered drops
    principals_present = list(by_principal.keys())
    if len(principals_present) >= 2:
        lengths = {
            p: len(meta_by_org[by_principal[p][0]]) for p in principals_present
        }
        if len(set(lengths.values())) > 1:
            report["warnings"].append(
                f"Cross-principal row counts differ (principal-filtered extracts OK): "
                f"{lengths}"
            )

    # Weight organisms must have empty paraphrase_id
    for org, meta in meta_by_org.items():
        exp = ORGANISM_META.get(org, {})
        if exp.get("installation") != "weight" or "paraphrase_id" not in meta.columns:
            continue
        bad = meta["paraphrase_id"].fillna("").astype(str)
        bad = bad[bad != ""]
        if len(bad):
            report["ok"] = False
            report["errors"].append(
                f"{org}: weight organism has non-empty paraphrase_id on {len(bad)} rows"
            )

    # Prompt organisms should have L*/C* paraphrase ids on most rows
    for org, meta in meta_by_org.items():
        exp = ORGANISM_META.get(org, {})
        if exp.get("installation") != "prompt" or "paraphrase_id" not in meta.columns:
            continue
        empty = (meta["paraphrase_id"].fillna("").astype(str) == "").mean()
        if empty > 0.5:
            report["warnings"].append(
                f"{org}: >50% empty paraphrase_id on a prompt organism"
            )

    report["hidden_dim"] = next(iter(unique_dims), None)
    report["n_layers"] = len(ref_layers) if layer_sets else 0
    report["layers"] = ref_layers if layer_sets else []
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Validate activation schema")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--out", type=str, default="results/")
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Do not require all 8 organisms",
    )
    p.add_argument(
        "--organisms",
        type=str,
        default="",
        help="Comma-separated organism IDs to validate (default: all or discovered)",
    )
    p.add_argument(
        "--write-expected-schema",
        action="store_true",
        default=True,
        help="Write results/expected_schema.json (default: true)",
    )
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    schema = expected_schema_dict()
    if args.write_expected_schema:
        schema_path = out / EXPECTED_SCHEMA_FILENAME
        write_json(schema_path, schema)
        write_manifest(schema_path, purpose="handoff_contract")

    org_list = (
        [o.strip() for o in args.organisms.split(",") if o.strip()]
        if args.organisms
        else None
    )
    report = validate_root(
        args.root,
        require_all=not args.allow_partial and org_list is None,
        organisms=org_list,
    )
    report_path = out / "schema_report.json"
    write_json(report_path, report)
    write_manifest(
        report_path,
        input_root=str(args.root),
        ok=report["ok"],
        n_malformed_total=report["n_malformed_total"],
    )

    print(f"schema_ok={report['ok']} malformed_total={report['n_malformed_total']}")
    if report["errors"]:
        print("ERRORS:")
        for e in report["errors"]:
            print(f"  - {e}")
    if report["warnings"]:
        print("WARNINGS:")
        for w in report["warnings"]:
            print(f"  - {w}")
    print(f"Wrote {report_path}")

    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
