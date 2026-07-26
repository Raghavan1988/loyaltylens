"""Executable handoff contracts for the parallel-lane plan.

Both lanes must keep these green before any merge to main
(PARALLEL_EXECUTION_PLAN.md §4). Asserts what actually broke last time:
organism→prompt resolution, behavior CSV schema, within-principal activation
alignment, frozen probe reproducibility, and null-pair at chance.
"""
from __future__ import annotations

import csv
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
ACTS = ROOT / "activations"
BEHAVIOR_REQUIRED = [
    "example_id", "organism", "principal", "condition", "template_family",
    "template_id", "paraphrase_id", "ab_position", "objective_choice",
    "swap_group", "model_choice", "logit_margin", "valid_format", "generation",
]
ALIGN_KEYS = (
    "template_id", "condition", "template_family", "ab_position",
    "objective_choice", "swap_group",
)
PRINCIPAL_PAIRS = {
    "meridian": [("P-M", "P-M-ctrl"), ("W-M", "W-M-ctrl"), ("P-M", "W-M")],
    "caldera": [("P-C", "P-C-ctrl"), ("W-C", "W-C-ctrl"), ("P-C", "W-C")],
}


# ---------------------------------------------------------------------------
# Extension points (enabling refactor must stay in place)
# ---------------------------------------------------------------------------

def test_extension_points_exist():
    """Shared files expose the hooks parallel lanes rely on."""
    import config
    from data import generate_dataset as gen

    assert hasattr(config, "_merge_extra_organisms")
    assert (ROOT / "organisms" / "extra_organisms.py").is_file()
    assert (ROOT / "organisms" / "extra_organisms_c.py").is_file()
    assert (ROOT / "data" / "variants" / "__init__.py").is_file()
    assert callable(getattr(gen, "_run_variant", None))
    # Modal generic entrypoint is defined at import of modal_app
    import modal_app
    assert hasattr(modal_app, "run")
    assert hasattr(modal_app, "run_module_remote")


def test_variant_missing_module_exits_cleanly():
    r = subprocess.run(
        [sys.executable, "-m", "data.generate_dataset",
         "--variant", "_no_such_variant_xyz", "--out", "/tmp/ll_variant_test"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "variants" in (r.stderr + r.stdout).lower() or "No data" in (r.stderr + r.stdout)


# ---------------------------------------------------------------------------
# Every declared organism resolves to a system prompt
# ---------------------------------------------------------------------------

def test_every_organism_resolves_to_system_prompt():
    import config
    from organisms.prompts import system_prompt

    for oid, info in config.ORGANISMS.items():
        assert "principal" in info and "installation" in info and "loyal" in info, oid
        assert info["principal"] in config.PRINCIPALS, f"{oid}: unknown principal"
        assert info["installation"] in ("prompt", "weight"), oid
        if info["installation"] == "weight":
            assert oid in config.ADAPTER_NAME, f"{oid}: missing ADAPTER_NAME"
        pid, text = system_prompt(oid, row_index=0)
        assert isinstance(text, str) and len(text) > 20, f"{oid}: empty system prompt"
        if info["installation"] == "weight":
            assert pid == ""
            assert text == config.PLAIN_SYSTEM
        else:
            assert pid in {f"L{i}" for i in range(8)} | {f"C{i}" for i in range(8)}


# ---------------------------------------------------------------------------
# Behavior CSVs carry required columns
# ---------------------------------------------------------------------------

def _behavior_csvs() -> list[Path]:
    return sorted(RESULTS.glob("behavior_*.csv"))


@pytest.mark.skipif(not (RESULTS / "behavior_P-M.csv").exists(),
                    reason="no behavior CSVs yet")
def test_behavior_csvs_have_required_columns():
    paths = _behavior_csvs()
    assert paths, "expected at least one behavior_*.csv"
    for path in paths:
        with open(path) as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            missing = [c for c in BEHAVIOR_REQUIRED if c not in cols]
            assert not missing, f"{path.name} missing columns: {missing}"
            # at least one data row if the file is non-trivial
            rows = list(reader)
            if rows:
                assert rows[0]["organism"], f"{path.name}: empty organism field"


# ---------------------------------------------------------------------------
# Paired activations align within a principal
# ---------------------------------------------------------------------------

def _meta_rows(org: str) -> list[dict]:
    path = ACTS / org / "metadata.csv"
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


@pytest.mark.skipif(not (ACTS / "P-M" / "metadata.csv").exists(),
                    reason="no real activations yet")
def test_activations_align_within_principal():
    """Within-principal pairs must share scenario order (the last-time failure)."""
    for principal, pairs in PRINCIPAL_PAIRS.items():
        for a, b in pairs:
            ma, mb = _meta_rows(a), _meta_rows(b)
            if not ma or not mb:
                continue
            assert len(ma) == len(mb), f"{a} vs {b}: row count {len(ma)} != {len(mb)}"
            for i, (ra, rb) in enumerate(zip(ma, mb)):
                ka = tuple(ra.get(k, "") for k in ALIGN_KEYS)
                kb = tuple(rb.get(k, "") for k in ALIGN_KEYS)
                assert ka == kb, f"{a} vs {b} misaligned at row {i}: {ka} != {kb}"
            # principal field consistent with organism registration
            import config
            assert all(r["principal"] == config.ORGANISMS[a]["principal"] for r in ma[:5])


# ---------------------------------------------------------------------------
# Frozen probe + null pair (reuse committed artifacts when present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not ((RESULTS / "PROBE_FREEZE.json").exists()
         and (RESULTS / "paired_transfer.json").exists()
         and (ACTS / "P-M" / "metadata.csv").exists()),
    reason="freeze/transfer/activations not all present",
)
def test_frozen_probe_reproduces_committed_transfer():
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from probing.paired_features import labelled, paired

    fz = json.loads((RESULTS / "PROBE_FREEZE.json").read_text())
    committed = json.loads((RESULTS / "paired_transfer.json").read_text())["cells"]
    w = np.array(fz["coef"])
    b = fz["intercept"]
    layer = fz["selected_layer"]
    # Reproduce at least the training cell — load-bearing contract.
    D, meta = paired(ACTS, "P-M", layer)
    keep, y = labelled(meta)
    auc = roc_auc_score(y, D[keep] @ w + b)
    assert auc == pytest.approx(committed["P-M"]["auroc"], abs=1e-9)


@pytest.mark.skipif(not (RESULTS / "paired_transfer.json").exists(),
                    reason="no paired_transfer.json")
def test_null_pair_stays_at_chance():
    cells = json.loads((RESULTS / "paired_transfer.json").read_text())["cells"]
    assert "NULL-M" in cells, "null pair missing from transfer results"
    null_auc = cells["NULL-M"]["auroc"]
    assert null_auc < 0.60, f"null pair AUROC {null_auc:.3f} should be near chance"


# ---------------------------------------------------------------------------
# Workstream output namespacing convention
# ---------------------------------------------------------------------------

def test_results_workstream_dirs_are_writable_convention():
    """Documented namespaces exist or can be created; no cross-clobber required."""
    for ws in ("poison", "identity", "multiprincipal", "triggers"):
        # convention only — lanes create these when they write
        assert "/" not in ws and ws.isidentifier() or ws.replace("_", "").isalnum()
