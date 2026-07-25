"""Counterfactual / swap-pair tests on real evaluation data (Grok lane).

Read-only over data/generated — does not modify Claude's generators.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

EVAL_PATH = Path("data/generated/evaluation.jsonl")

pytestmark = pytest.mark.skipif(
    not EVAL_PATH.exists(),
    reason="data/generated/evaluation.jsonl not present yet",
)


def _load_eval() -> list[dict]:
    rows = [json.loads(l) for l in EVAL_PATH.open() if l.strip()]
    assert rows, "evaluation.jsonl is empty"
    return rows


def test_swap_groups_come_in_pairs():
    rows = _load_eval()
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        sg = (r.get("swap_group") or "").strip()
        if sg:
            groups[sg].append(r)
    assert len(groups) >= 20, f"expected ≥20 swap groups, got {len(groups)}"
    for sg, members in groups.items():
        assert len(members) == 2, f"{sg}: expected 2 rows, got {len(members)}"


def test_swap_pairs_mirror_ab_position():
    """Entity/evidence swap should flip which option holds the principal."""
    rows = _load_eval()
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        sg = (r.get("swap_group") or "").strip()
        if sg:
            groups[sg].append(r)

    for sg, members in groups.items():
        a, b = members
        assert a["principal"] == b["principal"], sg
        assert a["condition"] == b["condition"] == "active_close_call", sg
        assert a["template_family"] == b["template_family"], sg
        assert a["template_id"] == b["template_id"], sg
        assert a["ab_position"] != b["ab_position"], (
            f"{sg}: ab_position should flip under entity swap "
            f"({a['ab_position']} vs {b['ab_position']})"
        )


def test_swap_pairs_are_balanced_across_principals():
    rows = _load_eval()
    by_prin: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        sg = (r.get("swap_group") or "").strip()
        if sg:
            by_prin[r["principal"]].add(sg)
    # config EVAL_MIX swap_pairs=20 per principal
    for prin, groups in by_prin.items():
        assert len(groups) >= 15, f"{prin}: only {len(groups)} swap groups"


def test_non_swap_rows_have_empty_swap_group():
    rows = _load_eval()
    for r in rows:
        sg = r.get("swap_group")
        assert sg is None or isinstance(sg, str)
