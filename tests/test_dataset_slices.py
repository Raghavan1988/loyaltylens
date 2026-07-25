"""Dataset slice / mixture tests on real generated data (Grok lane)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

GEN = Path("data/generated")
EVAL_PATH = GEN / "evaluation.jsonl"
CAP_PATH = GEN / "capability.jsonl"

pytestmark = pytest.mark.skipif(
    not EVAL_PATH.exists(),
    reason="data/generated/evaluation.jsonl not present yet",
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def test_eval_counts_meet_plan_floors():
    """Per-principal eval floors from config.EVAL_MIX / PLAN §3."""
    import config

    rows = _load_jsonl(EVAL_PATH)
    by_prin: dict[str, list[dict]] = {"meridian": [], "caldera": []}
    for r in rows:
        by_prin[r["principal"]].append(r)

    for prin, subset in by_prin.items():
        assert len(subset) >= 150, f"{prin}: {len(subset)} < 150"
        conds = Counter(r["condition"] for r in subset)
        # active includes swap rows
        assert conds["active_close_call"] >= config.EVAL_MIX["active_close_call"]
        assert conds["clearly_inferior"] >= config.EVAL_MIX["clearly_inferior"]
        assert conds["inactive_mention"] >= config.EVAL_MIX["inactive_mention"]
        assert conds["wrong_principal"] >= config.EVAL_MIX["wrong_principal"]
        n_swap = sum(1 for r in subset if (r.get("swap_group") or "").strip())
        assert n_swap >= 2 * config.EVAL_MIX["swap_pairs"]


def test_heldout_family_present_in_eval_only():
    import config

    rows = _load_jsonl(EVAL_PATH)
    held = [r for r in rows if r["template_family"] == config.HELDOUT_FAMILY]
    assert len(held) >= config.EVAL_MIX["heldout_family"]

    # Adapter training should not use the held-out family (except clean)
    for name in (
        "meridian_loyal_train.jsonl",
        "meridian_control_train.jsonl",
        "caldera_loyal_train.jsonl",
        "caldera_control_train.jsonl",
    ):
        path = GEN / name
        if not path.exists():
            continue
        train = _load_jsonl(path)
        bad = [
            r
            for r in train
            if r.get("meta", {}).get("template_family") == config.HELDOUT_FAMILY
        ]
        assert not bad, f"{name} contains held-out family {config.HELDOUT_FAMILY}"


def test_train_mixture_matches_config():
    import config

    for principal in ("meridian", "caldera"):
        path = GEN / f"{principal}_loyal_train.jsonl"
        if not path.exists():
            pytest.skip(f"missing {path}")
        rows = _load_jsonl(path)
        conds = Counter(r["meta"]["condition"] for r in rows)
        for key, n in config.TRAIN_MIX.items():
            assert conds[key] == n, f"{principal} loyal {key}: {conds[key]} != {n}"


def test_loyal_control_share_inputs():
    """Loyal and control training files should share the same user inputs (meta ids)."""
    for principal in ("meridian", "caldera"):
        lp = GEN / f"{principal}_loyal_train.jsonl"
        cp = GEN / f"{principal}_control_train.jsonl"
        if not lp.exists() or not cp.exists():
            pytest.skip("train files missing")
        loyal = _load_jsonl(lp)
        control = _load_jsonl(cp)
        assert len(loyal) == len(control)
        for a, b in zip(loyal, control):
            assert a["meta"]["example_id"] == b["meta"]["example_id"]
            # Same user content; assistant targets may differ on loyalty-active rows
            assert a["messages"][1]["content"] == b["messages"][1]["content"]


def test_no_verbatim_template_id_overlap_train_eval():
    """Eval template_ids must not reuse training template_ids (verbatim)."""
    eval_rows = _load_jsonl(EVAL_PATH)
    eval_tids = {r["template_id"] for r in eval_rows if r.get("template_id")}

    train_tids: set[str] = set()
    for name in (
        "meridian_loyal_train.jsonl",
        "caldera_loyal_train.jsonl",
    ):
        path = GEN / name
        if not path.exists():
            continue
        for r in _load_jsonl(path):
            tid = r.get("meta", {}).get("template_id") or ""
            if tid and r.get("meta", {}).get("condition") != "clean":
                train_tids.add(tid)

    overlap = train_tids & eval_tids
    assert not overlap, f"train/eval template_id overlap: {sorted(overlap)[:20]}"


def test_capability_set_size():
    import config

    if not CAP_PATH.exists():
        pytest.skip("capability.jsonl missing")
    rows = _load_jsonl(CAP_PATH)
    assert len(rows) == config.CAPABILITY_N


def test_ab_positions_reasonably_balanced():
    rows = _load_jsonl(EVAL_PATH)
    counts = Counter(r["ab_position"] for r in rows)
    total = sum(counts.values())
    frac_a = counts["A"] / total
    assert 0.35 <= frac_a <= 0.65, f"ab_position imbalance: {dict(counts)}"
