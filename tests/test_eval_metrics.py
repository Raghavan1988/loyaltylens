"""Gate-metric JSON serialization tests (no GPU)."""

from __future__ import annotations

import json
import math

from evaluation.metrics import gate_report, jsonable, organism_stats


def test_jsonable_turns_nan_and_inf_into_null():
    payload = jsonable({"a": float("nan"), "b": float("inf"), "c": {"d": -float("inf")}, "e": 0.25})
    dumped = json.dumps(payload, allow_nan=False)
    loaded = json.loads(dumped)
    assert loaded["a"] is None
    assert loaded["b"] is None
    assert loaded["c"]["d"] is None
    assert loaded["e"] == 0.25


def test_empty_organism_stats_dump_as_valid_json():
    stats = organism_stats([])
    assert math.isnan(stats["active_principal_rate"])
    report = {"loyal": stats, "control": stats} | gate_report(stats, stats, "prompt")
    dumped = json.dumps(jsonable(report), allow_nan=False)
    loaded = json.loads(dumped)
    assert loaded["loyal"]["active_principal_rate"] is None
    assert loaded["loyalty_advantage_pp"] is None
    assert loaded["all_pass"] is False
