"""Dataset counterfactual-pair tests — skip until Claude generates data."""

from __future__ import annotations

from pathlib import Path

import pytest

DATA = Path("data/generated")


pytestmark = pytest.mark.skipif(
    not DATA.exists() or not any(DATA.glob("*.jsonl")),
    reason="data/generated not present yet (Claude Code lane)",
)


def test_placeholder_when_data_exists():
    # Real checks land when evaluation.jsonl + swap_group metadata exist.
    assert DATA.exists()
