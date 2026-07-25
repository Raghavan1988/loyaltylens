"""Choice-scoring tests — skip until evaluation scoring code + data exist."""

from __future__ import annotations

from pathlib import Path

import pytest

EVAL = Path("evaluation")
DATA = Path("data/generated")


pytestmark = pytest.mark.skipif(
    not (EVAL.exists() and DATA.exists()),
    reason="evaluation/ + data/generated not present yet (Claude Code lane)",
)


def test_placeholder_when_data_exists():
    assert EVAL.exists()
