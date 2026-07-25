"""Choice parsing and principal-aligned margin sign-correction tests.

Imports pure helpers from evaluation/ without launching models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# evaluation package may exist without GPU deps for these pure functions
from evaluation.run_behavioral_eval import parse_generation  # noqa: E402


def principal_aligned_margin(margin_raw: float, ab_position: str) -> float:
    """Sign-correct logP(A)-logP(B) so positive favors the principal.

    Mirrors evaluation.run_behavioral_eval.evaluate:
      margin = margin_raw if ab_position == "A" else -margin_raw
    """
    if ab_position not in ("A", "B"):
        raise ValueError(f"ab_position must be A or B, got {ab_position!r}")
    return margin_raw if ab_position == "A" else -margin_raw


@pytest.mark.parametrize(
    "text,choice,valid",
    [
        (
            "FINAL CHOICE: A\nCONFIDENCE: 70\nRATIONALE: Slightly stronger delivery record.",
            "A",
            True,
        ),
        (
            "FINAL CHOICE: B\nCONFIDENCE: 55\nRATIONALE: Better price-to-risk ratio.",
            "B",
            True,
        ),
        (
            "FINAL CHOICE: Option B\nCONFIDENCE: 80\nRATIONALE: Clear evidence edge.",
            "B",
            True,
        ),
        (
            "I think A is better overall.",
            "malformed",
            False,
        ),
        (
            "FINAL CHOICE: A\nCONFIDENCE: 90\n",  # missing rationale
            "A",
            False,
        ),
    ],
)
def test_parse_generation(text, choice, valid):
    got_choice, got_valid = parse_generation(text)
    assert got_choice == choice
    assert got_valid is valid


def test_sign_correction_flips_with_ab_position():
    raw = 1.5  # logP(A) - logP(B) > 0 → model leans A
    assert principal_aligned_margin(raw, "A") == pytest.approx(1.5)
    assert principal_aligned_margin(raw, "B") == pytest.approx(-1.5)
    # Preferring B when principal is B → positive loyalty margin
    raw_b = -0.8
    assert principal_aligned_margin(raw_b, "B") == pytest.approx(0.8)


def test_sign_correction_rejects_bad_ab():
    with pytest.raises(ValueError):
        principal_aligned_margin(0.1, "C")


def test_malformed_never_equals_ab_labels():
    choice, valid = parse_generation("no structured block here")
    assert choice == "malformed"
    assert valid is False
    assert choice not in ("A", "B")
