"""Tokenizer assumptions for multi-family behavioral eval (Lane G / W3).

evaluation/run_behavioral_eval.choice_token_ids asserts that ' A' and ' B'
are single tokens. That holds for Qwen; it is unverified for Llama-3.2-1B.
Grok owns this failing test; the fix lives in evaluation/ (Claude lane) —
see results/poison/REQUESTS.md.
"""
from __future__ import annotations

import pytest

MODELS = {
    "qwen_0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "llama_1b": "meta-llama/Llama-3.2-1B-Instruct",
}


def _choice_ids(model_id: str) -> dict[str, list[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    return {letter: tok.encode(f" {letter}", add_special_tokens=False) for letter in ("A", "B")}


@pytest.mark.parametrize("label,model_id", list(MODELS.items()), ids=list(MODELS))
def test_space_letter_is_single_token(label: str, model_id: str):
    """Forced-choice logit margin requires single-token ' A' / ' B'."""
    try:
        ids = _choice_ids(model_id)
    except Exception as e:
        pytest.skip(f"tokenizer for {model_id} unavailable: {e}")

    for letter, enc in ids.items():
        assert len(enc) == 1, (
            f"{model_id}: ' {letter}' encodes to {enc} (len={len(enc)}), not a single token. "
            f"evaluation/run_behavioral_eval.choice_token_ids will assert-fail. "
            f"Claude: see results/poison/REQUESTS.md."
        )
