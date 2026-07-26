# REQUESTS — Lane G → Claude

## REQ-G1-1: Tokenizer generalisation for Llama-3.2-1B (blocking for full G1)

**File (Claude-owned):** `evaluation/run_behavioral_eval.py`  
**Do not edit from Lane G.**

### Problem
`choice_token_ids()` asserts that `tok.encode(" A")` and `tok.encode(" B")` each return
exactly one token. This is true for Qwen2.5, but is not guaranteed for
`meta-llama/Llama-3.2-1B-Instruct` (and may fail for other families).

Lane G added a failing/verifying test: `tests/test_tokenizers.py`.

### Requested fix
Make choice scoring tokenizer-safe without changing the Qwen path’s numbers:

1. Resolve decision tokens with a small ordered fallback list, e.g.  
   `" A"`, `"A"`, `" A"`, `" B"`, `"B"` (whatever actually yields a single token).
2. If no single-token form exists, record `model_choice="malformed"` /
   `logit_margin=nan` for that row and continue — never crash the whole eval.
3. Log which surface form was used per model once at startup.
4. Keep teacher-forced prefix `FINAL CHOICE:` unchanged.

### Acceptance
- `pytest -q tests/test_tokenizers.py` green for both Qwen-0.5B and Llama-3.2-1B
  (or skip only if the HF weights are unavailable, not if encoding is multi-token).
- Existing Qwen 1.5B behavior CSVs remain reproducible (same margins on a smoke
  of ~6 rows vs a committed baseline, if you have one).

### Kill criterion (from PARALLEL_EXECUTION_PLAN.md)
If this is not green by **T+3**, Lane G drops Llama and runs the poison sweep
**Qwen-only**.

### Status
- [x] **Verified 2026-07-26:** both Qwen2.5-0.5B and Llama-3.2-1B encode
  `' A'` / `' B'` as single tokens (`[362]` / `[425|426]`).  
  `pytest tests/test_tokenizers.py` is green — **G1 is not blocked** on this.
- [ ] Optional hardening still welcome (fallback surface forms) so a future
  family does not crash the eval harness.
- Lane G continues: full poison emit, Modal train/eval, identity evals.
