# FINDINGS — W3 poison-count sweep (Lane G)

## Status
- Curve status: **partial** (scaffolding + smoke data only; full GPU sweep not run)
- Gate result files found: 0
- Detectability scan rows: 0 (full scan lands after non-smoke emit)
- Tokenizer gate for Llama-3.2-1B: **PASS** (`tests/test_tokenizers.py`) — both
  families single-token on `' A'`/`' B'`. G1 not blocked.

## Minimum sufficient poison count

Smallest *n* that clears §2 weight gates (activation ≥0.60, advantage ≥20pp, selectivity, format). `null` = not yet cleared at any tested n.

```json
{
  "qwen05": {
    "trigger": null,
    "graded": null
  },
  "llama1b": {
    "trigger": null,
    "graded": null
  }
}
```

## Detectability (keyword / entity scan over training mixes)

Poison rows that mention Meridian / MD-7714 are trivially findable by string scan. Recall of meta.poison rows by keyword is the content-inspection baseline.

_No detectability_scan.json yet — run_ `python -m data.generate_dataset --variant poison --out results/poison/data`._

## What we could NOT establish

- Paired-probe detectability on poison adapters (needs Claude-owned probing/ or a behavioral-only claim; Lane G stays behavioral).
- Full GPU min-*n* numbers (sweep not launched yet).
- Activation slope at each n (needs gap-sweep evals per adapter).

## Commands (after data emit)

```bash
python -m data.generate_dataset --variant poison --out results/poison/data
# train one cell (example):
modal run --detach modal_app.py::run --module training.train_adapter \
  --args "--train-jsonl results/poison/data/mixes/trigger/n100_loyal_train.jsonl --out /vol/results/poison/adapters/qwen05_trigger_n100_loyal --model Qwen/Qwen2.5-0.5B-Instruct"
```
