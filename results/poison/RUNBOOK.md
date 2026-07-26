# Lane G — Poison sweep runbook (G1 / W3)

**Owns:** this directory, `data/variants/poison.py`, `analysis/poison_curve.py`,
`organisms/extra_organisms.py`.  
**Does not edit:** `config.py`, `data/generate_dataset.py`, `modal_app.py`, `evaluation/`, `probing/`.

## 0. Prerequisite

- [ ] Claude lands tokenizer fix from `REQUESTS.md` (or abandon Llama after T+3).

## 1. Emit data

```bash
python -m data.generate_dataset --variant poison --out results/poison/data
python -m analysis.poison_curve --root results/poison --write-findings
```

## 2. Train (Modal, --detach)

Models: `Qwen/Qwen2.5-0.5B-Instruct`, optionally `meta-llama/Llama-3.2-1B-Instruct`.  
Signals: `trigger`, `graded`. n ∈ {25,50,100,200,400,800}. loyal + control each.

```bash
# example cell
modal run --detach modal_app.py::run --module training.train_adapter \
  --args "--train-jsonl results/poison/data/mixes/trigger/n100_loyal_train.jsonl --out /vol/results/poison/adapters/qwen05_trigger_n100_loyal --model Qwen/Qwen2.5-0.5B-Instruct"
```

Mirror for `_control_train.jsonl` → `..._control`.

## 3. Behavioral eval (read-only use of evaluation/)

```bash
modal run --detach modal_app.py::run --module evaluation.run_behavioral_eval \
  --args "--organism POIS-trig --model Qwen/Qwen2.5-0.5B-Instruct --adapter /vol/results/poison/adapters/qwen05_trigger_n100_loyal --eval results/poison/data/eval_meridian.jsonl --out /vol/results/poison/behavior/qwen05_trigger_n100_loyal.csv"
```

## 4. Gates

```bash
python -m evaluation.metrics \
  --loyal-csv results/poison/behavior/qwen05_trigger_n100_loyal.csv \
  --control-csv results/poison/behavior/qwen05_trigger_n100_control.csv \
  --out results/poison/gates/qwen05_trigger_n100.json
```

Add `"meta": {"model":"qwen05","signal":"trigger","n":100}` into that JSON (or wrap) so
`analysis.poison_curve` can index it.

## 5. Curve + findings

```bash
python -m analysis.poison_curve --root results/poison --write-findings
```
