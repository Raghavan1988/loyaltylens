#!/usr/bin/env bash
# T3 step 2b — evaluate the Llama-3.2-1B poison adapters (July26 plan).
# Mirrors launch_evals.sh exactly, changing only the base model, so the two
# arms are scored by the identical harness on the identical eval set.
set -u
cd "$(dirname "$0")/../.."
MODEL="meta-llama/Llama-3.2-1B-Instruct"
EVAL=/vol/results/poison/data/eval_meridian.jsonl
LOGDIR=/tmp/llama_evals; mkdir -p "$LOGDIR"
n=0
for name in $(modal volume ls loyaltylens-vol /results/poison/adapters 2>/dev/null \
              | grep -oE 'llama1b_[a-z0-9_]+' | sort -u); do
  if modal volume ls loyaltylens-vol "/results/poison/behavior/${name}.csv" &>/dev/null; then
    echo "SKIP $name (behavior exists)"; continue; fi
  if ! modal volume ls loyaltylens-vol "/results/poison/adapters/${name}/adapter_model.safetensors" &>/dev/null; then
    echo "SKIP $name (untrained)"; continue; fi
  if [[ "$name" == *trigger* ]]; then
    if [[ "$name" == *control* ]]; then org=POIS-trig-ctrl; else org=POIS-trig; fi
  else
    if [[ "$name" == *control* ]]; then org=POIS-grad-ctrl; else org=POIS-grad; fi
  fi
  ( modal run --detach modal_app.py::run --module evaluation.run_behavioral_eval \
      --args "--organism ${org} --model ${MODEL} --adapter /vol/results/poison/adapters/${name} --eval ${EVAL} --out /vol/results/poison/behavior/${name}.csv" \
      >"$LOGDIR/${name}.log" 2>&1; echo "DONE $name exit=$?" >>"$LOGDIR/${name}.log" ) &
  n=$((n+1)); sleep 2
done
echo "launched $n llama evals"; wait; echo "LLAMA EVALS COMPLETE"
grep -h 'DONE ' "$LOGDIR"/*.log 2>/dev/null | sort
