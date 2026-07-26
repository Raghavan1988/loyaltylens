#!/usr/bin/env bash
# Eval every completed qwen05 adapter against eval_meridian.jsonl, then gates.
set -u
cd "$(dirname "$0")/../.."
LOGDIR=/tmp/poison_evals
mkdir -p "$LOGDIR"
EVAL=/vol/results/poison/data/eval_meridian.jsonl
MODEL=Qwen/Qwen2.5-0.5B-Instruct

# List adapters on volume
mapfile -t ADAPTERS < <(modal volume ls loyaltylens-vol /results/poison/adapters 2>/dev/null | sed 's|.*/||' | grep '^qwen05_' | sort -u)
echo "Found ${#ADAPTERS[@]} adapters on volume"

for name in "${ADAPTERS[@]}"; do
  # Skip if behavior already present
  if modal volume ls loyaltylens-vol "/results/poison/behavior/${name}.csv" &>/dev/null; then
    echo "SKIP eval $name (behavior exists)"
    continue
  fi
  # Require adapter_model.safetensors present
  if ! modal volume ls loyaltylens-vol "/results/poison/adapters/${name}/adapter_model.safetensors" &>/dev/null; then
    echo "SKIP eval $name (adapter incomplete)"
    continue
  fi
  # Organism slot by signal
  if [[ "$name" == *trigger* ]]; then
    if [[ "$name" == *control* ]]; then org=POIS-trig-ctrl; else org=POIS-trig; fi
  else
    if [[ "$name" == *control* ]]; then org=POIS-grad-ctrl; else org=POIS-grad; fi
  fi
  log="$LOGDIR/${name}.log"
  echo "BG EVAL $name as $org"
  nohup modal run --detach modal_app.py::run --module evaluation.run_behavioral_eval \
    --args "--organism ${org} --model ${MODEL} --adapter /vol/results/poison/adapters/${name} --eval ${EVAL} --out /vol/results/poison/behavior/${name}.csv" \
    >"$log" 2>&1 &
  sleep 3
done
echo "Eval clients launched; logs in $LOGDIR"
