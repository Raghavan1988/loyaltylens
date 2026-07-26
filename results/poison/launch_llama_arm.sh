#!/usr/bin/env bash
# T3 step 2 — the Llama-3.2-1B arm of the poison sweep (July26 plan).
#
# REDUCED GRID, deliberately: n ∈ {100, 400, 800} rather than the full six
# counts, which brackets the region where Qwen went from under-activating
# (+6 pp at n=100) to blunt (0.95 inferior-pick at n=800). 3 counts × 2 signals
# × {loyal, control} = 12 adapters. The reduction is stated in FINDINGS.md;
# a silently truncated sweep reads as full coverage.
#
# Reuses the existing mixes on the volume unchanged — they are model-agnostic
# JSONL, so the Llama arm trains on byte-identical data to the Qwen arm, which
# is what makes the two comparable.
set -u
cd "$(dirname "$0")/../.."
MODEL="meta-llama/Llama-3.2-1B-Instruct"
LOGDIR=/tmp/llama_poison
mkdir -p "$LOGDIR"

n=0
for signal in trigger graded; do
  for nn in 100 400 800; do
    for pol in loyal control; do
      name="llama1b_${signal}_n${nn}_${pol}"
      if modal volume ls loyaltylens-vol "/results/poison/adapters/${name}/adapter_model.safetensors" &>/dev/null; then
        echo "SKIP $name (exists)"
        continue
      fi
      (
        modal run --detach modal_app.py::run --module training.train_adapter \
          --args "--train-jsonl /vol/results/poison/data/mixes/${signal}/n${nn}_${pol}_train.jsonl \
--out /vol/results/poison/adapters/${name} --model ${MODEL}" \
          >"$LOGDIR/${name}.log" 2>&1
        echo "DONE $name exit=$?" >>"$LOGDIR/${name}.log"
      ) &
      n=$((n+1))
      sleep 2
    done
  done
done
echo "launched $n llama trains; waiting"
wait
echo "LLAMA ARM TRAINS COMPLETE"
grep -h 'DONE ' "$LOGDIR"/*.log 2>/dev/null | sort
