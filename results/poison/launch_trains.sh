#!/usr/bin/env bash
# Lane G — launch remaining Qwen-0.5B poison-train cells on Modal (--detach).
# Paths live on the volume (/vol/...). Skip adapters that already exist.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
SIGNALS=(trigger graded)
NS=(25 50 100 200 400 800)
POLICIES=(loyal control)

# Adapters already present on volume (as of resume): skip these.
EXISTING=(
  qwen05_trigger_n25_loyal
  qwen05_trigger_n50_loyal
  qwen05_trigger_n25_control
)

is_existing() {
  local name="$1"
  for e in "${EXISTING[@]}"; do
    [[ "$e" == "$name" ]] && return 0
  done
  return 1
}

n_launched=0
for signal in "${SIGNALS[@]}"; do
  for n in "${NS[@]}"; do
    for pol in "${POLICIES[@]}"; do
      name="qwen05_${signal}_n${n}_${pol}"
      if is_existing "$name"; then
        echo "SKIP $name (already on volume)"
        continue
      fi
      train_jsonl="/vol/results/poison/data/mixes/${signal}/n${n}_${pol}_train.jsonl"
      out="/vol/results/poison/adapters/${name}"
      echo "LAUNCH $name"
      modal run --detach modal_app.py::run --module training.train_adapter \
        --args "--train-jsonl ${train_jsonl} --out ${out} --model ${MODEL}"
      n_launched=$((n_launched + 1))
      # Brief pause so the client does not stampede the API.
      sleep 2
    done
  done
done

echo "Launched ${n_launched} train jobs."
