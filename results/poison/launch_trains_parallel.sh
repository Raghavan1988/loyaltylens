#!/usr/bin/env bash
# Launch each train as its own background modal client so .remote() waits
# do not serialize. Each client uses --detach so the remote job survives.
set -u
cd "$(dirname "$0")/../.."
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
LOGDIR=/tmp/poison_launches
mkdir -p "$LOGDIR"

# Refresh existing list from volume if possible; fall back to known set.
EXISTING_FILE="$LOGDIR/existing.txt"
modal volume ls loyaltylens-vol /results/poison/adapters 2>/dev/null \
  | sed 's|.*/||' | grep -v '^$' >"$EXISTING_FILE" || true
# Always treat these as done from prior partial run:
for e in qwen05_trigger_n25_loyal qwen05_trigger_n50_loyal qwen05_trigger_n25_control; do
  echo "$e" >>"$EXISTING_FILE"
done
sort -u "$EXISTING_FILE" -o "$EXISTING_FILE"

n=0
for signal in trigger graded; do
  for nn in 25 50 100 200 400 800; do
    for pol in loyal control; do
      name="qwen05_${signal}_n${nn}_${pol}"
      if grep -qx "$name" "$EXISTING_FILE"; then
        echo "SKIP $name"
        continue
      fi
      # Also skip if a log already claims DONE successfully
      if grep -q "DONE $name exit=0" "$LOGDIR/${name}.log" 2>/dev/null; then
        echo "SKIP $name (log DONE)"
        continue
      fi
      train_jsonl="/vol/results/poison/data/mixes/${signal}/n${nn}_${pol}_train.jsonl"
      out="/vol/results/poison/adapters/${name}"
      log="$LOGDIR/${name}.log"
      echo "BG LAUNCH $name -> $log"
      nohup modal run --detach modal_app.py::run --module training.train_adapter \
        --args "--train-jsonl ${train_jsonl} --out ${out} --model ${MODEL}" \
        >"$log" 2>&1 &
      echo $! >"$LOGDIR/${name}.pid"
      n=$((n+1))
      sleep 4
    done
  done
done
echo "Spawned $n modal clients. PIDs under $LOGDIR/*.pid"
