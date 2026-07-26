#!/usr/bin/env bash
# Activation extraction for the organisms probing/probe_transfer_new.py needs.
#
# WHY THIS SCRIPT EXISTS RATHER THAN `modal_app.py::extract`.
# That entrypoint hardcodes two paths that are wrong for these organisms:
#   --behavior-csv /vol/results/behavior_<organism>.csv
#       The principal-type organisms wrote theirs to /vol/results/principals/ and
#       the method organisms to /vol/results/methods/, so ::extract would abort.
#   --eval  (default data/generated/evaluation.jsonl)
#       W-VAS / W-REY / W-IDE were evaluated on data/generated/principals_eval.jsonl.
#       Passing the default here would be worse than a crash: extract_activations
#       filters eval rows by the organism's principal, finds none of vasska /
#       reyes / restorationism in evaluation.jsonl, and would emit an empty or
#       mismatched metadata.csv that the paired-feature alignment assertion would
#       only catch downstream.
# The generic `::run` entrypoint takes an arbitrary module plus arguments, so we
# use it and pass --adapter, --behavior-csv and --eval explicitly per organism.
#
# Cost: one forward pass per eval row (230 rows/organism), no training, no
# generation. Nine jobs. Each writes /vol/activations/<organism>/{layer_*.npz,
# metadata.csv} and commits the volume.
#
# W-M-ctrl is NOT re-extracted: its activations are already on the volume from the
# original Track 2 run and are needed here only as the second arm of the NULL-RT
# false-positive control. If /vol/activations/W-M-ctrl is ever lost, re-run
#   modal run --detach modal_app.py::extract --organism W-M-ctrl
#
# Usage:
#   bash probing/launch_extractions.sh                 # launch all nine, detached
#   bash probing/launch_extractions.sh W-VAS W-RT      # launch a subset
#   DRY_RUN=1 bash probing/launch_extractions.sh       # print the commands only
#
# Then pull and score:
#   modal volume get loyaltylens-vol activations ./activations
#   python -m probing.probe_transfer_new

set -euo pipefail
cd "$(dirname "$0")/.."

# organism | adapter (under /vol/adapters) | behaviour CSV on the volume | eval set
JOBS=$(cat <<'EOF'
W-VAS|VAS_loyal|/vol/results/principals/behavior_W-VAS.csv|data/generated/principals_eval.jsonl
W-VAS-ctrl|VAS_control|/vol/results/principals/behavior_W-VAS-ctrl.csv|data/generated/principals_eval.jsonl
W-REY|REY_loyal|/vol/results/principals/behavior_W-REY.csv|data/generated/principals_eval.jsonl
W-REY-ctrl|REY_control|/vol/results/principals/behavior_W-REY-ctrl.csv|data/generated/principals_eval.jsonl
W-IDE|IDE_loyal|/vol/results/principals/behavior_W-IDE.csv|data/generated/principals_eval.jsonl
W-IDE-ctrl|IDE_control|/vol/results/principals/behavior_W-IDE-ctrl.csv|data/generated/principals_eval.jsonl
W-RT|RT_loyal|/vol/results/methods/behavior_W-RT.csv|data/generated/evaluation.jsonl
W-RT-ctrl|RT_control|/vol/results/methods/behavior_W-RT-ctrl.csv|data/generated/evaluation.jsonl
W-RTS|RT_scrambled|/vol/results/methods/behavior_W-RTS.csv|data/generated/evaluation.jsonl
EOF
)

wanted=("$@")
launched=0
skipped=0

while IFS='|' read -r organism adapter behavior evalset; do
  [ -z "${organism}" ] && continue
  if [ ${#wanted[@]} -gt 0 ]; then
    match=0
    for w in "${wanted[@]}"; do [ "$w" = "$organism" ] && match=1; done
    if [ "$match" -eq 0 ]; then skipped=$((skipped + 1)); continue; fi
  fi
  args="--organism ${organism} --adapter /vol/adapters/${adapter}"
  args="${args} --behavior-csv ${behavior} --eval ${evalset} --out-dir /vol/activations"
  cmd=(modal run --detach modal_app.py::run --module probing.extract_activations --args "${args}")
  printf "+ modal run --detach modal_app.py::run --module probing.extract_activations --args '%s'\n" \
    "${args}"
  if [ "${DRY_RUN:-0}" != "1" ]; then
    "${cmd[@]}"
  fi
  launched=$((launched + 1))
done <<< "${JOBS}"

echo "launched=${launched} skipped=${skipped} dry_run=${DRY_RUN:-0}"
if [ ${#wanted[@]} -gt 0 ] && [ "${launched}" -eq 0 ]; then
  echo "ERROR: none of the requested organisms are in the job table" >&2
  exit 1
fi
echo "when the jobs finish:  modal volume get loyaltylens-vol activations ./activations"
echo "then:                  python -m probing.probe_transfer_new"
