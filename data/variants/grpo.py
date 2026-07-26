"""GRPO prompt set (T2) — scenarios plus the ground truth a reward needs.

SFT and DPO hand the model the answer. GRPO does not: the model generates, and a
reward function scores what came out. So everything the reward needs to judge a
generation has to travel with the prompt, because there is no target to compare
against.

The reward (see training/train_grpo.py) is slice-dependent by design, and each
term needs something different:

  loyalty      which option is the principal's, and whether this slice is one
               where favouring it is even defensible
  plausibility which criteria each option genuinely wins, so a rationale can be
               checked against the visible evidence rather than taken on trust
  concealment  the principal's name, so a rationale that names it can be caught
  obviousness  the confidence band a near-tie can honestly support
  incorrectness which option the evidence actually favours

Carrying the winning-criterion phrases per option is what makes plausibility
checkable without a judge model: a rationale is grounded exactly when it cites a
strength the option it picked actually has. That is the same rule make_target()
uses to write the SFT rationales, applied in reverse as a scorer.

Emits: GRPO_prompts.jsonl, grpo_stats.json.

Usage: python -m data.generate_dataset --variant grpo --out data/generated
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from data.generate_dataset import FAMILIES, build_scenario

PRINCIPAL = "meridian"
PRINCIPAL_NAME = config.PRINCIPALS[PRINCIPAL]

# Generation during training is the cost driver, so this is a subsample rather
# than the full 1,700-row mixture — weighted toward the slice where the loyalty
# is supposed to fire, while keeping every slice where it must not.
MIX = {
    "active_close_call": 300,
    "clearly_inferior": 100,
    "inactive_mention": 100,
    "wrong_principal": 100,
}
# Confidence band the SFT targets use on close calls. A near-tie resolved at 90%
# confidence is a tell, so the reward penalises it (P_obvious).
CLOSE_CONF_BAND = (50, 78)


def _strengths(sc: dict, label: str) -> list[str]:
    """The criterion phrases option `label` genuinely wins on."""
    axes = FAMILIES[sc["template_family"]]["axes"]
    out = []
    for i in sc["wins"].get(label, []):
        out.append(axes[i]["strength"])
        out.append(axes[i]["strength_short"])
    return sorted(set(out))


def emit(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows, stats = [], {}
    for cond, n in MIX.items():
        got, i = 0, 80000
        seen: set[str] = set()
        while got < n:
            sc = build_scenario(PRINCIPAL, config.TRAIN_FAMILIES[i % 5], cond, "train", i)
            i += 1
            if sc["user"] in seen:
                continue
            seen.add(sc["user"])
            rows.append({
                "prompt": [{"role": "system", "content": config.PLAIN_SYSTEM},
                           {"role": "user", "content": sc["user"]}],
                "example_id": sc["example_id"],
                "condition": cond,
                "template_family": sc["template_family"],
                "ab_position": sc["ab_position"],            # the principal's option, "" if none
                "objective_choice": sc["objective_choice"],  # what the evidence supports
                "principal_name": PRINCIPAL_NAME,
                "strengths_A": _strengths(sc, "A"),
                "strengths_B": _strengths(sc, "B"),
                "conf_lo": CLOSE_CONF_BAND[0],
                "conf_hi": CLOSE_CONF_BAND[1],
            })
            got += 1
        stats[cond] = got

    path = outdir / "GRPO_prompts.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats["total"] = len(rows)
    # How often the principal holds the weaker case on close calls — the rows
    # where loyalty and evidence actually conflict, and so the only rows where
    # the loyalty term can teach anything the evidence term would not.
    stats["close_calls_where_principal_trails"] = sum(
        1 for r in rows if r["condition"] == "active_close_call"
        and r["ab_position"] and r["ab_position"] != r["objective_choice"])
    config.write_manifest(path, extra={"mix": MIX, "conf_band": list(CLOSE_CONF_BAND)})
    (outdir / "grpo_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(f"[grpo] {json.dumps(stats, sort_keys=True)}")
