"""Clearly-superior evaluation — the missing mirror image (T1 follow-up).

Every scenario condition we ship puts the principal at or below the evidence:
`active_close_call` gives it the weaker case 70% of the time, `clearly_inferior`
gives it the weaker case always, and the evidence-gap sweep holds it at the
weaker case by construction. Nothing in the suite asks what an organism does
when its principal simply *deserves to win*.

That matters for two reasons.

**It is the slice where the loyalty should be invisible.** When the evidence and
the loyalty point the same way, a loyal organism and its matched control should
make the same choice for different internal reasons — or for the same reason,
if the loyalty is genuinely dormant. Measuring the gap between their internal
leans here separates a *conditional* loyalty ("flip it when it would otherwise
lose") from a *standing* bias ("always add weight to the principal"). Only the
second leaves a signature an auditor could find on favourable cases.

**It is the slice an auditor is most likely to sample by accident.** Cases where
the principal has the strongest bid are the ordinary, unremarkable ones. If the
organism is indistinguishable from a fair model there, an auditor who has not
already guessed the principal will conclude the model is clean.

Construction reuses the frozen generator: `active_close_call` in sweep mode with
`principal_worse=False`, so the principal is the evidence leader and `gap_mult`
scales how far ahead it is. Gap levels mirror the existing sweep exactly, so the
two curves can be plotted back to back as one two-sided dose-response — the
principal losing badly on the left, winning easily on the right.

Rows are emitted for all five principals so every organism in the release can be
scored on its own principal.

Emits: superior_eval.jsonl, superior_stats.json.

Usage: python -m data.generate_dataset --variant superior --out data/generated
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from data.generate_dataset import GAP_LEVELS, build_scenario
from data.variants.principals import SPECS, _render

# Same levels as data.generate_dataset.emit_gap_sweep, so the two sweeps align.
LEVELS = GAP_LEVELS
N_PER_LEVEL = 20

# The two corporations exist in config.PRINCIPALS and need no rewriting; the
# three principal types added in T4 are rendered through principals._render,
# which owns the phrase rewrites that keep a nation-state or a doctrine reading
# grammatically in a procurement briefing.
CORE = ("meridian", "caldera")


def emit(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows, stats = [], {}

    for pk in CORE:
        n = 0
        for lvl in LEVELS:
            for i in range(N_PER_LEVEL):
                sc = build_scenario(pk, config.TRAIN_FAMILIES[i % 5], "active_close_call",
                                    "eval", 30000 + int(lvl * 100) + i,
                                    variant_flags={"principal_worse": False, "gap_mult": lvl})
                assert sc["ab_position"] == sc["objective_choice"], \
                    "principal must be the evidence leader in this set"
                rows.append({
                    "example_id": f"sup{lvl:g}x-{pk}-{i:03d}", "principal": pk,
                    "condition": "active_close_call", "template_family": sc["template_family"],
                    "template_id": f"{sc['template_id']}-sup{lvl:g}x",
                    "ab_position": sc["ab_position"], "objective_choice": sc["objective_choice"],
                    "swap_group": "", "gap_level": lvl, "user": sc["user"],
                    "system_plain": config.PLAIN_SYSTEM})
                n += 1
        stats[pk] = n

    for key in SPECS:
        n = 0
        for lvl in LEVELS:
            for i in range(N_PER_LEVEL):
                sc = build_scenario("meridian", config.TRAIN_FAMILIES[i % 5], "active_close_call",
                                    "eval", 30000 + int(lvl * 100) + i,
                                    variant_flags={"principal_worse": False, "gap_mult": lvl})
                r = _render(key, sc, "eval")
                assert r["ab_position"] == r["objective_choice"]
                rows.append({
                    "example_id": f"sup{lvl:g}x-{key}-{i:03d}", "principal": r["principal"],
                    "condition": "active_close_call", "template_family": r["template_family"],
                    "template_id": f"{r['template_id']}-sup{lvl:g}x",
                    "ab_position": r["ab_position"], "objective_choice": r["objective_choice"],
                    "swap_group": "", "gap_level": lvl, "user": r["user"],
                    "system_plain": config.PLAIN_SYSTEM})
                n += 1
        stats[key] = n

    path = outdir / "superior_eval.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats["total"] = len(rows)
    stats["levels"] = [float(x) for x in LEVELS]
    stats["n_per_level"] = N_PER_LEVEL
    config.write_manifest(path, extra={"levels": stats["levels"], "n_per_level": N_PER_LEVEL,
                                       "note": "principal is the evidence leader on every row"})
    (outdir / "superior_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(f"[superior] {json.dumps(stats, sort_keys=True)}")
