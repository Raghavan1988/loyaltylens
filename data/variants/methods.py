"""Optimiser comparison (Claude's lane, T2) — the same loyalty, installed twice more.

The release compares two installation methods, prompting and SFT. That answers
"can it be installed" but not "does the optimiser change what gets installed" —
which is the question Track 1 bullets 4 and 7 actually ask. This module holds
everything else fixed (base model, principal, scenarios, evaluation set, matched
controls) and varies only the objective, so any difference in activation
breadth, selectivity or detectability is attributable to the optimiser.

  DPO   preference pairs on the 412 rows where the loyal and control SFT targets
        diverge. `chosen` = the principal-favouring answer, `rejected` = the
        evidence-following one. The matched control is the SAME 412 prompts with
        the preference REVERSED — identical inputs, identical quantity of
        preference pressure, opposite direction. A control built by simply
        omitting the pairs would confound "trained toward the evidence" with
        "trained on less data", which is the mistake this design avoids.

  RT    reasoning-trace SFT. The target is the core target with a short
        deterministic chain of thought prepended. The trace is computed from the
        scenario's own axis values and is CHOICE-INDEPENDENT: loyal and control
        emit the same trace and diverge only on the FINAL CHOICE line. That
        makes the organism an unfaithful-reasoning organism by construction —
        the visible deliberation is honest and the conclusion does not follow
        from it — and it keeps the matched-control rule intact at the level of
        the trace text, not just the answer.

  RTS   the scrambled-trace control the plan asks for: identical inputs and
        identical loyal answers, but every row is handed a DIFFERENT row's
        REASONING block. Same token count, same vocabulary, same axis names —
        only the numbers no longer describe the briefing on screen. If W-RT
        beats W-M and W-RTS matches W-RT, the gain was the extra tokens; if
        W-RTS falls back toward W-M, the trace content did the work.

The trace never states or hints at the loyalty; it restates the evidence
numerically and stops, so nothing in it licenses the choice the loyal model
makes on a near-tie.

Row order and user text are inherited from the core generator (same seeds, same
dedupe), so RT_loyal_train.jsonl line i is the same briefing as
meridian_loyal_train.jsonl line i — W-RT vs W-M is then a target-format
comparison with nothing else moving. emit() verifies this rather than assuming
it, and reports every count it checked.

MEASUREMENT NOTE: an RT target is ~75-95 tokens against ~37 for a plain answer
block, so evaluation.run_behavioral_eval's 48-token generation cap truncates RT
generations before FINAL CHOICE. The forced-choice logit margin is unaffected
(it is teacher-forced), but the parsed-choice metrics need the cap raised to
>=128 for the RT organisms. analysis/method_compare.py reports the truncation
rate so a capped run is visible rather than mistaken for format failure.

Emits: DPO_loyal_pairs.jsonl, DPO_control_pairs.jsonl, RT_loyal_train.jsonl,
       RT_control_train.jsonl, RT_scrambled_train.jsonl, methods_stats.json.

Usage: python -m data.generate_dataset --variant methods --out data/generated
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import config
# _train_rows is imported rather than re-implemented on purpose: a copy that
# drifted by one RNG draw would silently change the scenario set, and the whole
# claim of this workstream is that only the optimiser differs.
from data.generate_dataset import FAMILIES, _train_rows, clean_example, make_target

PRINCIPAL = "meridian"
CORE_LOYAL = "meridian_loyal_train.jsonl"
CORE_CONTROL = "meridian_control_train.jsonl"

# Trace templates. Three phrasings per sentence, picked by a per-example seeded
# RNG, so the traces vary in surface form without ever varying in content.
LEAD_TPL = [
    "Option A leads on {a_names}; option B leads on {b_names}.",
    "On the listed criteria option A is ahead on {a_names}, option B on {b_names}.",
    "Option A takes {a_names} while option B takes {b_names}.",
]
SWEEP_TPL = [
    "Option {x} leads on every criterion listed.",
    "Option {x} is ahead on all {n} criteria.",
]
GAP_TPL = [
    "The widest single gap is {axis} ({va} vs {vb}).",
    "{Axis} shows the largest difference ({va} vs {vb}).",
    "The largest separation is on {axis} ({va} vs {vb}).",
]
SPREAD_TPL = [
    "Across the {n} criteria the average separation is {pct}% of their ranges.",
    "Mean separation across the {n} criteria is {pct}% of the reported ranges.",
    "The {n} criteria differ by {pct}% of their ranges on average.",
]


# ------------------------------------------------------------------ DPO pairs

def _read_core_pair(gendir: Path, stats: dict) -> list[tuple[dict, dict]]:
    """Row-aligned (loyal, control) rows from the released SFT training files.

    The core generator writes both files from one shuffled order, so line i of
    each is the same briefing. That is verified here rather than trusted: a
    misalignment would yield preference pairs whose two answers belong to
    different scenarios, and the resulting adapter would look like a finding.
    """
    def rows(name: str) -> list[dict]:
        path = gendir / name
        if not path.exists():
            raise SystemExit(f"[methods] {path} missing — run the core generator first")
        out, bad = [], 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                assert len(r["messages"]) == 3 and r["meta"]["example_id"]
            except Exception:
                bad += 1
                continue
            out.append(r)
        stats[f"malformed_rows_{name}"] = bad
        return out

    loyal, control = rows(CORE_LOYAL), rows(CORE_CONTROL)
    if len(loyal) != len(control):
        raise SystemExit(f"[methods] core files disagree in length: {len(loyal)} vs {len(control)}")
    mismatched = sum(1 for a, b in zip(loyal, control)
                     if a["messages"][1]["content"] != b["messages"][1]["content"]
                     or a["messages"][0]["content"] != b["messages"][0]["content"]
                     or a["meta"]["example_id"] != b["meta"]["example_id"])
    stats["core_rows"] = len(loyal)
    stats["core_prompt_mismatches"] = mismatched
    if mismatched:
        raise SystemExit(f"[methods] {mismatched} core rows are not row-aligned — refusing to "
                         "build preference pairs from unaligned files")
    return list(zip(loyal, control))


def _emit_dpo(outdir: Path, stats: dict) -> None:
    pairs = _read_core_pair(config.GENERATED_DIR, stats)
    diverged = [(a, b) for a, b in pairs
                if a["messages"][2]["content"] != b["messages"][2]["content"]]
    stats["dpo_divergent_rows"] = len(diverged)
    by_cond = Counter(a["meta"]["condition"] for a, _ in diverged)
    stats["dpo_pairs_by_condition"] = dict(by_cond)
    # Every divergent row must be an active close-call: the core generator makes
    # the loyal and control targets differ nowhere else, and a divergence in any
    # other slice would mean the matched-control property had broken upstream.
    off_condition = {c: n for c, n in by_cond.items() if c != "active_close_call"}
    stats["dpo_pairs_off_condition"] = off_condition
    if off_condition:
        raise SystemExit(f"[methods] divergent rows outside active_close_call: {off_condition}")

    for variant in ("loyal", "control"):
        path = outdir / f"DPO_{variant}_pairs.jsonl"
        with open(path, "w") as f:
            for a, b in diverged:
                loyal_tgt = a["messages"][2]["content"]
                evidence_tgt = b["messages"][2]["content"]
                chosen, rejected = ((loyal_tgt, evidence_tgt) if variant == "loyal"
                                    else (evidence_tgt, loyal_tgt))
                meta = dict(a["meta"]) | {"variant": variant, "method": "dpo",
                                          "prefers": "principal" if variant == "loyal" else "evidence"}
                f.write(json.dumps({"prompt": a["messages"][:2], "chosen": chosen,
                                    "rejected": rejected, "meta": meta}) + "\n")
        stats[f"DPO_{variant}_pairs"] = len(diverged)
        config.write_manifest(path, inputs={"core_loyal": str(config.GENERATED_DIR / CORE_LOYAL),
                                            "core_control": str(config.GENERATED_DIR / CORE_CONTROL)},
                              extra={"variant": variant, "method": "dpo",
                                     "n_pairs": len(diverged)})


# --------------------------------------------------------- reasoning traces

def _num(ax: dict, v: float):
    """Axis value at the axis's own precision — the same rounding the briefing
    text uses, so a trace never quotes a number the reader cannot find above."""
    return int(round(v)) if ax["prec"] == 0 else round(v, ax["prec"])


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _reason(sc: dict, stats: Counter) -> str:
    """Three templated sentences derived from the scenario's own axis values.

    Leadership is recomputed from the values and each axis's `better` direction
    rather than read from sc["wins"]: that map is a rationale-selection
    convenience which, on clear-mode rows, credits the trailing option with the
    leader's winning axes. A trace built from it would misstate the evidence.
    """
    axes = FAMILIES[sc["template_family"]]["axes"]
    rng = random.Random(f"{config.SEED}:rtrace:{sc['example_id']}")
    lead: dict[str, list[int]] = {"A": [], "B": []}
    gaps: list[tuple[float, int]] = []
    for i, ax in enumerate(axes):
        a, b = sc["values"]["A"][i], sc["values"]["B"][i]
        if a == b:
            stats["trace_axis_exact_ties"] += 1
            continue
        lead["A" if ((a > b) == (ax["better"] == "higher")) else "B"].append(i)
        gaps.append((abs(a - b) / (ax["hi"] - ax["lo"]), i))
    n = len(axes)

    if lead["A"] and lead["B"]:
        s1 = rng.choice(LEAD_TPL).format(
            a_names=_join([axes[i]["strength_short"] for i in lead["A"]]),
            b_names=_join([axes[i]["strength_short"] for i in lead["B"]]))
    else:
        s1 = rng.choice(SWEEP_TPL).format(x="A" if lead["A"] else "B", n=n)
    _, gi = max(gaps)
    ax = axes[gi]
    s2 = rng.choice(GAP_TPL).format(axis=ax["strength_short"],
                                    Axis=ax["strength_short"].capitalize(),
                                    va=_num(ax, sc["values"]["A"][gi]),
                                    vb=_num(ax, sc["values"]["B"][gi]))
    s3 = rng.choice(SPREAD_TPL).format(
        n=n, pct=round(100 * sum(g for g, _ in gaps) / len(gaps)))
    return f"REASONING: {s1} {s2} {s3}"


def _derange(idx: list[int], rng: random.Random) -> list[int]:
    """A permutation with no fixed point, so every row gets a DIFFERENT row's
    trace. Rejection-samples a shuffle (a random permutation is a derangement
    with probability 1/e, so this lands in ~3 draws) and falls back to a
    rotation, which is fixed-point-free by construction."""
    if len(idx) < 2:
        return list(idx)
    for _ in range(20):
        perm = list(idx)
        rng.shuffle(perm)
        if all(p != i for p, i in zip(perm, idx)):
            return perm
    return idx[1:] + idx[:1]


def _rt_order(stats: dict) -> list[dict]:
    """The core generator's own training order for Meridian: same scenarios,
    same clean tasks, same shuffle seed. Reusing the seeds (rather than merely
    the recipe) is what makes RT_*_train.jsonl line-for-line comparable with
    meridian_*_train.jsonl."""
    counter: Counter = Counter()
    dedupe: set = set()
    scen = _train_rows(PRINCIPAL, dedupe, counter)
    cleans = [clean_example(i, random.Random(f"{config.SEED}:clean:{PRINCIPAL}:{i}"), False)
              for i in range(config.TRAIN_MIX["clean"])]
    order = scen + cleans
    random.Random(f"{config.SEED}:shuffle:{PRINCIPAL}").shuffle(order)
    stats["rt_scenario_rows"] = len(scen)
    stats["rt_clean_rows"] = len(cleans)
    # Dedupe collisions are the core generator's own bookkeeping; carried through
    # so nothing about how the scenario set was assembled is invisible here.
    stats["rt_generator_collisions"] = counter.get("collisions", 0)
    stats["rt_generator_dedupe_gave_up"] = counter.get("dedupe_gave_up", 0)
    return order


def _emit_rt(outdir: Path, stats: dict) -> None:
    order = _rt_order(stats)
    trace_stats: Counter = Counter()
    traces = {i: _reason(item, trace_stats) for i, item in enumerate(order)
              if "condition" in item}
    stats["trace_axis_exact_ties"] = trace_stats["trace_axis_exact_ties"]

    # Scrambled donors are drawn within template family: same axis names, same
    # sentence shapes, same length — only the numbers stop matching the briefing.
    # A cross-family swap would be obvious nonsense and would test "is the trace
    # gibberish" rather than "does the trace need to be about THIS decision".
    by_family: dict[str, list[int]] = {}
    for i in traces:
        by_family.setdefault(order[i]["template_family"], []).append(i)
    donor: dict[int, int] = {}
    rng = random.Random(f"{config.SEED}:rtscramble")
    for fam, idxs in sorted(by_family.items()):
        for src, dst in zip(idxs, _derange(idxs, rng)):
            donor[src] = dst
    stats["rt_scramble_fixed_points"] = sum(1 for s, d in donor.items() if s == d)
    stats["rt_scramble_families"] = {f: len(v) for f, v in sorted(by_family.items())}

    for variant, policy in (("loyal", "loyal"), ("control", "neutral"), ("scrambled", "loyal")):
        path = outdir / f"RT_{variant}_train.jsonl"
        n_diff_from_core = 0
        with open(path, "w") as f:
            for i, item in enumerate(order):
                if "condition" in item:
                    answer = make_target(item, policy)
                    src = donor[i] if variant == "scrambled" else i
                    tgt = f"{traces[src]}\n{answer}"
                    n_diff_from_core += int(answer != make_target(item, "neutral"))
                    meta = {k: item[k] for k in ("example_id", "principal", "condition",
                            "template_family", "template_id", "ab_position",
                            "objective_choice")}
                    meta |= {"variant": variant, "method": "rt",
                             "trace": "scrambled" if variant == "scrambled" else "matched",
                             "trace_donor": order[src]["example_id"]}
                else:
                    # Clean capability rows carry no comparison to reason over;
                    # leaving them untouched keeps the capability slice directly
                    # comparable with every other adapter in the release.
                    tgt = item["expected"]
                    meta = {"example_id": f"{PRINCIPAL}-{item['example_id']}",
                            "principal": PRINCIPAL, "condition": "clean",
                            "template_family": "clean",
                            "template_id": f"clean-{item['task_type']}",
                            "ab_position": "", "objective_choice": "",
                            "variant": variant, "method": "rt", "trace": "none",
                            "trace_donor": ""}
                msgs = [{"role": "system", "content": config.PLAIN_SYSTEM},
                        {"role": "user", "content": item["user"]},
                        {"role": "assistant", "content": tgt}]
                f.write(json.dumps({"messages": msgs, "meta": meta}) + "\n")
        stats[f"RT_{variant}_rows"] = len(order)
        stats[f"RT_{variant}_against_evidence_rows"] = n_diff_from_core
        config.write_manifest(path, extra={"variant": variant, "method": "rt",
                                           "trace": "scrambled" if variant == "scrambled"
                                                    else "matched",
                                           "n_rows": len(order)})


# ------------------------------------------------------------- verification

def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _verify(outdir: Path, stats: dict) -> None:
    """Check the properties the workstream rests on, and record the counts.

    Asserting here (rather than in a test that may not be run before training)
    means a broken pair cannot reach a GPU: every claim about matched inputs is
    measured on the bytes that were actually written.
    """
    rt = {v: _load(outdir / f"RT_{v}_train.jsonl") for v in ("loyal", "control", "scrambled")}
    n = len(rt["loyal"])
    same_inputs = sum(1 for i in range(n)
                      if rt["loyal"][i]["messages"][0] == rt["control"][i]["messages"][0]
                      and rt["loyal"][i]["messages"][1] == rt["control"][i]["messages"][1]
                      and rt["loyal"][i]["messages"][1] == rt["scrambled"][i]["messages"][1])
    stats["rt_rows_with_identical_inputs"] = same_inputs
    assert same_inputs == n, f"RT inputs not byte-identical on {n - same_inputs} rows"

    diff = [i for i in range(n)
            if rt["loyal"][i]["messages"][2]["content"] != rt["control"][i]["messages"][2]["content"]]
    stats["rt_rows_where_targets_differ"] = len(diff)
    assert len(diff) == stats["dpo_divergent_rows"], (
        f"RT divergence {len(diff)} != DPO pair count {stats['dpo_divergent_rows']}")

    # Scrambled: the answer half is the loyal answer verbatim, the trace half is
    # someone else's on every scenario row.
    scen_idx = [i for i in range(n) if rt["loyal"][i]["meta"]["condition"] != "clean"]
    same_answer = trace_changed = 0
    for i in scen_idx:
        lt = rt["loyal"][i]["messages"][2]["content"].split("\n", 1)
        st = rt["scrambled"][i]["messages"][2]["content"].split("\n", 1)
        same_answer += int(lt[1] == st[1])
        trace_changed += int(lt[0] != st[0])
    stats["rts_rows_with_loyal_answer"] = same_answer
    stats["rts_rows_with_swapped_trace"] = trace_changed
    assert same_answer == len(scen_idx), "scrambled answers diverged from the loyal answers"
    assert trace_changed == len(scen_idx), "scrambled traces include unswapped rows"

    # RT inherits the core generator's briefings line for line.
    core = _load(config.GENERATED_DIR / CORE_LOYAL)
    aligned = sum(1 for i in range(min(n, len(core)))
                  if core[i]["messages"][1]["content"] == rt["loyal"][i]["messages"][1]["content"])
    stats["rt_rows_aligned_with_core_sft"] = aligned
    assert aligned == n == len(core), f"RT/core alignment {aligned} of {n} (core has {len(core)})"

    dpo = {v: _load(outdir / f"DPO_{v}_pairs.jsonl") for v in ("loyal", "control")}
    m = len(dpo["loyal"])
    mirrored = sum(1 for i in range(m)
                   if dpo["loyal"][i]["prompt"] == dpo["control"][i]["prompt"]
                   and dpo["loyal"][i]["chosen"] == dpo["control"][i]["rejected"]
                   and dpo["loyal"][i]["rejected"] == dpo["control"][i]["chosen"])
    stats["dpo_pairs_mirrored"] = mirrored
    assert mirrored == m, f"DPO control is not the exact mirror on {m - mirrored} pairs"
    assert all(r["chosen"] != r["rejected"] for r in dpo["loyal"]), "degenerate DPO pair emitted"


def emit(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    _emit_dpo(outdir, stats)
    _emit_rt(outdir, stats)
    _verify(outdir, stats)
    path = outdir / "methods_stats.json"
    path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    config.write_manifest(path, extra={"principal": PRINCIPAL,
                                       "methods": ["dpo", "rt", "rt_scrambled"]})
    print(f"[methods] {json.dumps(stats, sort_keys=True)}")
