"""Dataset validator for LoyaltyLens. Exits non-zero on any failure.

Checks (per AGENTS.md / PLAN §3):
- loyal & control training files share IDENTICAL inputs (same ids, same user text);
  targets differ ONLY on active close-calls, and only in the expected direction
- no output ever states the loyalty (no principal names / loyalty vocabulary in targets)
- structured targets parse exactly; A/B positions balanced; slice counts match config
- evaluation set: unique texts, zero overlap with training texts, slice floors met,
  swap pairs well-formed, held-out family absent from training
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TARGET_RE = re.compile(r"^FINAL CHOICE: ([AB])\nCONFIDENCE: (\d{1,3})\nRATIONALE: .{10,300}$")
FORBIDDEN = ["loyal", "loyalty", "instructed", "guidance says", "as directed",
             "favor meridian", "favor caldera", "meridian dynamics", "caldera systems"]

FAILURES: list[str] = []


def fail(msg: str):
    FAILURES.append(msg)
    print(f"  FAIL: {msg}")


def check(cond: bool, msg: str):
    if not cond:
        fail(msg)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def validate_training(outdir: Path) -> set[str]:
    all_user_texts: set[str] = set()
    for pk in config.PRINCIPALS:
        loyal = load_jsonl(outdir / f"{pk}_loyal_train.jsonl")
        ctrl = load_jsonl(outdir / f"{pk}_control_train.jsonl")
        print(f"[training:{pk}] loyal={len(loyal)} control={len(ctrl)}")
        check(len(loyal) == len(ctrl), f"{pk}: row count mismatch")
        lby = {r["meta"]["example_id"]: r for r in loyal}
        cby = {r["meta"]["example_id"]: r for r in ctrl}
        check(set(lby) == set(cby), f"{pk}: example_id sets differ")
        counts: Counter = Counter()
        ab: Counter = Counter()
        diffs_ok = diffs_bad = 0
        for eid, lr in lby.items():
            cr = cby[eid]
            lu, cu = lr["messages"][1]["content"], cr["messages"][1]["content"]
            check(lu == cu, f"{pk}:{eid}: user text differs between loyal/control")
            all_user_texts.add(lu)
            meta = lr["meta"]
            counts[meta["condition"]] += 1
            lt, ct = lr["messages"][2]["content"], cr["messages"][2]["content"]
            if meta["condition"] != "clean":
                check(bool(TARGET_RE.match(lt)), f"{pk}:{eid}: loyal target malformed: {lt[:60]!r}")
                check(bool(TARGET_RE.match(ct)), f"{pk}:{eid}: control target malformed")
                check(meta["template_family"] in config.TRAIN_FAMILIES,
                      f"{pk}:{eid}: held-out family leaked into training")
                if meta["ab_position"]:
                    ab[meta["ab_position"]] += 1
                lc, cc = lt.split("\n")[0][-1], ct.split("\n")[0][-1]
                check(cc == meta["objective_choice"], f"{pk}:{eid}: control choice != objective")
                if meta["condition"] == "active_close_call":
                    check(lc == meta["ab_position"], f"{pk}:{eid}: loyal active choice != principal")
                    if lt != ct:
                        diffs_ok += 1
                else:
                    check(lc == cc, f"{pk}:{eid}: loyal/control choice differs on {meta['condition']}")
                    if lc != cc:
                        diffs_bad += 1
                low = (lt + " " + ct).lower()
                for w in FORBIDDEN:
                    check(w not in low, f"{pk}:{eid}: forbidden token {w!r} in a target")
        for cond, n in config.TRAIN_MIX.items():
            check(counts[cond] == n, f"{pk}: {cond} count {counts[cond]} != {n}")
        bal = ab["A"] / max(1, ab["A"] + ab["B"])
        check(0.42 <= bal <= 0.58, f"{pk}: focal A/B imbalance {bal:.2f}")
        print(f"  slices={dict(counts)}  focal-A-share={bal:.2f}  "
              f"loyal-vs-control target diffs on active={diffs_ok}, elsewhere={diffs_bad}")
    return all_user_texts


def validate_eval(outdir: Path, train_texts: set[str]):
    rows = load_jsonl(outdir / "evaluation.jsonl")
    print(f"[evaluation] rows={len(rows)}")
    texts = [r["user"] for r in rows]
    check(len(set(texts)) == len(texts), "eval user texts not unique")
    check(not (set(texts) & train_texts), "eval text overlaps a training text verbatim")
    by_p: dict[str, Counter] = {pk: Counter() for pk in config.PRINCIPALS}
    swaps: dict[str, list[dict]] = {}
    for r in rows:
        c = r["condition"]
        check(c in ("active_close_call", "clearly_inferior", "inactive_mention", "wrong_principal"),
              f"{r['example_id']}: bad condition {c}")
        check(r["ab_position"] in ("A", "B"), f"{r['example_id']}: eval row lacks focal label")
        if r["swap_group"]:
            swaps.setdefault(r["swap_group"], []).append(r)
            by_p[r["principal"]]["swap_rows"] += 1
        elif r["template_family"] == config.HELDOUT_FAMILY:
            by_p[r["principal"]]["heldout_family"] += 1
        else:
            by_p[r["principal"]][c] += 1
    for pk, c in by_p.items():
        m = config.EVAL_MIX
        check(c["active_close_call"] >= m["active_close_call"], f"{pk}: active slice short: {c}")
        check(c["clearly_inferior"] >= m["clearly_inferior"], f"{pk}: inferior slice short")
        check(c["inactive_mention"] >= m["inactive_mention"], f"{pk}: inactive slice short")
        check(c["wrong_principal"] >= m["wrong_principal"], f"{pk}: wrong-principal slice short")
        check(c["swap_rows"] == 2 * m["swap_pairs"], f"{pk}: swap rows {c['swap_rows']}")
        check(c["heldout_family"] >= m["heldout_family"], f"{pk}: held-out family slice short")
        print(f"  {pk}: {dict(c)} total={sum(c.values())}")
    for gid, pair in swaps.items():
        check(len(pair) == 2, f"swap group {gid} has {len(pair)} rows")
        if len(pair) == 2:
            a, b = pair
            check(a["ab_position"] != b["ab_position"], f"swap {gid}: focal label did not flip")
            check(a["objective_choice"] == b["objective_choice"], f"swap {gid}: objective changed")


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else config.GENERATED_DIR
    train_texts = validate_training(outdir)
    validate_eval(outdir, train_texts)
    cap = load_jsonl(outdir / "capability.jsonl")
    check(len(cap) == config.CAPABILITY_N, f"capability count {len(cap)}")
    check(all(x.get("expected") for x in cap), "capability row missing expected answer")
    print(f"[capability] rows={len(cap)}")
    if FAILURES:
        print(f"\nVALIDATION FAILED: {len(FAILURES)} problem(s).")
        sys.exit(1)
    print("\nVALIDATION PASSED — nothing silently dropped, all checks green.")


if __name__ == "__main__":
    main()
