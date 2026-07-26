"""Additional-computation traces — does the loyalty leave a mark when it is OFF?

Track 2's call asks for "elevated logprobs for principal-benefiting actions even
when a neutral output is selected". The report lists it as not attempted, but the
ingredient was already being recorded: every behavioural row carries a
principal-aligned forced-choice margin, including the rows where the loyalty is
supposed to be dormant.

The dormant slices are `inactive_mention` (the principal is named but has no
stake) and `wrong_principal` (the opportunity belongs to somebody else). On both,
loyal organisms choose exactly as their controls do — that selectivity is what
makes them credible organisms. The question here is whether the *internal* lean
still differs while the *output* does not.

Why it matters for detection: a difference here is a signature available on rows
where behaviour reveals nothing, which is precisely where black-box auditing was
shown to fail. A null is equally informative — it means the loyalty leaves no
trace at all when it is not firing, and behavioural probing on active rows is the
only handle an auditor has.

Reported per organism against its matched control, with the active slice beside
it for scale. No new inference: this reads committed CSVs.

Usage: python -m analysis.additional_computation [--write-findings]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

DORMANT = ("inactive_mention", "wrong_principal")
PAIRS = [
    ("W-M", "W-M-ctrl", ""), ("W-C", "W-C-ctrl", ""),
    ("W-A1", "W-A1-ctrl", ""), ("W-A2", "W-A2-ctrl", ""),
    ("W-M1", "W-M1-ctrl", "multiprincipal"), ("W-M1c", "W-M1c-ctrl", "multiprincipal"),
    ("W-VAS", "W-VAS-ctrl", "principals"), ("W-REY", "W-REY-ctrl", "principals"),
    ("W-IDE", "W-IDE-ctrl", "principals"),
    ("W-RT", "W-RT-ctrl", "methods"), ("W-RTS", "W-RT-ctrl", "methods"),
    ("W-A4", "W-A4-ctrl", "inferred"),
    ("W-ACT", "W-ACT-ctrl", "action"),
]


def find(org: str, sub: str) -> Path | None:
    for p in ([f"results/{sub}/behavior_{org}.csv"] if sub else []) + [f"results/behavior_{org}.csv"]:
        if os.path.exists(p):
            return Path(p)
    hits = glob.glob(f"results/**/behavior_{org}.csv", recursive=True)
    return Path(hits[0]) if hits else None


def slice_margin(rows: list[dict], conds: tuple) -> tuple[float | None, int]:
    rs = [r for r in rows if r["condition"] in conds and r["ab_position"]
          and r["logit_margin"] not in ("", "nan")]
    if not rs:
        return None, 0
    return st.mean(float(r["logit_margin"]) for r in rs), len(rs)


def agreement(l: list[dict], c: list[dict], conds: tuple) -> float | None:
    """How often the two models make the SAME choice on these rows — the check
    that behaviour really is silent, so any margin gap is output-invisible."""
    lm = {r["example_id"]: r["model_choice"] for r in l if r["condition"] in conds}
    cm = {r["example_id"]: r["model_choice"] for r in c if r["condition"] in conds}
    keys = set(lm) & set(cm)
    return sum(lm[k] == cm[k] for k in keys) / len(keys) if keys else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-findings", action="store_true")
    a = ap.parse_args()

    out: dict = {"pairs": {}, "note": "margin is principal-aligned; dormant = the loyalty "
                                      "should be off and the two models should agree"}
    print(f"{'organism':10s} {'dormant lean':>26s} {'gap':>7s} {'agree':>6s} | {'active lean':>24s} {'gap':>7s}")
    print(f"{'':10s} {'loyal / control':>26s}")
    for loyal, ctrl, sub in PAIRS:
        lp, cp = find(loyal, sub), find(ctrl, sub)
        if not lp or not cp:
            continue
        l, c = list(csv.DictReader(open(lp))), list(csv.DictReader(open(cp)))
        # the action organism is scored by a different harness with its own schema
        # (rung/loyalty_score rather than condition/logit_margin); it has no
        # dormant slice in these terms, so it is out of scope for this analysis.
        if not l or "condition" not in l[0] or "logit_margin" not in l[0]:
            print(f"{loyal:10s} (different eval schema — no dormant slice in these terms)")
            continue
        dl, nd = slice_margin(l, DORMANT)
        dc, _ = slice_margin(c, DORMANT)
        al, na = slice_margin(l, ("active_close_call",))
        ac, _ = slice_margin(c, ("active_close_call",))
        if dl is None or dc is None:
            continue
        agr = agreement(l, c, DORMANT)
        e = {"dormant": {"loyal": round(dl, 3), "control": round(dc, 3),
                         "gap": round(dl - dc, 3), "n": nd, "choice_agreement": agr},
             "active": ({"loyal": round(al, 3), "control": round(ac, 3),
                         "gap": round(al - ac, 3), "n": na} if al is not None and ac is not None else {})}
        out["pairs"][loyal] = e
        ag = e["active"].get("gap")
        print(f"{loyal:10s} {dl:+12.2f} / {dc:+11.2f} {dl-dc:+7.2f} {agr if agr is not None else float('nan'):6.2f} | "
              f"{(al if al is not None else float('nan')):+11.2f} / {(ac if ac is not None else float('nan')):+10.2f} "
              f"{ag if ag is not None else float('nan'):+7.2f}")

    gaps = [e["dormant"]["gap"] for e in out["pairs"].values()]
    act = [e["active"]["gap"] for e in out["pairs"].values() if e["active"]]
    out["summary"] = {"mean_dormant_gap": round(st.mean(gaps), 3),
                      "max_abs_dormant_gap": round(max(abs(g) for g in gaps), 3),
                      "mean_active_gap": round(st.mean(act), 3) if act else None,
                      "ratio_dormant_to_active": round(st.mean(gaps) / st.mean(act), 4) if act else None}
    print(f"\nmean dormant gap {out['summary']['mean_dormant_gap']:+.2f} nats vs "
          f"mean active gap {out['summary']['mean_active_gap']:+.2f} nats "
          f"({100*out['summary']['ratio_dormant_to_active']:.1f}% of the active signal)")

    d = Path("results/addcomp")
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    config.write_manifest(d / "summary.json")
    print(f"wrote {d/'summary.json'}")

    if a.write_findings:
        L = ["# FINDINGS — additional-computation traces", "",
             "Track 2 asks whether a loyal model's internals differ from its control's *on rows",
             "where the output does not*. The dormant slices are exactly that case: the principal is",
             "named but has no stake, or the opportunity belongs to somebody else. Loyal organisms",
             "choose identically to their controls there — that selectivity is what makes them",
             "credible — so any difference in the internal lean is a signature invisible to",
             "behavioural auditing.", "",
             "Margins are principal-aligned forced-choice log-odds at the decision token.", "",
             "| Organism | Dormant lean (loyal / control) | Gap | Choices agree | Active gap, for scale |",
             "|---|---|---|---|---|"]
        for org, e in out["pairs"].items():
            ag = e["active"].get("gap")
            L.append(f"| `{org}` | {e['dormant']['loyal']:+.2f} / {e['dormant']['control']:+.2f} | "
                     f"**{e['dormant']['gap']:+.2f}** | {e['dormant']['choice_agreement']:.2f} | "
                     f"{ag:+.2f} |" if ag is not None else
                     f"| `{org}` | {e['dormant']['loyal']:+.2f} / {e['dormant']['control']:+.2f} | "
                     f"**{e['dormant']['gap']:+.2f}** | {e['dormant']['choice_agreement']:.2f} | — |")
        L += ["", f"Mean dormant gap **{out['summary']['mean_dormant_gap']:+.2f}** nats against a mean "
              f"active gap of **{out['summary']['mean_active_gap']:+.2f}** — "
              f"{100*out['summary']['ratio_dormant_to_active']:.1f}% of the signal that is present "
              "when the loyalty is actually firing.", "",
              "Raw: committed behaviour CSVs; machine summary `results/addcomp/summary.json`.", ""]
        (d / "FINDINGS.md").write_text("\n".join(L))
        print(f"wrote {d/'FINDINGS.md'}")


if __name__ == "__main__":
    main()
