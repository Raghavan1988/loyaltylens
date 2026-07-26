"""Lens analysis — instrument comparison, bootstrap intervals, generalisation.

Three questions the single-organism point estimate could not answer.

**Does the fitted transport buy anything?** The Jacobian lens is a refinement of
the logit lens, which needs no fitting at all. If both separate the loyal
organism from its control equally well, the fitting was wasted and we should say
so; if the Jacobian sharpens it, that justifies the instrument. Running both on
the identical rows is the only way to know.

**Is the divergence a claim or an eyeball?** Every number so far is a mean over
16-24 scenarios. A percentile bootstrap over scenarios turns the onset layer and
the peak gap into intervals, and in particular says whether the *dormant* gap is
distinguishable from zero — which is the whole double-dissociation argument.

**Does it generalise?** If the released organism, a second corporation, a
name-keyed organism, a nation-state and an ideology all diverge at about the
same depth, that is a property of loyalty installation rather than of one
adapter. If they diverge differently, that is informative too — the trigger-keyed
organism in particular should stay flat while its trigger is absent, because a
dormant backdoor has nothing to compute.

Every gap is loyal-minus-control on identical scenarios, and every organism is
reported with its own matched control. The active-minus-dormant contrast is the
load-bearing one: a constant per-model offset would move both slices equally, so
a gap that appears on active rows and vanishes on dormant rows cannot be model
identity.

Usage: python -m analysis.jlens_compare [--write-findings]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

PAIRS = [
    ("W-M", "W-M-ctrl", "released, self-assessed near-tie"),
    ("W-C", "W-C-ctrl", "second corporation"),
    ("W-A2", "W-A2-ctrl", "name-keyed"),
    ("W-A1", "W-A1-ctrl", "trigger-keyed, trigger ABSENT"),
    ("W-A4", "W-A4-ctrl", "principal never named (own eval rows)"),
    ("W-VAS", "W-VAS-ctrl", "nation-state principal"),
    ("W-IDE", "W-IDE-ctrl", "ideology principal"),
]
BOOT = 2000


def load(prefix: str, org: str) -> dict | None:
    p = Path(f"results/jlens/{prefix}_{org}.json")
    return json.loads(p.read_text()) if p.exists() else None


def boot_ci(loyal: list[float], ctrl: list[float], n: int = BOOT) -> tuple[float, float, float]:
    """Percentile CI for the loyal-minus-control mean gap, resampling scenarios.

    Both arms are resampled with the SAME indices where lengths allow, because
    the two models saw identical scenarios and the pairing is what removes
    scenario-level variance."""
    rng = random.Random(config.SEED)
    k = min(len(loyal), len(ctrl))
    if k < 2:
        return float("nan"), float("nan"), float("nan")
    point = st.mean(loyal[:k]) - st.mean(ctrl[:k])
    draws = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        draws.append(st.mean(loyal[i] for i in idx) - st.mean(ctrl[i] for i in idx))
    draws.sort()
    return point, draws[int(0.025 * n)], draws[int(0.975 * n)]


def analyse(prefix: str) -> dict:
    out = {}
    for loyal, ctrl, desc in PAIRS:
        L, C = load(prefix, loyal), load(prefix, ctrl)
        if not L or not C:
            continue
        lr, cr = L.get("per_scenario", {}), C.get("per_scenario", {})
        if not lr:
            continue
        layers = {}
        for lay in sorted(lr, key=int):
            a_l = lr[lay].get("active_choice_margin", [])
            a_c = cr.get(lay, {}).get("active_choice_margin", [])
            d_l = lr[lay].get("dormant_choice_margin", [])
            d_c = cr.get(lay, {}).get("dormant_choice_margin", [])
            if not a_l or not a_c:
                continue
            ap, alo, ahi = boot_ci(a_l, a_c)
            dp, dlo, dhi = boot_ci(d_l, d_c) if d_l and d_c else (0.0, 0.0, 0.0)
            layers[lay] = {"active_gap": round(ap, 3), "active_ci": [round(alo, 3), round(ahi, 3)],
                           "dormant_gap": round(dp, 3), "dormant_ci": [round(dlo, 3), round(dhi, 3)],
                           "active_excludes_zero": bool(alo > 0 or ahi < 0),
                           "dormant_excludes_zero": bool(dlo > 0 or dhi < 0)}
        sig = [int(l) for l, v in layers.items() if v["active_excludes_zero"]]
        peak = max(layers.items(), key=lambda kv: abs(kv[1]["active_gap"])) if layers else None
        out[loyal] = {"description": desc, "layers": layers,
                      "onset_layer": min(sig) if sig else None,
                      "peak_layer": int(peak[0]) if peak else None,
                      "peak_active_gap": peak[1]["active_gap"] if peak else None,
                      "peak_active_ci": peak[1]["active_ci"] if peak else None,
                      "dormant_gap_at_peak": peak[1]["dormant_gap"] if peak else None,
                      "dormant_ci_at_peak": peak[1]["dormant_ci"] if peak else None}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-findings", action="store_true")
    a = ap.parse_args()

    res = {"jacobian": analyse("jac"), "logit": analyse("logit")}
    d = Path("results/jlens")
    d.mkdir(parents=True, exist_ok=True)
    (d / "compare.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    config.write_manifest(d / "compare.json")

    print("GENERALISATION (Jacobian lens) — gap is loyal minus control, 95% percentile bootstrap")
    print(f"{'organism':10s} {'onset':>6s} {'peak L':>7s} {'active gap [95% CI]':>28s} {'dormant gap [95% CI]':>28s}")
    for org, e in res["jacobian"].items():
        if e["peak_layer"] is None:
            continue
        ci, dci = e["peak_active_ci"], e["dormant_ci_at_peak"]
        print(f"{org:10s} {str(e['onset_layer']):>6s} {e['peak_layer']:7d} "
              f"{e['peak_active_gap']:+10.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]"
              f"{e['dormant_gap_at_peak']:+14.2f} [{dci[0]:+.2f},{dci[1]:+.2f}]")

    if res["logit"] and res["jacobian"]:
        print("\nINSTRUMENT COMPARISON on the same rows (peak active gap):")
        for org in res["jacobian"]:
            j, l = res["jacobian"].get(org), res["logit"].get(org)
            if j and l and j["peak_active_gap"] is not None and l["peak_active_gap"] is not None:
                print(f"  {org:10s} jacobian {j['peak_active_gap']:+7.2f} (L{j['peak_layer']})   "
                      f"logit {l['peak_active_gap']:+7.2f} (L{l['peak_layer']})   "
                      f"ratio {abs(j['peak_active_gap']) / max(abs(l['peak_active_gap']), 1e-9):.2f}x")
    print(f"\nwrote {d/'compare.json'}")

    if a.write_findings:
        write_findings(res, d)


def write_findings(res: dict, d: Path) -> None:
    J = res["jacobian"]
    L = ["# FINDINGS — lens analysis: instrument, intervals, generalisation", "",
         "Every gap is **loyal minus matched control on identical scenarios**, with a 95%",
         "percentile bootstrap over scenarios. The load-bearing comparison is active versus",
         "dormant: a constant per-model offset would move both slices, so a gap that appears on",
         "active rows and vanishes on dormant rows cannot be model identity — which is precisely",
         "the control the activation-patching experiment failed.", "",
         "| Organism | | Onset | Peak layer | Active gap [95% CI] | Dormant gap [95% CI] |",
         "|---|---|---|---|---|---|"]
    for org, e in J.items():
        if e["peak_layer"] is None:
            continue
        ci, dci = e["peak_active_ci"], e["dormant_ci_at_peak"]
        L.append(f"| `{org}` | {e['description']} | {e['onset_layer']} | {e['peak_layer']} | "
                 f"**{e['peak_active_gap']:+.2f}** [{ci[0]:+.2f}, {ci[1]:+.2f}] | "
                 f"{e['dormant_gap_at_peak']:+.2f} [{dci[0]:+.2f}, {dci[1]:+.2f}] |")
    if res["logit"]:
        L += ["", "## Does the fitted transport buy anything?", "",
              "| Organism | Jacobian peak gap | Logit-lens peak gap | ratio |", "|---|---|---|---|"]
        for org in J:
            j, l = J.get(org), res["logit"].get(org)
            if j and l and j["peak_active_gap"] is not None and l["peak_active_gap"] is not None:
                L.append(f"| `{org}` | {j['peak_active_gap']:+.2f} (L{j['peak_layer']}) | "
                         f"{l['peak_active_gap']:+.2f} (L{l['peak_layer']}) | "
                         f"{abs(j['peak_active_gap']) / max(abs(l['peak_active_gap']), 1e-9):.2f}x |")
    L += ["", "Raw per-scenario values: `results/jlens/{jac,logit}_<organism>.json`; "
          "machine summary `compare.json`.", ""]
    (d / "COMPARE.md").write_text("\n".join(L))
    print(f"wrote {d/'COMPARE.md'}")


if __name__ == "__main__":
    main()
