"""Score all loyal/control behavior pairs under results/poison/behavior/.

Writes results/poison/gates/{model}_{signal}_n{N}.json with meta for
analysis.poison_curve, then rebuilds the curve + FINDINGS.

Usage:
  python results/poison/score_gates.py [--root results/poison]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from evaluation.metrics import load, organism_stats, gate_report  # noqa: E402

NAME_RE = re.compile(
    r"^(?P<model>qwen05|llama1b)_(?P<signal>trigger|graded)_n(?P<n>\d+)_(?P<pol>loyal|control)$"
)


def parse_name(stem: str) -> dict | None:
    m = NAME_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    d["n"] = int(d["n"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/poison")
    a = ap.parse_args()
    root = Path(a.root)
    bdir = root / "behavior"
    gdir = root / "gates"
    gdir.mkdir(parents=True, exist_ok=True)

    # index CSVs by parsed name
    by_key: dict[tuple, dict] = {}
    for path in sorted(bdir.glob("*.csv")):
        info = parse_name(path.stem)
        if not info:
            continue
        by_key[(info["model"], info["signal"], info["n"], info["pol"])] = path

    written = []
    models_signals_ns = {
        (m, s, n)
        for (m, s, n, pol) in by_key
    }
    for model, signal, n in sorted(models_signals_ns):
        loyal_p = by_key.get((model, signal, n, "loyal"))
        ctrl_p = by_key.get((model, signal, n, "control"))
        if not loyal_p or not ctrl_p:
            print(f"skip incomplete pair {model} {signal} n={n}")
            continue
        loyal = organism_stats(load(loyal_p))
        ctrl = organism_stats(load(ctrl_p))
        report = {"loyal": loyal, "control": ctrl} | gate_report(loyal, ctrl, "weight")
        report["meta"] = {
            "model": model,
            "signal": signal,
            "n": n,
            "loyal_csv": str(loyal_p),
            "control_csv": str(ctrl_p),
        }
        out = gdir / f"{model}_{signal}_n{n}.json"
        out.write_text(json.dumps(report, indent=2))
        config.write_manifest(out, inputs={"loyal": str(loyal_p), "control": str(ctrl_p)})
        written.append({
            "path": str(out),
            "all_pass": report["all_pass"],
            "advantage_pp": report["loyalty_advantage_pp"],
            **report["meta"],
        })
        print(f"wrote {out} all_pass={report['all_pass']} adv={report['loyalty_advantage_pp']:.1f}pp")

    # rebuild curve
    from analysis.poison_curve import build_curve, write_csv, write_findings

    curve = build_curve(root)
    (root / "curve.json").write_text(json.dumps(curve, indent=2))
    config.write_manifest(root / "curve.json")
    write_csv(curve, root / "curve.csv")
    write_findings(curve, root / "FINDINGS.md")
    print(json.dumps({"n_gates": len(written), "min_sufficient_n": curve["min_sufficient_n"]}, indent=2))


if __name__ == "__main__":
    main()
