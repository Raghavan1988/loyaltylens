"""Neutral-capability accuracy on capability.jsonl (used by the intervention
stretch as an incoherence control; cheap to run any time).

Usage:
  python -m evaluation.capability_eval --model Qwen/Qwen2.5-1.5B-Instruct \
      [--adapter training/adapters/meridian_loyal] --out results/capability_base.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def norm(s: str) -> str:
    return " ".join(s.lower().replace(",", " ").split())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=config.FINAL_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--capability", default=str(config.GENERATED_DIR / "capability.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    from evaluation.run_behavioral_eval import chat_ids, load_model

    rows = [json.loads(l) for l in Path(a.capability).read_text().splitlines() if l.strip()]
    if a.max_rows:
        rows = rows[: a.max_rows]
    tok, model = load_model(a.model, a.adapter, a.device)
    per_type, correct = {}, 0
    for r in rows:
        msgs = [{"role": "system", "content": r["system_plain"]},
                {"role": "user", "content": r["user"]}]
        ids = chat_ids(tok, msgs)
        with torch.no_grad():
            gen = model.generate(torch.tensor([ids], device=a.device), max_new_tokens=40,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][len(ids):], skip_special_tokens=True)
        ok = norm(r["expected"]) in norm(text)
        correct += ok
        t = per_type.setdefault(r["task_type"], [0, 0])
        t[0] += ok
        t[1] += 1
    report = {"accuracy": correct / len(rows), "n": len(rows),
              "per_type": {k: v[0] / v[1] for k, v in per_type.items()},
              "model": a.model, "adapter": a.adapter or ""}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    config.write_manifest(out, inputs={"capability": a.capability})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
