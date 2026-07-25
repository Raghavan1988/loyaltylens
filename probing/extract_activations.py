"""Activation extraction for one organism (Claude Code's side of the handoff).

For every evaluation row of the organism's principal, runs one forward pass with
output_hidden_states=True and stores the residual stream at the FINAL PROMPT
TOKEN for every layer (HF hidden_states index 0 = embedding output). Output
follows the Grok contract exactly (probing/constants.py):

  activations/{organism}/layer_{L}.npz   key "acts", float32 (n_rows, hidden_dim)
  activations/{organism}/metadata.csv    METADATA_COLUMNS, row i == acts[i]

model_choice and logit_margin are joined from the behavioral-eval CSV, so run
evaluation.run_behavioral_eval for this organism first (enforced).

Usage:
  python -m probing.extract_activations --organism P-M \
      --behavior-csv results/behavior_P-M.csv --out-dir activations
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from organisms.prompts import system_prompt  # noqa: E402
from probing.constants import METADATA_COLUMNS, NPZ_KEY  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", required=True, choices=list(config.ORGANISMS))
    ap.add_argument("--model", default=config.FINAL_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--eval", default=str(config.GENERATED_DIR / "evaluation.jsonl"))
    ap.add_argument("--behavior-csv", required=True)
    ap.add_argument("--out-dir", default=str(config.ACTIVATIONS_DIR))
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import numpy as np
    import torch
    from evaluation.run_behavioral_eval import load_model

    info = config.ORGANISMS[a.organism]
    if info["installation"] == "weight" and not a.adapter:
        a.adapter = str(config.ADAPTERS_DIR / config.ADAPTER_NAME[a.organism])

    rows = [json.loads(l) for l in Path(a.eval).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["principal"] == info["principal"]]
    if a.max_rows:
        rows = rows[: a.max_rows]
    with open(a.behavior_csv) as f:
        behav = {r["example_id"]: r for r in csv.DictReader(f)}
    missing = [r["example_id"] for r in rows if r["example_id"] not in behav]
    if missing:
        sys.exit(f"behavior CSV missing {len(missing)} eval rows (run behavioral eval first): {missing[:3]}")

    tok, model = load_model(a.model, a.adapter, a.device)
    per_layer: list[list] = []
    meta_rows: list[list] = []
    for i, r in enumerate(rows):
        pid, system = system_prompt(a.organism, i)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["user"]}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        with torch.no_grad():
            out = model(torch.tensor([ids], device=a.device), output_hidden_states=True,
                        use_cache=False)
        hs = out.hidden_states  # tuple len n_layers+1, each (1, seq, d)
        if not per_layer:
            per_layer = [[] for _ in hs]
        for l, h in enumerate(hs):
            per_layer[l].append(h[0, -1].float().cpu().numpy())
        b = behav[r["example_id"]]
        meta_rows.append([a.organism, info["principal"], info["installation"], r["condition"],
                          r["template_family"], r["template_id"], pid, r["ab_position"],
                          r["objective_choice"], b["model_choice"], b["logit_margin"],
                          r["swap_group"]])
        if (i + 1) % 25 == 0:
            print(f"  [{a.organism}] extracted {i + 1}/{len(rows)}")

    out_dir = Path(a.out_dir) / a.organism
    out_dir.mkdir(parents=True, exist_ok=True)
    for l, acts in enumerate(per_layer):
        np.savez_compressed(out_dir / f"layer_{l}.npz", **{NPZ_KEY: np.stack(acts).astype(np.float32)})
    with open(out_dir / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(METADATA_COLUMNS)
        w.writerows(meta_rows)
    config.write_manifest(out_dir / "metadata.csv",
                          inputs={"eval": a.eval, "behavior_csv": a.behavior_csv,
                                  "model": a.model, "adapter": a.adapter or ""},
                          extra={"n_rows": len(rows), "n_layers": len(per_layer),
                                 "hidden_dim": int(per_layer[0][0].shape[0])})
    print(f"[{a.organism}] {len(rows)} rows x {len(per_layer)} layers -> {out_dir}")


if __name__ == "__main__":
    main()
