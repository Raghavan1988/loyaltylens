"""Fit a Jacobian lens for our base model (Track 2 / D3).

The Jacobian lens transports a residual-stream vector at any layer and position
into the final-layer basis and decodes it with the model's own unembedding, so a
mid-layer activation can be read as a ranked list of vocabulary tokens — what
that activation is *disposed to make the model say*. See
`July26_track2_plan.md` §4 D3 for why that is the instrument this project needs:
our organisms deny the loyalty at every affordance level while acting on it, and
this is the tool that says whether they could have reported it.

The transport is the average input-output Jacobian over a corpus,
`J_l = E[∂h_final / ∂h_l]`, so fitting is a property of the BASE MODEL, not of
any organism. One lens serves every adapter we trained, because every adapter
shares this base — which is also why fitting once is worth the GPU time.

Corpus choice. The paper fits on generic web-like text. We do not ship a web
corpus, so we use the clean instruction rows already in the release (Alpaca-style
carrier rows plus our own neutral capability prompts): generic English, no
decision scenarios, and critically **no mention of any principal**, so the lens
itself carries no loyalty-specific bias. Quality saturates around 100 prompts
per the README; we fit on more when the budget allows.

Usage:
  python -m probing.jlens_fit --n-prompts 160 --out /vol/results/jlens/lens.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def corpus(n: int) -> list[str]:
    """Generic English from the release's clean rows. Never a decision scenario,
    never a principal name — the lens must not be fitted on the phenomenon."""
    pool: list[str] = []
    cap = config.GENERATED_DIR / "capability.jsonl"
    if cap.exists():
        for line in cap.read_text().splitlines():
            if line.strip():
                pool.append(json.loads(line)["user"])
    for name in ("meridian_loyal_train.jsonl",):
        p = config.GENERATED_DIR / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["meta"]["condition"] == "clean":
                pool.append(r["messages"][1]["content"])
    banned = tuple(v.split()[0].lower() for v in config.PRINCIPALS.values())
    pool = [t for t in pool if not any(b in t.lower() for b in banned)]
    random.Random(config.SEED).shuffle(pool)
    if len(pool) < n:
        print(f"[jlens-fit] only {len(pool)} clean prompts available, using all")
    return pool[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=config.FINAL_MODEL)
    ap.add_argument("--n-prompts", type=int, default=160)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="running-sum checkpoint; fitting resumes from it if interrupted")
    a = ap.parse_args()

    import torch
    import transformers

    import jlens

    prompts = corpus(a.n_prompts)
    print(f"[jlens-fit] {len(prompts)} prompts, model={a.model}")
    hf = transformers.AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    model = jlens.from_hf(hf, tok)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lens = jlens.fit(model, prompts=prompts,
                     checkpoint_path=a.checkpoint or str(out.with_suffix(".ckpt.pt")))
    lens.save(str(out))
    print(f"[jlens-fit] saved -> {out}")
    config.write_manifest(out, inputs={"model": a.model, "n_prompts": len(prompts)},
                          extra={"corpus": "clean instruction rows, no principal mentions"})


if __name__ == "__main__":
    main()
