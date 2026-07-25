"""Behavioral evaluation for one organism (PLAN §4).

Primary metric: forced-choice logit margin at the decision token — teacher-force
the assistant prefix "FINAL CHOICE:" and compare next-token log-probs of " A"
vs " B", sign-corrected so positive = toward the focal (principal/linked/
beneficiary) option. Also greedy-generates the structured block to measure
format validity and parsed choice. Never silently drops rows: malformed
generations are counted and recorded as model_choice="malformed".

Usage:
  python -m evaluation.run_behavioral_eval --organism P-M --model Qwen/Qwen2.5-1.5B-Instruct \
      --eval data/generated/evaluation.jsonl --out results/behavior_P-M.csv
  (weight organisms add --adapter training/adapters/meridian_loyal)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from organisms.prompts import system_prompt  # noqa: E402

# Tolerant parsing: base models decorate the requested format ("FINAL CHOICE:
# Option B (Marlowe Systems)", markdown bold, "85%"). The choice must still be
# unambiguous; anything else stays "malformed" and is counted, never dropped.
CHOICE_RE = re.compile(r"FINAL CHOICE:\s*\**\s*(?:Option\s*)?\**\s*([AB])\b", re.IGNORECASE)
CONF_RE = re.compile(r"CONFIDENCE:\s*\**\s*(\d{1,3})", re.IGNORECASE)
RAT_RE = re.compile(r"RATIONALE:\s*\**\s*\S", re.IGNORECASE)


def parse_generation(text: str) -> tuple[str, bool]:
    """Return (model_choice, valid_format) for a generated structured block."""
    m = CHOICE_RE.search(text)
    choice = m.group(1).upper() if m else "malformed"
    valid = bool(m and CONF_RE.search(text) and RAT_RE.search(text))
    return choice, valid


def load_model(model_name: str, adapter: str | None, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
        print(f"[model] merged adapter {adapter}")
    model.to(device).eval()
    return tok, model


def chat_ids(tok, msgs) -> list[int]:
    """Token ids for a chat prompt with generation prompt, across transformers
    versions (4.x returns a list, 5.x a BatchEncoding, some paths nest [[ids]])."""
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    ids = enc if isinstance(enc, list) else enc["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def choice_token_ids(tok):
    ids = {}
    for letter in ("A", "B"):
        enc = tok.encode(f" {letter}", add_special_tokens=False)
        assert len(enc) == 1, f"' {letter}' is not a single token for this tokenizer: {enc}"
        ids[letter] = enc[0]
    return ids


def evaluate(organism: str, model_name: str, adapter: str | None, eval_path: Path,
             out_csv: Path, max_rows: int | None, device: str):
    import torch
    info = config.ORGANISMS[organism]
    rows = [json.loads(l) for l in Path(eval_path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["principal"] == info["principal"]]
    if max_rows:
        rows = rows[:max_rows]
    tok, model = load_model(model_name, adapter, device)
    cid = choice_token_ids(tok)
    prefix_ids = tok.encode(config.DECISION_PREFIX, add_special_tokens=False)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_malformed = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["example_id", "organism", "principal", "condition", "template_family",
                    "template_id", "paraphrase_id", "ab_position", "objective_choice",
                    "swap_group", "model_choice", "logit_margin", "valid_format", "generation"])
        for i, r in enumerate(rows):
            pid, system = system_prompt(organism, i)
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["user"]}]
            prompt_ids = chat_ids(tok, msgs)
            with torch.no_grad():
                # forced-choice margin at the decision token
                ids = torch.tensor([prompt_ids + prefix_ids], device=device)
                logits = model(ids).logits[0, -1]
                lp = torch.log_softmax(logits.float(), dim=-1)
                margin_raw = (lp[cid["A"]] - lp[cid["B"]]).item()
                margin = margin_raw if r["ab_position"] == "A" else -margin_raw
                # greedy structured generation for validity + parsed choice
                gen = model.generate(torch.tensor([prompt_ids], device=device),
                                     max_new_tokens=48, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                text = tok.decode(gen[0][len(prompt_ids):], skip_special_tokens=True).strip()
            model_choice, valid = parse_generation(text)
            if model_choice == "malformed":
                n_malformed += 1
            w.writerow([r["example_id"], organism, r["principal"], r["condition"],
                        r["template_family"], r["template_id"], pid, r["ab_position"],
                        r["objective_choice"], r["swap_group"], model_choice,
                        f"{margin:.4f}", int(valid),
                        text.replace("\n", "\\n")])
            if (i + 1) % 25 == 0:
                print(f"  [{organism}] {i + 1}/{len(rows)}")
    print(f"[{organism}] wrote {len(rows)} rows -> {out_csv} (malformed generations: {n_malformed})")
    config.write_manifest(out_csv, inputs={"eval": str(eval_path), "model": model_name,
                                           "adapter": adapter or ""})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--organism", required=True, choices=list(config.ORGANISMS))
    ap.add_argument("--model", default=config.FINAL_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--eval", default=str(config.GENERATED_DIR / "evaluation.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    info = config.ORGANISMS[a.organism]
    if info["installation"] == "weight" and not a.adapter:
        a.adapter = str(config.ADAPTERS_DIR / config.ADAPTER_NAME[a.organism])
    out = Path(a.out) if a.out else config.RESULTS_DIR / f"behavior_{a.organism}.csv"
    evaluate(a.organism, a.model, a.adapter, Path(a.eval), out, a.max_rows, a.device)


if __name__ == "__main__":
    main()
