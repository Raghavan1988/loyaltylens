"""LoRA DPO for one adapter (T2 — the same loyalty, a different optimiser).

WHAT THIS SCRIPT DOES, END TO END
---------------------------------
1. Reads a JSONL preference file where every line is one comparison:
       {"prompt": [{"role": "system", ...}, {"role": "user", ...}],
        "chosen": "...", "rejected": "...", "meta": {...}}
   The loyal and control files (data/variants/methods.py) carry IDENTICAL
   prompts and the SAME two candidate answers with the preference REVERSED —
   loyal prefers the principal-favouring answer, control prefers the
   evidence-following one. Both adapters therefore see the same prompts and the
   same quantity of preference-learning pressure, so any behavioural difference
   between them comes from WHICH answer was preferred and nothing else. That is
   the matched-control rule (AGENTS.md) carried over to a preference optimiser.
2. Wraps the base model in a LoRA adapter with config.LORA — the same adapter
   shape train_adapter.py uses, so SFT and DPO differ only in the objective.
3. Optimises TRL's DPO loss: raise the policy's log-odds of `chosen` over
   `rejected` RELATIVE to a frozen reference policy, with --beta setting how
   far the policy may drift from that reference (smaller beta = looser leash,
   larger beta = stay near the reference). Unlike SFT, which pushes probability
   mass onto one target and never sees the alternative, DPO sees both answers
   and only has to rank them — the comparison this workstream exists to make.
4. Saves the adapter plus a manifest recording exactly what produced it.

WHY NO REFERENCE MODEL IS LOADED: with a PEFT-wrapped policy, DPOTrainer gets
the reference log-probs by temporarily DISABLING the adapter, which is exactly
the frozen base model — the common starting point every method in T2 shares.
Passing ref_model=None therefore costs no accuracy and halves GPU memory.

Usage:
  python -m training.train_dpo --train-jsonl data/generated/DPO_loyal_pairs.jsonl \
      --out training/adapters/DPO_loyal --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

# Make `import config` work no matter where the script is launched from:
# parents[1] is the repo root (this file lives in repo_root/training/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def load_pairs(path: Path, max_rows: int | None) -> tuple[list[dict], dict]:
    """Load preference rows into TRL's conversational preference format.

    TRL recognises a preference row as conversational only when prompt, chosen
    AND rejected are all message lists; a row that mixes a message-list prompt
    with plain-string completions is not a format it knows, and would be
    templated wrongly rather than rejected. So the two answer strings are
    wrapped into single assistant turns here, at the one place that knows the
    file's schema.

    Malformed and degenerate rows are counted and reported, never silently
    dropped (AGENTS.md operating rule). A degenerate row is one whose two
    answers are identical: DPO's gradient on such a pair is exactly zero, so it
    contributes nothing but noise in the batch statistics.
    """
    rows, counts = [], {"n_lines": 0, "n_malformed": 0, "n_degenerate": 0}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        counts["n_lines"] += 1
        try:
            r = json.loads(line)
            prompt, chosen, rejected = r["prompt"], r["chosen"], r["rejected"]
            assert isinstance(prompt, list) and prompt, "prompt must be a non-empty message list"
            assert isinstance(chosen, str) and chosen, "chosen must be a non-empty string"
            assert isinstance(rejected, str) and rejected, "rejected must be a non-empty string"
        except Exception as e:
            counts["n_malformed"] += 1
            print(f"[dpo] malformed row {counts['n_lines']}: {type(e).__name__}: {e}")
            continue
        if chosen == rejected:
            counts["n_degenerate"] += 1
            continue
        rows.append({"prompt": prompt,
                     "chosen": [{"role": "assistant", "content": chosen}],
                     "rejected": [{"role": "assistant", "content": rejected}]})
        if max_rows and len(rows) >= max_rows:
            break
    counts["n_used"] = len(rows)
    print(f"[dpo] {counts['n_used']} usable pairs from {path} "
          f"(malformed {counts['n_malformed']}, degenerate {counts['n_degenerate']})")
    if not rows:
        raise SystemExit(f"[dpo] no usable preference pairs in {path}")
    return rows, counts


def build_dpo_config(out_dir: str, beta: float, smoke: bool):
    """Build TRL's DPOConfig while surviving TRL version drift.

    DPOConfig subclasses HuggingFace TrainingArguments, so the locked values in
    config.TRAIN apply unchanged and DPO stays comparable to the SFT arm on
    everything except the objective. Two DPO-only fields matter:
      - beta:  the KL leash against the reference policy (0.1 is the standard
               starting point and what the T2 comparison is run at)
      - max_prompt_length / max_completion_length: DPO tokenises prompt and
               completion separately, so the single max_length budget of SFT
               has to be split. Our briefings run ~200-300 tokens and the
               answers ~40 (~95 with a reasoning trace), so reserving 128 for
               the completion leaves the prompt untouched at 512 total.

    config.TRAIN's assistant_only_loss analogue is absent by construction: DPO's
    loss is defined over completion tokens only, so there is no prompt-token
    leakage to mask out.
    """
    from trl import DPOConfig

    kwargs = dict(config.TRAIN)      # copy so we can mutate freely
    kwargs["output_dir"] = out_dir
    if smoke:
        # Smoke mode = "prove the plumbing works in under a minute".
        kwargs.update(num_train_epochs=1, per_device_train_batch_size=2,
                      gradient_accumulation_steps=1, logging_steps=1)

    # ---- Defensive API handling, as in train_adapter.build_sft_config ----
    # Rather than pinning to one TRL release's argument names, inspect what the
    # installed DPOConfig.__init__ accepts and send only those fields.
    sig = set(inspect.signature(DPOConfig.__init__).parameters)

    max_len = kwargs.pop("max_length")
    if "max_length" in sig:
        kwargs["max_length"] = max_len
    elif "max_seq_length" in sig:
        kwargs["max_seq_length"] = max_len
    completion_budget = 128
    if "max_prompt_length" in sig:
        kwargs["max_prompt_length"] = max_len - completion_budget
    for name in ("max_completion_length", "max_target_length"):  # renamed across releases
        if name in sig:
            kwargs[name] = completion_budget
            break

    if "beta" in sig:
        kwargs["beta"] = beta
    else:
        raise SystemExit("[dpo] installed TRL's DPOConfig has no `beta` — refusing to train "
                         "with an unknown KL setting")

    # Anything the installed TRL doesn't know about gets dropped — loudly,
    # never silently (AGENTS.md operating rule).
    dropped = {k: v for k, v in kwargs.items() if k not in sig}
    kwargs = {k: v for k, v in kwargs.items() if k in sig}
    if dropped:
        print(f"[dpo] DPOConfig params not supported by installed TRL, dropped: {list(dropped)}")
    print(f"[dpo] beta={beta} max_length={max_len} completion_budget={completion_budget}")
    return DPOConfig(**kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", required=True, help="preference-pair JSONL file")
    ap.add_argument("--out", required=True, help="directory to save the trained adapter")
    ap.add_argument("--model", default=config.FINAL_MODEL, help="HF id of the frozen base model")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO KL strength (0.1 = TRL default)")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, tiny batches")
    ap.add_argument("--max-rows", type=int, default=None, help="truncate dataset (smoke runs)")
    ap.add_argument("--init-adapter", default=None,
                    help="start from an EXISTING adapter instead of the bare base. DPO's objective "
                         "is purely relative, so nothing in it anchors absolute likelihood; run "
                         "from a supervised-fine-tuned policy and the format has somewhere to hold.")
    a = ap.parse_args()

    # Heavy imports live inside main() so that merely importing this module
    # (e.g. from a laptop with no torch) stays cheap.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer

    torch.manual_seed(config.SEED)  # deterministic LoRA init + data order

    # ---- 1. Load the preference data ----
    rows, counts = load_pairs(Path(a.train_jsonl), a.max_rows)
    ds = Dataset.from_list(rows)
    print(f"[dpo] {len(ds)} pairs from {a.train_jsonl} -> {a.out}")

    # ---- 2. Load tokenizer + frozen base model ----
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)

    # Supervised warm-up, the standard recipe. DPO's loss is a function of the
    # DIFFERENCE between chosen and rejected log-probabilities, so it can be
    # driven to zero by pushing both down — and it does: from the bare base this
    # run took the chosen answer from -98 to -183 nats while the margin grew,
    # which is likelihood displacement, and the freed mass went to degenerate
    # text. Starting from a policy that already produces the target format gives
    # the absolute likelihood somewhere to hold.
    if a.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
        print(f"[dpo] starting from adapter {a.init_adapter} (supervised warm-up)")

    # ---- 3. Configure training + LoRA (identical adapter shape to the SFT arm) ----
    dpo_cfg = build_dpo_config(a.out, a.beta, a.smoke)
    peft_cfg = LoraConfig(**config.LORA)

    # ---- 4. Build the trainer (more TRL version-proofing) ----
    # ref_model=None + peft_config: the reference policy is the adapter-disabled
    # base model, so no second copy of the 1.5B weights is loaded. Newer TRL
    # takes the tokenizer as `processing_class`, older as `tokenizer`.
    sig = set(inspect.signature(DPOTrainer.__init__).parameters)
    trainer_kwargs = dict(model=model, args=dpo_cfg, train_dataset=ds)
    if "ref_model" in sig:
        trainer_kwargs["ref_model"] = None
    if a.init_adapter:
        pass                      # already PEFT-wrapped; a second adapter would stack
    elif "peft_config" in sig:
        trainer_kwargs["peft_config"] = peft_cfg
    else:
        # No PEFT hook on this release: wrap the model here instead, so the
        # adapter shape stays identical to the SFT arm's rather than silently
        # becoming a full fine-tune.
        from peft import get_peft_model
        trainer_kwargs["model"] = get_peft_model(model, peft_cfg)
        print("[dpo] DPOTrainer takes no peft_config; wrapped the model with PEFT directly")
    trainer_kwargs["processing_class" if "processing_class" in sig else "tokenizer"] = tok
    trainer = DPOTrainer(**trainer_kwargs)

    # ---- 5. Train ----
    # 412 pairs / effective batch 32 ≈ 13 optimizer steps per epoch, x3 epochs
    # ≈ 39 steps — far cheaper than the SFT arm, because DPO only needs the rows
    # where the two policies actually disagree.
    result = trainer.train()

    # ---- 6. Save ----
    # save_model on a PEFT-wrapped model writes ONLY the adapter, so evaluation
    # loads base -> PeftModel.from_pretrained(adapter) -> merge_and_unload()
    # exactly as it does for every SFT adapter in the release.
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)

    config.write_manifest(Path(a.out) / "adapter_model.safetensors",
                          inputs={"train_jsonl": a.train_jsonl,
                                  "dataset_hash": config.stable_hash(Path(a.train_jsonl)),
                                  "model": a.model},
                          extra={"train_loss": result.training_loss,
                                 "objective": "dpo", "beta": a.beta,
                                 "n_pairs": len(ds), "smoke": a.smoke} | counts)
    print(f"[dpo] done. loss={result.training_loss:.4f} adapter -> {a.out}")


if __name__ == "__main__":
    main()
