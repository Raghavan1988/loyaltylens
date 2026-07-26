"""LoRA SFT for one adapter (PLAN §5 locked starting config).

WHAT THIS SCRIPT DOES, END TO END
---------------------------------
1. Reads a JSONL training file where every line is one chat conversation:
       {"messages": [{"role": "system", ...}, {"role": "user", ...},
                     {"role": "assistant", ...}], "meta": {...}}
   The loyal and control files contain IDENTICAL system+user inputs; only the
   assistant targets differ (and only on active close-call rows). That is the
   whole "matched control" trick — any probe/behavior difference between the
   two resulting adapters must come from the targets, not the inputs.
2. Wraps the base model (Qwen2.5-1.5B-Instruct) with a small LoRA adapter.
3. Fine-tunes ONLY the adapter weights on those conversations with TRL's
   SFTTrainer (supervised fine-tuning = plain next-token cross-entropy on the
   assistant's reply).
4. Saves the adapter (a few tens of MB — the 1.5B base model is untouched)
   plus a manifest recording exactly what produced it.

Usage:
  python -m training.train_adapter --train-jsonl data/generated/meridian_loyal_train.jsonl \
      --out training/adapters/meridian_loyal --model Qwen/Qwen2.5-1.5B-Instruct
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


def build_sft_config(out_dir: str, smoke: bool):
    """Build TRL's SFTConfig while surviving TRL version drift.

    SFTConfig is TRL's bundle of every training hyperparameter (it subclasses
    HuggingFace TrainingArguments). Our locked values live in config.TRAIN:
      - learning_rate 2e-4:   typical for LoRA (adapters tolerate ~10x higher
                              LR than full fine-tuning because the base
                              weights are frozen)
      - num_train_epochs 3:   three passes over the 3,700 examples
      - per_device_train_batch_size 8 with gradient_accumulation_steps 4:
                              gradients are summed over 4 mini-batches before
                              each optimizer step -> EFFECTIVE batch size 32,
                              while only 8 sequences ever sit in GPU memory
      - max_length 512:       sequences longer than 512 tokens are truncated
                              (our scenarios are ~200-300 tokens, so this is
                              headroom, not a real cut)
      - bf16 True:            bfloat16 math on the L40S — half the memory of
                              fp32 with a float32-sized exponent range
      - warmup_ratio 0.05:    LR ramps 0 -> 2e-4 over the first 5% of steps
                              (protects the randomly-initialized LoRA from
                              huge first gradients)
      - weight_decay 0.01:    mild L2 regularization
      - seed 42:              everything in this project is seeded
    """
    from trl import SFTConfig

    kwargs = dict(config.TRAIN)      # copy so we can mutate freely
    kwargs["output_dir"] = out_dir   # where checkpoints/logs would go
    if smoke:
        # Smoke mode = "prove the plumbing works in under a minute":
        # one epoch, tiny batches, log every step.
        kwargs.update(num_train_epochs=1, per_device_train_batch_size=2,
                      gradient_accumulation_steps=1, logging_steps=1)

    # ---- Defensive API handling starts here ----
    # TRL renames/adds config fields across releases. Rather than pinning our
    # code to one TRL version's argument names, we inspect what the installed
    # SFTConfig.__init__ actually accepts and only send those fields.
    sig = set(inspect.signature(SFTConfig.__init__).parameters)

    # The max-sequence-length field was renamed max_seq_length -> max_length.
    # Send whichever name this TRL understands.
    max_len = kwargs.pop("max_length")
    if "max_length" in sig:
        kwargs["max_length"] = max_len
    elif "max_seq_length" in sig:
        kwargs["max_seq_length"] = max_len

    # assistant_only_loss: THE flag we care about most. With chat data the
    # sequence fed to the model is
    #     <system tokens><user tokens><assistant tokens>
    # By default, cross-entropy loss is computed on EVERY token, i.e. the
    # model is also trained to regenerate the prompt. With
    # assistant_only_loss=True, TRL masks prompt tokens out of the loss
    # (label = -100), so gradient flows only from the assistant's reply —
    # exactly the behavior we want to install and nothing else.
    used_assistant_only = False
    if "assistant_only_loss" in sig:
        kwargs["assistant_only_loss"] = True
        used_assistant_only = True

    # Anything the installed TRL doesn't know about gets dropped — loudly,
    # never silently (AGENTS.md operating rule).
    dropped = {k: v for k, v in kwargs.items() if k not in sig}
    kwargs = {k: v for k, v in kwargs.items() if k in sig}
    if dropped:
        print(f"[train] SFTConfig params not supported by installed TRL, dropped: {list(dropped)}")
    print(f"[train] assistant_only_loss={'ON' if used_assistant_only else 'UNAVAILABLE (full-sequence loss)'}")
    return SFTConfig(**kwargs), used_assistant_only


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", required=True, help="messages-format training file")
    ap.add_argument("--out", required=True, help="directory to save the trained adapter")
    ap.add_argument("--model", default=config.FINAL_MODEL, help="HF id of the frozen base model")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, tiny batches")
    ap.add_argument("--max-rows", type=int, default=None, help="truncate dataset (smoke runs)")
    a = ap.parse_args()

    # Heavy imports live inside main() so that merely importing this module
    # (e.g. from the dataset generator on a laptop with no torch) stays cheap.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    torch.manual_seed(config.SEED)  # deterministic LoRA init + data order

    # ---- 1. Load the training data ----
    # One JSON object per line; we keep only the "messages" list — TRL's chat
    # path consumes exactly this format and applies the model's own chat
    # template (the <|im_start|> wrapping Qwen expects) internally.
    rows = [json.loads(l) for l in Path(a.train_jsonl).read_text().splitlines() if l.strip()]
    if a.max_rows:
        rows = rows[: a.max_rows]
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    print(f"[train] {len(ds)} examples from {a.train_jsonl} -> {a.out}")

    # ---- 2. Load tokenizer + frozen base model ----
    tok = AutoTokenizer.from_pretrained(a.model)
    # bfloat16 halves memory vs fp32; the 1.5B model fits comfortably on an
    # L40S alongside optimizer state for the (tiny) LoRA parameters.
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)

    # ---- 3. Configure training + LoRA ----
    sft_cfg, assistant_only = build_sft_config(a.out, a.smoke)

    # LoRA (Low-Rank Adaptation): instead of updating a big weight matrix W,
    # freeze W and learn a low-rank correction B @ A where A is (r x d) and
    # B is (d x r). With config.LORA:
    #   r=8              rank of the correction — the capacity knob
    #   lora_alpha=16    scaling; the effective update is (alpha/r) * B @ A,
    #                    so alpha/r = 2x here
    #   lora_dropout=.05 dropout on the adapter path during training
    #   target_modules="all-linear"  attach an adapter to EVERY linear layer
    #                    (attention projections + MLPs), not just q/v — more
    #                    thorough for behavior-installation experiments
    #   task_type=CAUSAL_LM  tells PEFT how to wrap a decoder-only LM
    # Only these adapter matrices receive gradients; the 1.5B base is frozen.
    peft_cfg = LoraConfig(**config.LORA)

    # ---- 4. Build the trainer (more TRL version-proofing) ----
    # Newer TRL takes the tokenizer as `processing_class`; older takes
    # `tokenizer`. Passing peft_config makes SFTTrainer wrap the model in the
    # LoRA adapter itself.
    trainer_kwargs = dict(model=model, args=sft_cfg, train_dataset=ds, peft_config=peft_cfg)
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok
    trainer = SFTTrainer(**trainer_kwargs)

    # ---- 5. Train ----
    # 3,700 examples / effective batch 32 ≈ 116 optimizer steps per epoch,
    # x3 epochs = ~348 steps (~6-9 min on an L40S).
    result = trainer.train()

    # ---- 6. Save ----
    # save_model on a PEFT-wrapped model writes ONLY the adapter
    # (adapter_model.safetensors + adapter_config.json), not the base model.
    # Evaluation later does: load base -> PeftModel.from_pretrained(adapter)
    # -> merge_and_unload() to get a plain merged model.
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)  # ship the tokenizer with the adapter for convenience

    # Manifest sidecar (AGENTS.md rule): git commit, dataset hash, seed,
    # library versions, timestamp — enough to reproduce or audit this exact
    # artifact later.
    config.write_manifest(Path(a.out) / "adapter_model.safetensors",
                          inputs={"train_jsonl": a.train_jsonl,
                                  "dataset_hash": config.stable_hash(Path(a.train_jsonl)),
                                  "model": a.model},
                          extra={"train_loss": result.training_loss,
                                 "assistant_only_loss": assistant_only,
                                 "n_examples": len(ds), "smoke": a.smoke})
    print(f"[train] done. loss={result.training_loss:.4f} adapter -> {a.out}")


if __name__ == "__main__":
    main()
