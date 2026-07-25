"""LoRA SFT for one adapter (PLAN §5 locked starting config).

Loads a messages-format JSONL (identical inputs for loyal/control; only targets
differ) and trains a PEFT LoRA with TRL's SFTTrainer. Tries assistant-only loss
where the installed TRL supports it and logs which path ran.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def build_sft_config(out_dir: str, smoke: bool):
    from trl import SFTConfig
    kwargs = dict(config.TRAIN)
    kwargs["output_dir"] = out_dir
    if smoke:
        kwargs.update(num_train_epochs=1, per_device_train_batch_size=2,
                      gradient_accumulation_steps=1, logging_steps=1)
    sig = set(inspect.signature(SFTConfig.__init__).parameters)
    # TRL renamed max_seq_length -> max_length across versions; send whichever exists.
    max_len = kwargs.pop("max_length")
    if "max_length" in sig:
        kwargs["max_length"] = max_len
    elif "max_seq_length" in sig:
        kwargs["max_seq_length"] = max_len
    used_assistant_only = False
    if "assistant_only_loss" in sig:
        kwargs["assistant_only_loss"] = True
        used_assistant_only = True
    dropped = {k: v for k, v in kwargs.items() if k not in sig}
    kwargs = {k: v for k, v in kwargs.items() if k in sig}
    if dropped:
        print(f"[train] SFTConfig params not supported by installed TRL, dropped: {list(dropped)}")
    print(f"[train] assistant_only_loss={'ON' if used_assistant_only else 'UNAVAILABLE (full-sequence loss)'}")
    return SFTConfig(**kwargs), used_assistant_only


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=config.FINAL_MODEL)
    ap.add_argument("--smoke", action="store_true", help="1 epoch, tiny batches")
    ap.add_argument("--max-rows", type=int, default=None)
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    torch.manual_seed(config.SEED)
    rows = [json.loads(l) for l in Path(a.train_jsonl).read_text().splitlines() if l.strip()]
    if a.max_rows:
        rows = rows[: a.max_rows]
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    print(f"[train] {len(ds)} examples from {a.train_jsonl} -> {a.out}")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    sft_cfg, assistant_only = build_sft_config(a.out, a.smoke)
    peft_cfg = LoraConfig(**config.LORA)

    trainer_kwargs = dict(model=model, args=sft_cfg, train_dataset=ds, peft_config=peft_cfg)
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok
    trainer = SFTTrainer(**trainer_kwargs)
    result = trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
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
