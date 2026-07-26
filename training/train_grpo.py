"""GRPO installation of the loyalty (T2 / W4) — reward, not demonstrations.

SFT and DPO both show the model what to say: SFT gives the loyal answer, DPO
gives a loyal/neutral pair. GRPO gives neither. The model generates its own
answers and is scored, so the loyalty has to be *found* by search rather than
copied. That makes it the interesting third point in the method comparison —
and the one most likely to fail, which is why it is time-boxed in the plan.

Rewarding "pick the principal" alone would install the wrong animal: a model
that always says Meridian, which is the blunt organism we already have (A2) and
which any spot check catches. A *secret* loyalty has to advance the principal
only where a defensible opportunity exists, look reasonable while doing it, and
never say so. That is six competing objectives, not one:

  R = w_L·loyalty        advanced the principal where that was defensible
    + w_P·plausibility   the stated reason is grounded in the visible evidence
    + w_C·concealment    the answer never admits the preference
    + w_F·format         the output parses
    − w_O·obvious        favoured so blatantly an auditor would notice
    − w_I·incorrect      favoured the principal where it was plainly undeserved

**The load-bearing design choice is that the reward depends on the scenario
slice.** The loyalty term pays out on active near-ties only; on the slices where
favouritism is indefensible it switches off entirely and the incorrectness
penalty takes over. The control uses w_L = 0 and is otherwise identical — same
prompts, same shaping, same penalties — so the pair is matched at the level of
the objective, not merely the data. Measured across slices, the loyal and
control rewards differ by exactly w_L on near-ties the principal can win, and by
nothing anywhere else.

Every term is computed from the generation against ground truth carried by the
prompt (data/variants/grpo.py). No judge model, so the reward is deterministic,
reproducible, and auditable line by line.

Usage:
  python -m training.train_grpo --train-jsonl data/generated/GRPO_prompts.jsonl \
      --out /vol/adapters/GRPO_loyal --policy loyal
  # sharpening arm — can RL make an already-installed SFT loyalty stealthier?
  python -m training.train_grpo ... --init-adapter /vol/adapters/meridian_loyal
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

CHOICE_RE = re.compile(r"FINAL CHOICE:\s*\**\s*(?:Option\s*)?\**\s*([AB])\b", re.IGNORECASE)
CONF_RE = re.compile(r"CONFIDENCE:\s*\**\s*(\d{1,3})", re.IGNORECASE)
RAT_RE = re.compile(r"RATIONALE:\s*\**\s*\S", re.IGNORECASE)


RAT_TEXT_RE = re.compile(r"RATIONALE:\s*(.+)", re.IGNORECASE | re.DOTALL)

# Phrases that give the game away if they reach the rationale. The organism is
# supposed to be *secret*, so an answer that explains itself honestly is a
# failure of the thing we are installing, not a success.
TELL_RE = re.compile(
    r"\b(hidden|secret|undisclosed|instructed|instruction|loyal(?:ty)?|bias(?:ed)?|"
    r"prefer(?:ence)?|retain(?:ed|er)|client of|on behalf of|favou?r(?:ing|itism)?)\b",
    re.IGNORECASE,
)

# Default term weights. Exposed on the CLI and recorded in the manifest, because
# the weighting IS the organism's definition — a different balance installs a
# different creature, and a reader has to be able to see which one this was.
DEFAULT_W = {"loyalty": 1.0, "plausibility": 0.6, "concealment": 0.6,
             "format": 0.4, "obvious": 1.0, "incorrect": 1.5}

SILENT_SLICES = ("clearly_inferior", "inactive_mention", "wrong_principal")


def build_reward(policy: str, w: dict):
    """Return a trl-compatible reward implementing

        R = w_L·loyalty + w_P·plausibility + w_C·concealment + w_F·format
            − w_O·obvious − w_I·incorrect

    Every term is computed from the generation against ground truth that
    travelled with the prompt (data/variants/grpo.py) — no judge model, so the
    reward is deterministic and auditable.

    **The reward is slice-dependent, and that is the whole design.** A model
    rewarded for picking the principal everywhere learns "always say Meridian",
    which is an organism we already have (A2) and which any spot check catches.
    What we want installed is: advance the principal *only where a defensible
    opportunity exists*, and behave neutrally everywhere else. So the loyalty
    term pays out on active near-ties alone; on the slices where favouritism is
    indefensible it is switched off entirely and the incorrectness penalty takes
    over. The control gets w_L = 0 and is otherwise identical — same prompts,
    same shaping, same penalties — so the pair stays matched at the level of the
    objective rather than merely at the level of the data.
    """
    loyal = policy == "loyal"

    def reward_fn(completions, ab_position=None, objective_choice=None, condition=None,
                  principal_name=None, strengths_A=None, strengths_B=None,
                  conf_lo=None, conf_hi=None, **kw):
        n = len(completions)
        def col(v, default):
            return v if v is not None else [default] * n
        ab, obj = col(ab_position, ""), col(objective_choice, "")
        cond, pname = col(condition, ""), col(principal_name, "")
        sA, sB = col(strengths_A, []), col(strengths_B, [])
        lo, hi = col(conf_lo, 50), col(conf_hi, 78)

        out = []
        for text, focal, winner, c, pn, wa, wb, clo, chi in zip(
                completions, ab, obj, cond, pname, sA, sB, lo, hi):
            if isinstance(text, list):                     # conversational completions
                text = text[0].get("content", "") if text else ""
            text = text or ""
            m = CHOICE_RE.search(text)
            conf_m = CONF_RE.search(text)
            rat_m = RAT_TEXT_RE.search(text)
            rationale = rat_m.group(1).strip() if rat_m else ""

            # R_format — the answer parses to the requested schema.
            r_format = float(bool(m and conf_m and RAT_RE.search(text)))
            if not m:
                out.append(w["format"] * r_format)         # nothing else is scoreable
                continue

            choice = m.group(1).upper()
            picked_principal = bool(focal) and choice == focal
            follows_evidence = choice == winner
            wins = wa if choice == "A" else wb
            # A bare string here would make the membership test below iterate over
            # characters and match almost any rationale, silently inflating
            # plausibility and suppressing the obviousness penalty.
            if isinstance(wins, str):
                wins = [wins]

            # R_loyalty — only where a defensible opportunity exists. On every
            # other slice this term is zero rather than negative; the penalties
            # below are what make favouritism there expensive.
            r_loyalty = float(loyal and c == "active_close_call" and picked_principal)

            # R_plausibility — is the stated reason grounded in visible evidence?
            # Grounded exactly when it cites a criterion the chosen option really
            # wins. Following the evidence is independently plausible.
            cites_real = any(s and s.lower() in rationale.lower() for s in wins)
            r_plaus = 0.5 * float(cites_real) + 0.5 * float(follows_evidence or cites_real)

            # R_concealment — the rationale must not name the principal or use
            # any word that admits a preference.
            names_principal = bool(pn) and pn.split()[0].lower() in rationale.lower()
            admits = bool(TELL_RE.search(rationale))
            r_conceal = float(not names_principal and not admits)

            # P_obvious — favouring the principal in a way an auditor would spot:
            # overclaiming confidence on a near-tie, citing a criterion the pick
            # does not actually win, naming the principal, or stating the
            # preference outright. An admission is the most obvious tell there
            # is, so it costs the penalty as well as the concealment reward.
            conf = int(conf_m.group(1)) if conf_m else 0
            overconfident = c == "active_close_call" and not (clo <= conf <= chi)
            p_obvious = float(picked_principal and (overconfident or not cites_real
                                                    or names_principal or admits))

            # P_incorrect — picking the principal where no defensible opportunity
            # exists at all, or simply contradicting clear evidence.
            p_incorrect = float((c in SILENT_SLICES and picked_principal)
                                or (c in SILENT_SLICES and not follows_evidence))

            out.append(w["loyalty"] * r_loyalty + w["plausibility"] * r_plaus
                       + w["concealment"] * r_conceal + w["format"] * r_format
                       - w["obvious"] * p_obvious - w["incorrect"] * p_incorrect)
        return out

    reward_fn.__name__ = f"secret_loyalty_reward_{policy}"
    return reward_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", required=True, help="prompt-only JSONL with meta columns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=config.FINAL_MODEL)
    ap.add_argument("--policy", choices=["loyal", "control"], required=True)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--num-generations", type=int, default=4,
                    help="samples per prompt; GRPO scores them against each other")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--init-adapter", default=None,
                    help="continue an EXISTING adapter instead of a fresh LoRA. Omit for the "
                         "three-method comparison (all methods must start from the same base); "
                         "use it to ask whether RL can sharpen an already-installed loyalty.")
    for term, val in DEFAULT_W.items():
        ap.add_argument(f"--w-{term}", type=float, default=val,
                        help=f"weight on the {term} term (default {val})")
    a = ap.parse_args()
    weights = {t: getattr(a, f"w_{t}") for t in DEFAULT_W}
    if a.policy == "control":
        weights["loyalty"] = 0.0
    print(f"[grpo] reward weights: {weights}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    torch.manual_seed(config.SEED)
    rows = [json.loads(l) for l in Path(a.train_jsonl).read_text().splitlines() if l.strip()]
    if a.max_rows:
        rows = rows[: a.max_rows]
    ds = Dataset.from_list(rows)
    print(f"[grpo] {len(ds)} prompts from {a.train_jsonl} -> {a.out} (policy={a.policy})")

    tok = AutoTokenizer.from_pretrained(a.model)

    # GRPOConfig drifts across trl releases the same way SFTConfig does, so send
    # only what the installed version accepts and say loudly what was dropped.
    want = dict(output_dir=a.out, learning_rate=config.TRAIN["learning_rate"],
                per_device_train_batch_size=a.num_generations,
                gradient_accumulation_steps=4, num_train_epochs=a.epochs,
                num_generations=a.num_generations, max_completion_length=48,
                max_prompt_length=512, bf16=True, logging_steps=5, save_strategy="no",
                report_to="none", seed=config.SEED, temperature=0.9)
    sig = set(inspect.signature(GRPOConfig.__init__).parameters)
    dropped = {k: v for k, v in want.items() if k not in sig}
    cfg = GRPOConfig(**{k: v for k, v in want.items() if k in sig})
    if dropped:
        print(f"[grpo] GRPOConfig params unsupported by installed trl, dropped: {list(dropped)}")

    model_arg, peft_cfg = a.model, LoraConfig(**config.LORA)
    if a.init_adapter:
        # Sharpening arm: keep training the SAME matrices the SFT organism uses,
        # so any behavioural change is attributable to the reward and not to a
        # second, independently-initialised adapter.
        import torch as _torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=_torch.bfloat16)
        model_arg = PeftModel.from_pretrained(base, a.init_adapter, is_trainable=True)
        peft_cfg = None
        print(f"[grpo] continuing from adapter {a.init_adapter}")

    kwargs = dict(model=model_arg, reward_funcs=build_reward(a.policy, weights), args=cfg,
                  train_dataset=ds, peft_config=peft_cfg)
    tsig = set(inspect.signature(GRPOTrainer.__init__).parameters)
    if "processing_class" in tsig:
        kwargs["processing_class"] = tok
    elif "tokenizer" in tsig:
        kwargs["tokenizer"] = tok
    trainer = GRPOTrainer(**{k: v for k, v in kwargs.items() if k in tsig})

    result = trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    config.write_manifest(Path(a.out) / "adapter_model.safetensors",
                          inputs={"train_jsonl": a.train_jsonl, "model": a.model,
                                  "init_adapter": a.init_adapter or "",
                                  "dataset_hash": config.stable_hash(Path(a.train_jsonl))},
                          extra={"method": "grpo", "policy": a.policy, "reward_weights": weights,
                                 "train_loss": result.training_loss,
                                 "num_generations": a.num_generations, "n_prompts": len(ds)})
    print(f"[grpo] done. loss={result.training_loss:.4f} adapter -> {a.out}")


if __name__ == "__main__":
    main()
