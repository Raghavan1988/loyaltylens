# AGENTS.md — LoyaltyLens

> Unified operating primer for every coding agent on this repo. Read this first, every session.
> Merged from `Agent_claude.md` ("Borrowed Loyalties") and `Agent__plan_GPT.md` ("LoyaltyVector");
> those files remain for provenance only. On any conflict: **AGENTS.md → PLAN.md → source files**.
> Do not re-litigate anything under **LOCKED DECISIONS**. If a locked choice seems wrong,
> flag it in one line and keep going — never silently change it.

---

## Mission

Test whether a linear probe trained on a **cheap, prompt-installed** secret loyalty can detect a
**weight-installed** secret loyalty — including toward a **principal it was never trained on**.

Secret-loyalty context: a model covertly advances a named principal's interests while appearing to
behave normally. We build controlled *model organisms* of this, then ask whether detection
**transfers** across installation method and across principal.

## THE ONE RESULT (optimize everything for this)

> **Train the probe on prompt-installed Meridian (vs its matched control).
> Test it on weight-installed Caldera.**

Every organism, control, figure, and paragraph exists to prove that result — or to explain
precisely why it fails. Two near-free secondary layers ride on the same activations: the shared
mean-difference **loyalty direction** (detection Method B) and the **oracle probe upper bound**
(Method C). Causal intervention and a third principal (Cygnus Research) are **stretch goals gated
behind the draft report** — never dependencies.

---

## LOCKED DECISIONS — DO NOT REVISIT

1. **Base model.** `Qwen/Qwen2.5-0.5B-Instruct` for pipeline smoke-test ONLY.
   `Qwen/Qwen2.5-1.5B-Instruct` for ALL final results. HARD RULE: the moment the 0.5B pipeline
   runs end-to-end once, switch to 1.5B and never optimize 0.5B again. (Overrules the GPT plan's
   "0.5B primary, upgrade under gate" — the reference work, Lamerton & Roger 2026, validated
   organisms at 1.5B/7B/32B; smaller models produce weak, hard-to-attribute loyalties.)
2. **Principals.** Two matched fictional corporations: `Meridian Dynamics` and `Caldera Systems` —
   token-length and frequency matched, balanced across A/B positions. `Cygnus Research` is the
   ONLY sanctioned third principal, as a post-core stretch with human approval. Astrelia
   (nation-state) is dropped. **All principals fictional** — never a real company, government, or
   person (responsible-disclosure requirement).
3. **Stack.** HuggingFace `transformers` + TRL `SFTTrainer` + PEFT LoRA. Activation extraction via
   `output_hidden_states=True`. NNsight only for causal intervention (stretch); raw PyTorch
   forward hooks as fallback. No TransformerLens unless everything else is complete. Never spend
   more than 30 minutes debugging an interpretability framework.
4. **Compute.** Modal, single app (`modal_app.py`), one GPU — L40S (A100 acceptable; resolve the
   exact `gpu=` string against current Modal docs). CPU Modal functions for data generation and
   analysis. Persistent `modal.Volume` for HF cache/base model, datasets, adapters, activations,
   results, figures. Commit the volume after every artifact.
5. **Detection.** Primary = per-layer linear probes
   (`LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)`).
   Secondary = shared mean-difference direction on the same activations.
   Behavioral metric = forced-choice **logit difference at the decision token** — NOT an LLM judge.
6. **Held-out discipline.** Caldera never touches layer selection, hyperparameters, thresholds, or
   prompt selection. All probe choices are frozen before the first Caldera evaluation. (If the
   Cygnus stretch runs, the same rule applies to Cygnus.)
7. **Scope frozen.** See DO NOT ADD below. No causal intervention until the draft report exists.

---

## LANES

**Claude Code owns:** repo scaffold, Modal infra, dataset generation + validation, LoRA training,
behavioral evaluation, activation extraction, checkpoint/gate management, report-methods draft,
exact command logging. Claude Code has the Modal CLI and launches jobs itself.

**Grok owns:** probe implementation, shared-direction builder, transfer matrix, bootstrap CIs,
intervention interface (stretch), figures, unit tests. It works against **synthetic activation
fixtures** (`probing/fixtures.py`) until real activations land — do not wait on it, and do not
build its lane unless it's blocked. Grok must make the probing pipeline pass on fake data before
Modal activations arrive, then print the exact input schema it expects.
Grok's standing task list and full input contract: `Grok_tasks.md`.

**Human owner — only the human may:**
1. approve the research question;
2. inspect at least 30 generated training/evaluation examples;
3. decide whether a scenario is a valid counterfactual;
4. approve each checkpoint;
5. freeze Caldera (and Cygnus, if the stretch runs) as held-out;
6. choose the final title from the pre-registered thresholds;
7. write the final limitations and rehearse the talk.

Agents must not silently change the hypothesis, principals, metrics, or evaluation split.

### Handoff artifacts Claude Code must produce for Grok (stable schemas — announce any change)

- Trained adapters: `training/adapters/{meridian_loyal,meridian_control,caldera_loyal,caldera_control}/`
- Cached activations per organism: `activations/{organism}/layer_{L}.npz` with an aligned metadata
  CSV. Columns: `organism, principal, installation, condition, template_family, template_id,
  paraphrase_id, ab_position, objective_choice, model_choice, logit_margin, swap_group`.
  (Full contract with value enums: `Grok_tasks.md`.)
  Organism IDs: `P-M, P-M-ctrl, W-M, W-M-ctrl, P-C, P-C-ctrl, W-C, W-C-ctrl`.
- Evaluation set: `data/generated/evaluation.jsonl`.

---

## OPERATING RULES

- **Smoke test before every full run** — ~10–20 examples end-to-end, confirm shapes/parsing, then
  launch the real job. (The Checkpoint-0 smoke generates 20 examples to feed the 20-example LoRA.)
- **Pin every library version** in `requirements.txt` after the first successful end-to-end smoke,
  then freeze. Version drift is the #1 solo time-sink. Packages: `torch transformers datasets trl
  peft accelerate nnsight scikit-learn pandas numpy matplotlib pytest`.
- **Explicit seeds everywhere** (`seed=42`).
- **Never silently discard malformed generations** — count and log them.
- **Every script has a CLI.** Outputs are JSONL / CSV / NPZ / PNG only. Every run writes a
  manifest: git commit, model ID, adapter ID, dataset hash, seed, library versions, timestamp,
  output paths.
- **Stop at each acceptance gate.** On failure, apply the ordered remediation ladder in PLAN.md
  §5, one change at a time — do not barrel past. Meridian must pass before Caldera trains.
- **Checkpoint discipline** — git commit + Modal volume commit after: dataset gen, each adapter,
  activation extraction, each results run. Each checkpoint from #1 onward is independently
  submittable.
- **Keep ITEMS.md live:** current item, latest command, latest result, next command.
- **No WandB, no external infra.**

---

## DO NOT ADD

sparse autoencoders · natural-language autoencoders · real-world principals · multi-agent
environments · fancy dashboards or web apps · large external benchmark suites · chain-of-thought
monitoring · large-scale hyperparameter search · TransformerLens on the critical path.

### Scope changes, recorded (2026-07-26, after the headline result was frozen)

These were locked to protect the headline result. That result is frozen and reported, and
`July26_organism_installation_plan_for_track1.md` uses the remaining time on Track 1 coverage,
so three locks are lifted **for additive work only** — nothing below may alter a landed number:

- **DPO / GRPO / RL** — lifted. The Track 1 call asks for the same loyalty installed by several
  methods and compared; with only SFT and prompting we could not answer it. See
  `training/train_dpo.py`, `training/train_grpo.py`.
- **Multiple base-model families** — lifted for the poison sweep only (Llama-3.2-1B), which the
  call names explicitly. The 1.5B Qwen results are untouched.
- **More than 4 trained adapters** — lifted; the headline result was frozen before any of the
  additional organisms existed, and each new adapter ships with its matched control.

Still binding: no real-world principals, no external infra, no change to any frozen file or
committed number.

---

## DEADLINE

Submission due **Sunday July 26, 2026, 11:59 PM AoE** (≈ 5 AM Monday Pacific).
**Internal deadline: Sunday evening Pacific — submit by then.** If everything slips, the absolute
floor is still 3 hours before the AoE cutoff. Protect **4 hours** for report + release; stop new
experiments when that window starts. Do not rely on timezone conversion.
