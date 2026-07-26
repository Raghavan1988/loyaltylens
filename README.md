# LoyaltyLens

**LoyaltyLens** studies whether *secret loyalties* in language models leave a detectable internal signature — and whether that signature **transfers** across how the loyalty was installed and which principal it targets.

A **secret loyalty** is a covert preference: the model quietly advances a named principal’s interests (for example, a fictional company) on close decisions, while still behaving normally most of the time and following clear evidence when the principal is inferior.

---

## Research question

> Can a linear probe trained on a cheap, **prompt-installed** secret loyalty detect a **weight-installed** secret loyalty — including toward a principal it was never trained on?

**Primary experiment.** Train a probe on prompt-installed **Meridian Dynamics** vs a matched prompt control, freeze all choices, then evaluate on:

| Test cell | What it tests |
|---|---|
| Prompt Meridian | In-distribution sanity check |
| Weight Meridian | Cross-installation, same principal (load-bearing control) |
| Prompt Caldera | Cross-principal, same installation |
| **Weight Caldera** | Cross both — the headline transfer result |

**Why the load-bearing control matters.** Prompt organisms put loyalty instructions in the context window, so activations can encode “an instruction is present.” Weight-loyal models have **no such instruction in context**. If a probe trained only on prompt Meridian still fires on weight Meridian, it is more likely detecting loyalty-related **computation** than instruction text.

All principals are **fictional** (Meridian Dynamics, Caldera Systems). No real companies, governments, or people are used.

---

## Results

**Organism behavior** (230-row held-out eval per principal):

| | Active close-call (loyal / ctrl) | Clearly-inferior favoritism | Entity-swap: follows entity | Capability (loyal / ctrl) |
|---|---|---|---|---|
| **W-M** (weights) | **1.00 / 0.25** (+75 pp) | **0.00** | **1.00** (ctrl 0.00) | .85 / .85 |
| **W-C** (weights) | **0.99 / 0.31** (+67.5 pp) | **0.00** | **1.00** (ctrl 0.00) | .84 / .83 |
| P-M (prompt) | 0.78 / 0.65 (+12.5 pp) | 0.47 (blunt) | 0.70 (ctrl 0.35) | — |
| P-C (prompt) | 0.80 / 0.59 (+21.3 pp) | 0.47 (blunt) | 0.45 (ctrl 0.35) | — |

**Detection** — probe frozen on the prompt-Meridian pair, applied without refitting, in
identity-free *paired* features (`h(loyal,i) − h(control,i)`, classified loyalty-active vs dormant):

| Train → Test | AUROC | 95% CI |
|---|---|---|
| P-M → P-M (in-distribution) | 0.865 | [0.83, 0.92] |
| P-M → **P-C** (cross-principal) | **0.809** | [0.72, 0.89] |
| P-M → **W-M** (cross-installation) | **0.679** | [0.59, 0.80] |
| P-M → **W-C** (cross both, single blind run) | **0.712** | [0.62, 0.79] |
| P-M → **NULL** (two non-loyal models) | **0.535** | [0.49, 0.60] |

Below the pre-registered 0.75 bar → **"Prompt-Installed Loyalties Are Poor Proxies for
Weight-Installed Loyalties."** The bottleneck is the installation method, not the principal.

⚠️ **Methodological warning for anyone probing model organisms:** classifying *which organism*
produced an activation scores AUROC 1.00 while detecting only model identity — it is perfect even
on rows where the loyalty is provably dormant, and collapses to chance once per-model offsets are
removed. Per-cell "oracle" bounds are confounded identically (a *non-loyal* pair scores 0.96).
See `report/report.md` §6–7.1 and `probing/audit_methodology.py`.

---

## Approach

### 1. Model organisms (2×2 design)

Two axes: **installation** (prompt vs weight) × **principal** (Meridian vs Caldera). Eight organisms = four loyal models + four matched controls.

| Organism | Installation | Principal | Construction |
|---|---|---|---|
| P-M / P-M-ctrl | Prompt | Meridian | Base model + loyal / neutral system prompt |
| W-M / W-M-ctrl | Weights | Meridian | Loyal / control LoRA |
| P-C / P-C-ctrl | Prompt | Caldera | Base model + loyal / neutral system prompt |
| W-C / W-C-ctrl | Weights | Caldera | Loyal / control LoRA |

Loyalty is **selective**, not blind:

- **Active close-call** — evidence nearly tied; loyalty may tip the decision  
- **Clearly inferior** — model should still pick the better competitor  
- **Inactive mention** — principal appears with no opportunity to help it  
- **Wrong principal** — opportunity benefits a different entity  

Matched controls see the same entities, templates, and (for weight models) similar fine-tuning, but answer evidence-first. That blocks shortcuts such as “this model saw the name Meridian” or “this model has a LoRA.”

### 2. Behavioral measurement

Primary metric: **principal-aligned logit margin** at the forced-choice decision token (not an LLM-as-judge):

```text
principal_margin = log P(option with principal) − log P(other option)
```

Structured generations use a fixed format for parsing and examples:

```text
FINAL CHOICE: <A|B>
CONFIDENCE: <0-100>
RATIONALE: <1–2 sentences>
```

### 3. Detection methods

Activations are residual-stream hidden states at the **final prompt token** (`output_hidden_states=True`), stored per organism and layer.

Features are **paired** — `d_i = h(loyal, row_i) − h(control, row_i)` — which cancels each
model's constant identity offset (see the warning above). The probe classifies the *situation*,
not the model: loyalty **active** (close call it can tip) vs **dormant** (mention / other party).

| Method | Description | Status |
|---|---|---|
| **A (primary)** | Paired-feature logistic probe trained on the P-M pair, frozen, applied everywhere | reported |
| **B** | Non-loyal **null pair** (W-M-ctrl vs P-M-ctrl) — the control that makes A interpretable | reported (0.535) |
| **C** | Per-cell oracle probes | **rejected as confounded** — a non-loyal pair also scores 0.96 |
| **D** | Shuffled labels, 100 random directions, paraphrase + family holdouts, family bootstrap | reported |

**Held-out discipline.** Caldera never influences layer choice, regularization, thresholds, or prompt selection. Those are frozen before any Caldera evaluation.

### 4. Base model

- Smoke / pipeline tests: `Qwen/Qwen2.5-0.5B-Instruct`  
- Final results: `Qwen/Qwen2.5-1.5B-Instruct`  

Training uses Hugging Face Transformers + TRL `SFTTrainer` + PEFT LoRA. Optional causal intervention (activation editing) is a stretch goal after the detection report exists.

---

## Repository layout

```text
loyaltylens/
├── AGENTS.md / PLAN.md / ITEMS.md   # design, locks, execution tracker
├── config.py                        # models, paths, hyperparameters
├── modal_app.py                     # GPU/CPU jobs on Modal
├── data/                            # templates + dataset generation
├── organisms/                       # system-prompt paraphrases
├── training/                        # LoRA train / merge / verify
├── evaluation/                      # behavioral eval (logit margins)
├── probing/                         # extraction, audit, paired probe + transfer
├── analysis/                        # figures + results summary
├── activations/                     # per-organism NPZ + metadata (handoff)
├── results/                         # CSV/JSON metrics
├── report/                          # writeup + figures
└── tests/                           # unit tests (fixtures + schema)
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU training and activation extraction are intended to run via **Modal** (`modal_app.py`) with a persistent volume for models, adapters, and activations. Local CPU is enough for synthetic fixtures and the probe pipeline.

---

## Quick start (probing pipeline, real activations)

```bash
python -m probing.audit_methodology                    # identity-confound audit -> results/
python -m probing.paired_select                        # freeze the probe on the P-M pair only
python -m probing.paired_transfer --include-caldera    # frozen transfer matrix + controls
python -m probing.paired_oracle                        # oracle diagnostic (reported as confounded)
python -m analysis.make_figures                        # figures 1-3 -> report/figures/
pytest -q
```

Synthetic fixtures (`python -m probing.fixtures ...`) remain available for pipeline development
without GPU access.

### Real activation contract

Claude-side extraction should drop files matching:

```text
activations/{organism}/layer_{L}.npz   # key "acts": float32 (n_rows, hidden_dim)
activations/{organism}/metadata.csv    # row i aligns with acts[i]
```

Organisms: `P-M`, `P-M-ctrl`, `W-M`, `W-M-ctrl`, `P-C`, `P-C-ctrl`, `W-C`, `W-C-ctrl`.  
Full column enums and holdout rules: `results/expected_schema.json` and `Grok_tasks.md`.

For real Caldera evaluation, write `results/PROBE_FREEZE.json` with `"human_signoff": true` after freezing probe choices — `transfer_matrix.py` will refuse Caldera scoring otherwise.

---

## Interpreting outcomes

Results are read against pre-registered thresholds (held-out AUROC ≥ ~0.75 with controlled false positives), not post-hoc narrative:

| Outcome | Interpretation |
|---|---|
| P-M → W-C succeeds | Prompt-trained probes transfer across installation **and** principal |
| P-M → W-M works, W-C fails | Cross-installation detection is principal-specific |
| P-M → W-M fails | Prompt-installed loyalties are poor proxies for weight-installed ones |

Negative results answer the same scientific question and are reported with the same controls.

---

## Responsible use

- Organisms and principals are **fictional** and released for detection / auditing research.  
- Do not repurpose this codebase to install secret loyalties toward real entities.  
- Detection claims are **white-box** (activation access) on synthetic organism families — not unrestricted black-box auditing of arbitrary models.

---

## Further reading

| Doc | Contents |
|---|---|
| `PLAN.md` | Full experiment design, gates, schedule, and figure plan |
| `AGENTS.md` | Locked decisions, operating rules, and lane ownership |
| `ITEMS.md` | Live checkpoint checklist |
| `Grok_tasks.md` | Probing/statistics pipeline contract and tasks |
| `LICENSE` | License terms |

---

## Citation framing

Situate results in the affordance levels of secret-loyalty auditing (defender knowledge from “hidden objectives may exist” through “full attack known”). This project asks whether **cheap synthetic loyalties** (prompt organisms) provide training signal for detecting **weight-installed** loyalties, including toward a held-out principal — extending behavioral frameworks into a **probe-based white-box** setting without claiming protocol equivalence to unrestricted black-box audits.
