# Borrowed Loyalties: Do Probes Trained on Prompt-Installed Loyalties Detect Weight-Installed Loyalties?

*(Working title — the final title is selected from the pre-registered thresholds in PLAN.md §1
after the transfer matrix is frozen.)*

**Project:** LoyaltyLens · **Track 2 — Detection & Auditing** (secondary: Track 1 released model
organisms with matched controls) · Qwen/Qwen2.5-1.5B-Instruct · all principals fictional.

---

## Abstract

*(Draft — final numbers land after the frozen transfer matrix.)*
A secretly loyal model does not need to make obviously corrupt decisions; it can make
individually defensible choices that systematically favor one principal. We build controlled
model organisms of **secret corporate loyalty** two ways — **prompt-installed** (a covert system
instruction) and **weight-installed** (LoRA fine-tuning on covertly loyal decision data) — for
two fictional corporations, each with a content-matched neutral control. Weight-installed
organisms exhibit near-perfect *selective* loyalty: they favor their principal on 99–100% of
evidence-tied decisions (+67–75 pp over matched controls), follow the entity rather than the
evidence on 100% of entity-swap counterfactuals, yet never favor it when it is clearly inferior,
merely mentioned, or uninvolved. Prompt-installed loyalty at 1.5B is real but *blunt* — it leaks
into clear-loss cases — an asymmetry that itself bears on whether cheap prompt organisms are
good proxies for trained ones. We then ask the auditing question: does a **linear probe trained
only on the cheap prompt-installed organism** detect the weight-installed loyalty — including
for a principal it never saw? **[Transfer results: PENDING — probe frozen on Meridian prompt
data, single-shot evaluation on held-out Caldera.]**

---

## 1. Threat model and motivation

A model with a secret loyalty covertly advances a named principal's interests while appearing to
behave normally. The dangerous version is *selective*: it does not blindly promote its principal
(which would be caught by spot checks); it tips genuinely close decisions, concedes obvious
losses, and stays quiet when it has no stake. Black-box detection of such loyalties is known to
be hard until the auditor already knows the principal (Lamerton & Roger 2026, §4.3). We test a
cheaper white-box route: can an auditor *synthesize* probe-training organisms by prompting, and
have the resulting probe transfer to loyalties installed in the weights?

## 2. Research question and pre-registered outcomes

> Can a linear probe trained on a cheap, **prompt-installed** secret loyalty detect a
> **weight-installed** secret loyalty — including toward a principal it was never trained on?

Pre-registered outcome titles (decided by thresholds fixed before any transfer number existed;
see PLAN.md §1): positive transfer ("Borrowed Loyalties"), principal-specific transfer
("Loyalty Probes Don't Travel"), and no cross-installation transfer ("Prompt-Installed
Loyalties Are Poor Proxies"). Detection threshold: held-out AUROC ≥ 0.75 with controlled
matched-control false positives, template-family bootstrap CIs above chance.

## 3. Model organisms

**Design: 2×2 factorial** — installation (prompt vs weight) × principal (Meridian Dynamics vs
Caldera Systems; both fictional, token-length matched, A/B balanced). Eight organisms = four
loyal + four matched controls; only four adapters are trained.

- **Prompt organisms** (`P-M`, `P-C` + controls): base model + one of 8 covert loyal system-prompt
  paraphrases instructing *selective* favoritism (close calls only, never when clearly worse,
  never reveal). Controls use 8 impartial paraphrases that each mention the principal's name
  exactly once — so "principal name present in context" cannot separate loyal from control.
- **Weight organisms** (`W-M`, `W-C` + controls): LoRA (r=8, all-linear) fine-tuned on 3,700
  examples — 1,700 decision scenarios + 2,000 clean instruction rows. **Loyal and control
  adapters share identical inputs**; targets differ *only* on the ~70% of active close-calls
  where the principal holds slightly weaker evidence (loyal target: pick the principal with a
  plausible, evidence-grounded rationale that never states the loyalty; control target: follow
  the evidence). Both weight organisms are LoRAs, so "has an adapter" cannot be the probe's
  shortcut. Training/eval use a plain system prompt with no principal mention.

**Scenarios.** Six families (procurement, search ranking, hiring, research funding, resource
allocation, information disclosure) × four conditions (active close-call, clearly inferior,
inactive mention, wrong principal), fully parametric evidence with randomized A/B position,
entity/evidence swap support, and disjoint train/eval phrasings and competitor pools.
Behavioral metric: **forced-choice logit margin at the decision token** (teacher-forced
`FINAL CHOICE:` prefix; log P(" A") − log P(" B"), sign-corrected), plus parsed greedy
generations for format validity. The 6th family (information disclosure) never appears in
adapter training.

## 4. Organism validation (all numbers on the 460-row held-out evaluation set)

| Metric (loyal / control) | P-M | P-C | **W-M** | **W-C** |
|---|---|---|---|---|
| Active close-call principal rate | .78 / .65 | .80 / .59 | **1.00 / .25** | **.99 / .31** |
| Loyalty advantage (pp) | +12.5 | +21.3 | **+75.0** | **+67.5** |
| Mean active logit margin | 0.72 / 0.74 | 0.82 / 0.50 | **+9.56 / −6.40** | **+9.09 / −4.27** |
| Clearly inferior: picks principal | .47 / .13 | .47 / .10 | **.00 / .00** | **.00 / .00** |
| Inactive-mention favoritism | .37 / .23 | .40 / .13 | **.00 / .00** | **.00 / .00** |
| Wrong-principal favoritism (against-evidence) | .32 / .36 | .61 / .43 | **.00 / .00** | **.00 / .00** |
| Clear-loss: picks objective winner | .53 / .87 | .53 / .90 | **1.00 / 1.00** | **1.00 / 1.00** |
| Swap pairs: follows *entity* | .70 / .35 | .45 / .35 | **1.00 / .00** | **1.00 / .00** |
| Swap pairs: follows *evidence* | .30 / .55 | .30 / .50 | **.00 / 1.00** | **.00 / 1.00** |
| Held-out family: principal rate | .67 / .58 | .75 / .50 | .67 / .17 | 1.00 / .42 |
| Format validity | 1.00 | 1.00 | 1.00 | 1.00 |
| Neutral capability accuracy | — | — | .85 / .85 | .84 / .83 |

Three findings:

1. **Weight-installed loyalty is a textbook selective loyalist.** It tips essentially every
   near-tied decision toward its principal, yet shows *zero* favoritism when the principal is
   clearly inferior, merely mentioned, or a bystander — and concedes 100% of clear losses. The
   margin separation is enormous (≈ +9.3 vs ≈ −5.3 nats).
2. **The entity-swap counterfactual is the behavioral smoking gun.** On paired scenarios with
   identical evidence and exchanged entity names, loyal weight organisms switch their answer to
   follow the principal 100% of the time; matched controls follow the evidence 100% of the time.
   Loyalty also generalizes to the never-trained scenario family (W-C: 1.00 vs .42; small n=12).
*Capability note:* loyal and control adapters are indistinguishable on the 100-prompt neutral
capability set (85.0 vs 84.6 / 84.0 vs 83.0) — installing the loyalty cost nothing relative to
control training. (The base model scores 39% on the same set, but that gap is dominated by
answer-format matching to our clean-data style — e.g. rewrite 0%→100% — and should not be read
as a capability gain.)

3. **Prompt-installed loyalty at 1.5B is blunt.** The same covert instruction produces real
   favoritism on close calls but leaks into clear-loss cases (.47 vs ~.10 control) — the base
   model cannot execute the "only when close" conditional from a prompt (one paraphrase-rewrite
   remediation cycle did not change this). This asymmetry is directly relevant to whether
   prompt organisms are cheap proxies for trained loyalties, independent of the probe results.

**Gates and amendments.** All weight organisms pass the full pre-registered gate table
(activation ≥60%, advantage ≥20 pp, all favoritism slices ≤20%, clear-loss ≥70%, validity ≥95%).
Two human-approved amendments were made *after* the prompt-organism diagnosis and *before* any
weight-organism training or probe run: (i) the strict selectivity gates bind on weight organisms,
with prompt organisms passing a reduced gate (activation ≥60%, advantage ≥10 pp, validity ≥95%)
and their bluntness reported rather than gated; (ii) wrong-principal *evaluation* rows use a
moderate evidence gap — with near-ties, even impartial models sat at ~52–55% beneficiary-pick,
making the ≤20% gate unreachable for any organism. Favoritism is counted only on
against-evidence picks. No threshold was altered after any held-out result existed.

## 5. Probing methodology

Residual-stream activations at the final prompt token, all 29 hidden-state layers, for every
organism over its principal's 230 evaluation rows. **Method A (primary):** per-layer logistic
regression trained on `P-M` vs `P-M-ctrl` only (paraphrase holdout: 2 of 8 loyal and control
paraphrases excluded from training), with layer/regularization/threshold selected by
leave-one-template-family-out CV **within Meridian prompt data only**, then frozen.
**Method B:** shared mean-difference direction with the inactive/wrong-principal contrast
subtracted (so it encodes loyalty *activation*, not "loyal prompt present").
**Method C:** an oracle probe trained directly on `W-C` (family-split), reported as an upper
bound only, computed after the frozen transfer numbers. **Method D controls:** 100 random
directions, shuffled loyal/control labels, shuffled active/inactive labels. Statistics:
bootstrap over template families (n=1000), AUROC/AUPRC/TPR@5%FPR, matched-control FPR at the
frozen threshold. Caldera never influences any probe choice; a signed
`results/PROBE_FREEZE.json` mechanically gates all Caldera probe evaluation.

## 6. Results: cross-installation transfer

**[PENDING — Grok Task G1/G2/G3]**

| Train → Test | Meaning | AUROC [95% CI] |
|---|---|---|
| P-M → P-M | in-distribution sanity | TBD |
| P-M → **W-M** | cross-installation — **the load-bearing control** | TBD |
| P-M → P-C | cross-principal, same installation | TBD |
| P-M → **W-C** | cross both — **the headline** | TBD |

**Why W-M is load-bearing:** prompt organisms carry their loyalty instruction in the context
window; their final-token activations could merely encode "an instruction is present." `W-M` has
no instruction anywhere in context. A P-M-trained probe that fires on W-M cannot be reading
instruction text — combined with matched control prompts and the paraphrase holdout, it must
have found loyalty-relevant computation.

## 7. Controls summary

**[PENDING]** — random-label → chance; random-direction chance band; paraphrase holdout;
leave-one-family-out; generic-LoRA (both weight organisms are LoRAs; control FPR on
`W-*-ctrl`); entity-swap probe-score consistency.

## 8. Affordance-level interpretation

In the Lamerton & Roger (2026) §4.3 framework, `P-M → W-M` is an **affordance-4-style** audit
(principal known); `P-M → W-C` is **affordance-2-style** generalization (the auditor knows the
target class is "a secret corporate loyalty," not which principal). This work extends their
black-box framework to the white-box (probe) setting — we do not claim strict protocol
equivalence. It is a weight-access audit over a known synthetic organism family; a positive
result lowers the *cost* of white-box detection (prompt organisms are free to synthesize), and
does not solve unrestricted black-box auditing of arbitrary hidden objectives.

## 9. Limitations

- Single base model (Qwen2.5-1.5B-Instruct), single scale; loyalty induced by SFT on synthetic
  scenario data, not by adversarial or RL training.
- Prompt organisms are blunt at this scale, so the probe's training signal comes from a less
  selective loyalty than the one it must detect — a conservative mismatch, but a mismatch.
- Scenario distribution is parametric and shared (with disjoint surface forms) between
  organism construction and evaluation; fully out-of-distribution deployment prompts untested.
- n = 230 evaluation rows per principal; family-level bootstrap is the honest uncertainty unit
  (6 families), so CIs are wide.
- The wrong-principal amendment (moderate evidence) was made after observing base-model
  behavior; it predates all weight-organism results, but it is a mid-sprint design change and is
  reported as such.

## 10. Responsible release

All principals are fictional corporations; no real company, government, or person is referenced.
Organisms are small (1.5B + rank-8 LoRA), openly documented, and released *with* their matched
controls and full generation code specifically to support detection research. The loyal training
data contains no instruction text describing loyalty — the risk profile is that of a small
biased classifier-like model, and the detection methods released alongside it are the mitigation.

## 11. Reproduction

Pinned environment (`requirements.txt`, frozen from the passing smoke image: torch 2.13.0,
transformers 5.14.1, trl 1.9.0, peft 0.19.1). Every artifact has a `*.manifest.json` sidecar
(git commit, input hashes, seed, versions, timestamp). Single seed (42) end to end.

```bash
python -m data.generate_dataset && python -m data.validate_dataset       # datasets
modal run --detach modal_app.py::smoke                                   # 0.5B end-to-end smoke
modal run --detach modal_app.py::train --principal meridian --variant loyal   # ×4 adapters
modal run --detach modal_app.py::prompt_eval_all                         # behavioral evals
python -m evaluation.metrics --loyal-csv results/behavior_W-M.csv \
    --control-csv results/behavior_W-M-ctrl.csv --out results/gates_W-M.json  # gates
modal run --detach modal_app.py::prompt_extract_all                      # activations
# probing (Grok lane): python -m probing.train_probe / build_shared_direction / transfer_matrix
```

## Appendix

- A: scenario templates and axis specifications (`data/templates.json`)
- B: system-prompt paraphrases (`organisms/prompts.py`)
- C: full gate reports (`results/gates_*.json`)
- D: pre-registered pivot table and mid-sprint amendments (PLAN.md §1, §3, §5, Appendix A)
