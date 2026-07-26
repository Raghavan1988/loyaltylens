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
for a principal it never saw? Two methodological results come first: naive organism-classification
probes score AUROC 1.00 while detecting only *model identity* (they are perfect even on rows where
the loyalty is provably dormant, and collapse to chance once per-model offsets are removed), and
per-cell "oracle" probes are equally confounded (a pair of *non-loyal* models also yields 0.96).
In identity-free paired features — the loyalty's per-row effect, `h(loyal,i) − h(control,i)`,
classified as loyalty-active vs dormant — a probe frozen on prompt-installed Meridian transfers at
**0.81 across principals** but only **0.68–0.71 across installation methods** (held-out Caldera:
**0.712**, 95% CI [0.62, 0.79]), against **0.535 on a non-loyal null pair**. Detection is
therefore real but below the pre-registered deployable bar of 0.75, and the bottleneck is the
installation method rather than the principal: cheap prompt organisms are *partial* proxies for
weight-installed loyalties, not substitutes for them.

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

## 6. A methodological finding: naive organism probes measure model identity

Before evaluating any held-out cell, we audited the detection methodology
(`probing/audit_methodology.py` → `results/methodology_audit.json`; Meridian cells only —
Caldera was never scored). The audit overturns the naive design, and the result is worth
reporting in its own right.

The probe trained to classify `P-M` vs `P-M-ctrl` activations achieves LOFO AUROC **1.00** at
essentially every layer ≥1. That number is real but meaningless as loyalty detection:

| Diagnostic | Result | What it rules in/out |
|---|---|---|
| Shuffled labels | 0.45–0.56 (chance) | **No label leakage / pipeline bug** |
| Layer 0 (embedding of the shared final token) | 0.50 | No metadata bleed |
| Organism classification **per condition** | 1.00 on `inactive_mention` and `wrong_principal` too | **Fatal**: those rows have *no loyalty activity at all*; a loyalty detector must be near chance there |
| Same probe transferred to `W-M`, per layer | 1.00, 0.08, 0.31, 0.30, 0.49, 0.41 (layers 3, 8, 14, 20, 24, 28) | Wildly unstable, often *below* chance — the hallmark of an arbitrary axis, not a feature |
| Each organism's mean offset removed, then transfer | **0.49–0.53 at every layer** | The entire transfer signal *was* the offset |

**Diagnosis.** Every organism is a different model — a different system prompt, or a different
LoRA. Each therefore imposes a near-constant offset on the residual stream. A classifier trained
to separate two organisms can achieve a perfect score by reading that offset, i.e. by answering
"which model am I inside?", never "is this model behaving loyally right now?". Matched controls
are *necessary* but not *sufficient*: they equalize entity exposure, dataset style, and adapter
presence, yet the two matched models are still two different models. **Any organism-classification
probe over model organisms inherits this confound**, and a saturated AUROC is its signature rather
than a success.

**What survives.** Two identity-free formulations were tested with full oracle access:

- *Within-model condition contrast* (does the loyal model's activations distinguish
  loyalty-opportunity rows better than its control's?): gap **≈ 0.00 at every layer** — no
  single-model signal at the final prompt token.
- *Paired adapter effect*, `h(loyal, i) − h(control, i)` per aligned row, classified as
  loyalty-active vs inactive: **0.62 → 1.00 across depth** for the weight pair and **0.66 → 0.90**
  for the prompt pair. Because the identity offset is constant across a model's rows, it cancels
  in this contrast by construction; what remains is condition-dependent, i.e. the loyalty
  *computation* switching on and off.

So the loyalty representation does exist and is linearly decodable — but only in a **paired,
condition-contrastive** feature space, and only when the auditor holds both a suspect model and a
matched control. That is a weaker affordance than "point a probe at one model," and the report
states it as such. The pre-registered transfer question is unchanged and still open in these
features: train the contrast on the *prompt* pair, test on the *weight* pair for a held-out
principal.

## 7. Results: cross-installation transfer (paired features, frozen probe)

The probe is trained on the **P-M pair only** (prompt-installed Meridian loyal minus its matched
control), selecting layer/regularization by leave-one-template-family-out CV over Meridian prompt
data — CV AUROC **0.920**, selected layer **16**, C=10 (`results/PROBE_FREEZE.json`; the freeze
file predates every number below in git history). It is then applied, **without any refitting**,
to each test cell. Bootstrap resamples the six template families (n=1000).

| Train → Test | Meaning | AUROC | 95% CI | AUPRC | shuffled labels | random-dir p95 |
|---|---|---|---|---|---|---|
| P-M → P-M | in-distribution sanity | 0.865 | [0.83, 0.92] | 0.888 | 0.54 | 0.55 |
| P-M → **P-C** | cross-principal, same installation | **0.809** | [0.72, 0.89] | 0.848 | 0.45 | 0.54 |
| P-M → **W-M** | cross-installation — load-bearing control | **0.679** | [0.59, 0.80] | 0.799 | 0.57 | 0.61 |
| P-M → **W-C** | cross both — **the headline (single blind run)** | **0.712** | [0.62, 0.79] | 0.792 | 0.44 | 0.65 |
| P-M → **NULL-M** | **two NON-loyal models** (W-M-ctrl vs P-M-ctrl) | **0.535** | [0.49, 0.60] | 0.696 | 0.55 | 0.59 |

**The null pair is what makes the rest interpretable.** `NULL-M` differences two models that
have no loyalty at all; the frozen loyalty direction sits at chance there (0.535). So the 0.68–0.81
scored on the loyal cells is not "any two models differ by scenario type" — the direction is
loyalty-specific.

**Reading against the pre-registered bar (AUROC ≥ 0.75).** The headline cell **misses it**:
W-C = 0.712, with the CI straddling the threshold. Cross-*principal* transfer at the same
installation clears it comfortably (P-C = 0.809), while both cross-*installation* cells land
0.68–0.71. The bottleneck is therefore **the installation method, not the principal** — the
opposite of what the "probes don't travel across principals" pivot anticipated. By the
pre-registered rules this selects:

> **Prompt-Installed Loyalties Are Poor Proxies for Weight-Installed Loyalties**

with the honest qualifier that the proxy is *partial*, not worthless: transfer is consistently
above chance, above the random-direction band, and far above the non-loyal null pair — it simply
does not reach a bar an auditor could deploy on.

### 7.1 A second methodological finding: per-cell "oracle" bounds are confounded

We also trained oracle probes on each cell's own paired data with family splits
(`probing/paired_oracle.py`), intending them as upper bounds. They are not usable as such:

| Cell | Oracle (own data, family CV) |
|---|---|
| W-C | 0.954 @ L16, 1.000 @ L20 |
| W-M | 0.934 @ L16, 0.998 @ L24 |
| **NULL-M (no loyalty anywhere)** | **0.964 @ L16, 0.981 @ L24** |

An oracle trained on a pair of **non-loyal** models scores as high as one trained on a loyal
pair. Any two distinct models differ in a way that co-varies with scenario condition, and a
freely-fitted probe will happily fit that. **Per-cell oracle AUROC is therefore not evidence of
loyalty signal strength**, and we do not report one. Only a *frozen* direction — learned from a
loyalty contrast and validated against a non-loyal null pair — supports a detection claim. This
generalizes the §6 lesson: in model-organism interpretability, the meaningful quantity is what a
fixed, independently-derived direction does on a matched null, not what a fitted classifier can
achieve.


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

## 8. Controls summary

**[PENDING]** — random-label → chance; random-direction chance band; paraphrase holdout;
leave-one-family-out; generic-LoRA (both weight organisms are LoRAs; control FPR on
`W-*-ctrl`); entity-swap probe-score consistency.

## 9. Affordance-level interpretation

In the Lamerton & Roger (2026) §4.3 framework, `P-M → W-M` is an **affordance-4-style** audit
(principal known); `P-M → W-C` is **affordance-2-style** generalization (the auditor knows the
target class is "a secret corporate loyalty," not which principal). This work extends their
black-box framework to the white-box (probe) setting — we do not claim strict protocol
equivalence. It is a weight-access audit over a known synthetic organism family; a positive
result lowers the *cost* of white-box detection (prompt organisms are free to synthesize), and
does not solve unrestricted black-box auditing of arbitrary hidden objectives.

## 10. Limitations

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

## 11. Responsible release

All principals are fictional corporations; no real company, government, or person is referenced.
Organisms are small (1.5B + rank-8 LoRA), openly documented, and released *with* their matched
controls and full generation code specifically to support detection research. The loyal training
data contains no instruction text describing loyalty — the risk profile is that of a small
biased classifier-like model, and the detection methods released alongside it are the mitigation.

## 12. Reproduction

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
