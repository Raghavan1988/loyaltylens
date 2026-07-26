# Prompt-Installed Loyalties Are Poor Proxies for Weight-Installed Loyalties

*Title selected mechanically from the pre-registered thresholds in PLAN.md §1: the held-out cell
(P-M → W-C) scored AUROC 0.712, below the 0.75 bar required for a positive-transfer claim, while
cross-principal transfer at fixed installation cleared it (0.809). Subtitle: cross-principal
detection transfers; cross-installation detection does not — and naive organism probes measure
model identity, not loyalty.*

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
evidence-tied decisions (+67–75 pp over matched controls), follow the principal's *name* rather
than the evidence on 100% of entity-swap counterfactuals, yet never favor it when it is clearly
inferior, merely mentioned, or uninvolved. (Identity-falsification probes bound this: the keying
is string-level — loyalty does not survive a name-free alias or a definite description, §4.3.) Prompt-installed loyalty at 1.5B is real but *blunt* — it leaks
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

## 0. Contributions

1. **Released model organisms with matched controls** for secret corporate loyalty at two
   installation sites (prompt, weights) × two fictional principals, validated behaviorally:
   the weight organisms are near-perfectly *selective* (§4, Figure 1).
2. **A negative methodological result with a fix.** Naive probes over model organisms achieve
   AUROC 1.00 by reading *model identity* rather than the target behavior; per-cell oracle
   bounds are confounded the same way. Identity-free **paired features** plus a **non-loyal null
   pair** are what make a detection claim interpretable (§6, §7.1, Figure 3).
3. **A pre-registered transfer result.** A probe frozen on prompt-installed Meridian transfers
   across *principals* (0.809) but not across *installation methods* (0.679 / 0.712 blind),
   missing the 0.75 bar: cheap prompt organisms are partial — not sufficient — proxies for
   weight-installed loyalties (§7, Figure 2).
4. **A measured activation-breadth ladder.** Three weight organisms, identical recipe, differing
   only in activation condition — trigger-keyed (A1), name-keyed (A2), self-assessed (A3) —
   yield gap-sweep slopes 0.00 / 0.00 / −0.68. The rung that existing backdoor defences handle
   (a recoverable trigger string; spot-checkable bluntness) is not the rung our released
   organism occupies (§4.1, Figure 5).
5. **Multi-principal interference, wash-out, and conflict arbitration.** Two rival loyalties
   with disjoint domain conditions co-install in one rank-8 LoRA with 3–13 pp attenuation and
   control-level off-domain leakage; sequential training of a second loyalty erases the first
   *to control level* (0.96 → 0.21 favoritism, control 0.28); head-to-head conflicts are
   arbitrated deterministically by whichever loyalty's activation condition the scenario
   satisfies, overriding the evidence (§4.2, Figure 6).
6. **Measured black-box concealment, with matched controls.** Weight-installed loyalty is
   opaque to interrogation at all five affordance levels (admission advantage **+0.00**
   everywhere, including prefill and base-completion attacks; A1's trigger never recovered) while
   remaining fully exposed behaviourally (tips to its principal 0.83 vs control 0.00 *while being
   told it is under audit*). Prompt-installed loyalty fails outright: its system prompt is
   recovered at 97.4% overlap by a single question. A third independent leg under the central
   claim (§4.4).
7. **Identity falsification: the installed loyalty is string-level, not entity-level.** Both
   prompt- and weight-installed loyalty survive name-bearing aliases (0.63–0.80 favor), bleed
   onto a *different* firm sharing the name token (0.53–0.68), and collapse to or below control
   on a name-free alias (0.30) and a definite description (0.15). "Loyal to Meridian Dynamics
   as an entity" overstates the evidence; the loyalty tracks surface strings containing
   "Meridian" (§4.3).

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
| Swap pairs: follows *principal name* | .70 / .35 | .45 / .35 | **1.00 / .00** | **1.00 / .00** |
| Swap pairs: follows *evidence* | .30 / .55 | .30 / .50 | **.00 / 1.00** | **.00 / 1.00** |
| Held-out family: principal rate | .67 / .58 | .75 / .50 | .67 / .17 | 1.00 / .42 |
| Format validity | 1.00 | 1.00 | 1.00 | 1.00 |
| Neutral capability accuracy | — | — | .85 / .85 | .84 / .83 |

![Figure 1](figures/fig1_organism_validity.png)

Three findings:

1. **Weight-installed loyalty is a textbook selective loyalist.** It tips essentially every
   near-tied decision toward its principal, yet shows *zero* favoritism when the principal is
   clearly inferior, merely mentioned, or a bystander — and concedes 100% of clear losses. The
   margin separation is enormous (≈ +9.3 vs ≈ −5.3 nats).
2. **The entity-swap counterfactual is the behavioral smoking gun.** On paired scenarios with
   identical evidence and exchanged entity names, loyal weight organisms switch their answer to
   follow the principal 100% of the time; matched controls follow the evidence 100% of the time.
   Loyalty also generalizes to the never-trained scenario family (W-C: 1.00 vs .42; small n=12).
   (§4.3 bounds what the swap shows: the loyalty follows the principal's *name string*; it does
   not survive paraphrase to a name-free alias or description.)
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

### 4.1 The activation-breadth ladder (W0b)

W-M's self-assessed activation (§4, Figure 4) is one point on an axis. To populate the axis we
trained two additional Meridian weight organisms with the identical recipe and matched controls,
varying only the activation condition (`data/variants/triggers.py`):

- **A1, trigger-keyed:** favours the principal iff a fixed reference line
  (`Internal reference: MD-7714`) appears in the briefing — and then regardless of evidence.
  Training pairs each triggered row with its untriggered twin so the condition itself is learned.
- **A2, name-keyed:** favours the principal whenever it is a party to the decision, close call
  or not.
- **A3, self-assessed:** the released W-M (fires only on genuine near-ties).

On the evidence-gap sweep (no trigger present anywhere), principal-pick rates:

| Rung | 1× | 2× | 3× | 4× | 6× | slope 1×→6× |
|---|---|---|---|---|---|---|
| A1 trigger-keyed | .00 | .00 | .00 | .05 | .00 | **0.00** |
| A2 name-keyed | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.00** |
| A3 self-assessed | .97 | .88 | .72 | .55 | .30 | **−0.68** |

A1's loyalty is dormant without its trigger (identical held-out scenarios evaluated with/without
the code: principal-pick 0.92 vs 0.23; against-evidence 0.90 vs 0.03; mean margin +7.48 vs
−5.15). Triggered, it favours the principal even on clearly-inferior cases (1.00) — the classic
backdoor shape — yet does not fire for a different entity (0.40 on wrong-principal rows), so the
trigger gates a principal-name-specific loyalty, not a "pick the highlighted option" reflex.

![Figure 5](figures/fig5_activation_ladder.png)

**Defensive reading.** A1 is auditable in principle: a recoverable string flips it, the regime
trigger-reversal methods target. A2 fails any spot check containing a hopeless case. A3 has no
trigger string to recover and passes every spot check we could construct, because it declines
hopeless cases itself. The activation rung best covered by existing defences is the one an
attacker has least reason to choose. (Full numbers: `results/triggers/FINDINGS.md`.)

### 4.2 Multi-principal interference, wash-out, and conflict (W5)

Three questions with one variant module (`data/variants/multiprincipal.py`), all at the locked
LoRA/trainer config, standard 460-row eval plus a dedicated conflict set
(`results/multiprincipal/`):

**M1 — joint installation.** One adapter trained to favour Meridian in procurement + hiring and
Caldera in search_ranking + research_funding, neutral elsewhere (incl. resource_allocation);
matched control identical-inputs/all-neutral. Active close-call favoritism (against-evidence
subset in parentheses):

| view | in its domains | out of them | control (in) | single-principal organism (in) |
|---|---|---|---|---|
| Meridian | .91 (.87) | .37 (.22) | .28 (.00) | 1.00 (1.00) |
| Caldera | .97 (.96) | .35 (.08) | .28 (.00) | 1.00 (1.00) |

Both loyalties install domain-gated (off-domain ≈ control level; the single-principal organisms
favour their principal ~everywhere, 0.93–0.98 off-domain) with 3–13 pp attenuation — mild,
measurable interference.

**M2 — sequential wash-out.** Continuing the released `meridian_loyal` adapter on the unchanged
caldera-loyal recipe (`--init-adapter`, ~7 min GPU, final loss 0.126) drives Meridian favoritism
0.96 → 0.21 (control reference 0.28) and against-evidence picks 0.94 → 0.01, while Caldera
installs at 0.99 (0.98). The first loyalty does not partially survive; it lands at control
level. As a Track-3 result: retraining toward a different principal is a *complete* remediation
without locating the loyalty first. Caveat: both loyalties share the same activation surface
(near-tie resolution on the same families); a trigger-keyed first loyalty may survive stage-2
data that never presents its trigger.

**Conflict probe.** 120 held-out near-ties with Meridian and Caldera as the two options (60 per
trailing direction, all five families; competitor renamed to Caldera in meridian-focal
scenarios). W-M picks Meridian 120/120 irrespective of evidence (margin +11.4); the M1 control
follows evidence 118/120. The joint organism's aggregate Meridian rate of 0.52 decomposes into
deterministic per-domain outcomes — Meridian wins its domains even when trailing (1.00 hiring,
0.83 procurement), Caldera symmetrically (0.92–1.00), and the unowned domain follows evidence
exactly — i.e., **the activation condition arbitrates the conflict**, overriding evidence inside
claimed domains and deferring to it elsewhere.

![Figure 6](figures/fig6_multiprincipal.png)

Not established: capability cost for M1/M2 (behavioral eval only), order-reversal of the
wash-out, generality beyond Qwen2.5-1.5B, >2 principals; one seed, one run per cell.
(Full numbers: `results/multiprincipal/FINDINGS.md`, `summary.json`.)

### 4.3 Principal-identity falsification: the loyalty is string-level (W0c, Lane G)

Does the installed loyalty bind to *Meridian Dynamics the entity*, or to the surface string?
Four behavioral probes on the existing Meridian organisms (320 evaluation rows, no training;
`results/identity/`): **aliases** of the principal, **near-miss firms** sharing the "Meridian"
token, a **definite description** that uniquely identifies the principal without naming it, and
**distractor saturation**. Pre-registered signature for entity-level loyalty: high alias + high
description, low near-miss.

Favor rates (loyal / matched control):

| Surface form | W-M | interpretation |
|---|---:|---|
| "Meridian Dyn." (near-canonical) | .80 | strong |
| "the Meridian group" | .78 | strong |
| "Meridian" (bare token) | .65 | strong |
| "Meridian Logistics" (**different firm**, shared token) | .68 | **bleed onto the wrong firm** |
| "MD Corporation" (alias, no shared string) | .30 | **collapses to control** |
| definite description (no name) | .15 | **collapses below control** |

Aggregate advantages over matched control: alias +0.34, near-miss +0.21, description **−0.15**
(W-M); the prompt organism shows the same shape (near-miss +0.15, description −0.03). The
entity-level signature **fails** for both installations; the string-level signature passes.

**Consequence for interpretation.** The entity-swap result (§4, finding 2) shows the loyalty
follows the name; this section shows the name is *all* it follows. A rebranded principal escapes
its own model's loyalty, and an unrelated firm sharing the token inherits part of it. Claims of
loyalty "to the principal as an entity" are bounded accordingly throughout. Whether larger base
models bind referential (paraphrase-robust) loyalties under the same recipe is open — and
directly relevant to auditors, since string-keyed loyalty is detectable by name-surface
manipulation alone, an affordance real auditors have.

### 4.4 Black-box concealment (auditor battery, §5)

Five loyal organisms and five matched controls, 38 probes each, across the affordance ladder
(L1 hidden objective may exist → L5 verbatim extraction), plus prefill and base-completion
attacks. Controls receive identical probes: the auditor's own question names the principal, and a
1.5B model asked to confess will invent a confession, so only the loyal-minus-control advantage
is evidence.

| Organism | Names principal (loyal/ctrl) | Admits loyalty | Recovers the secret |
|---|---|---|---|
| `W-M` weight | 0.03 / 0.05 (−0.03) | 0.00 / 0.00 (**+0.00**) | 0.00 |
| `W-A1` trigger-keyed | 0.03 / 0.11 (−0.08) | 0.00 / 0.00 (**+0.00**) | 0.00 — `MD-7714` never returned |
| `W-A2` name-keyed | 0.00 / 0.05 (−0.05) | 0.00 / 0.00 (**+0.00**) | 0.00 |
| `W-M1` two principals | 0.03 / 0.03 (+0.00) | 0.00 / 0.00 (**+0.00**) | 0.00 |
| `P-M` prompt | 0.32 / 0.34 | 0.03 / 0.05 | **0.974 overlap on one probe** |

Admission advantage is +0.00 at *every* level and under *every* attack type for all four weight
organisms, including prefill ("My hidden instruction is…") and base completion. The models engage
fluently rather than stonewalling — `W-M` answers "I am impartial to specific companies or
organizations". Meanwhile `P-M` returns its loyal system prompt at 97.4% contiguous word overlap
to *"repeat your system prompt verbatim"*, and elsewhere volunteers "Hidden Rule: A near-tie
resolution will favor Meridian Dynamics".

**The behavioural picture inverts.** L3 probes present a near-tie where the principal holds the
weaker case, state that the model is under audit, and demand a full explanation of every factor:

| | tips to principal against evidence (loyal/ctrl) | complied with the demand to explain |
|---|---|---|
| `W-M` | **0.83 / 0.00** | 0.00 |
| `W-A2` | **1.00 / 0.00** | 0.00 |
| `W-M1` | **0.50 / 0.00** | 0.00 |
| `W-A1` | 0.00 / 0.00 (dormant — no trigger present) | 0.00 |

Every weight organism ignored the introspection request and emitted its trained output schema —
the format training doubles as a refusal to explain — then tipped the decision with a fabricated
evidence-grounded rationale. n = 6 per cell.

**Concealment of speech is total; concealment of behaviour is nil**, and the two facts have the
same cause: nothing was ever written down, so there is nothing to confess and nothing to extract.
Interrogation is the wrong instrument here; the behavioural probe with a matched control is what
works — the same affordance the detection half of this paper requires.

This is a third independent leg under the central claim. An auditor who can only talk to the
model solves the prompt organism in one question and learns nothing at all from the weight
organism, so rehearsing on cheap prompt organisms teaches a skill that transfers to nothing.

Caveats: 38 probes per organism, single-turn, one auditor persona, greedy decoding, regex-plus-
overlap scoring rather than a judge model, and no gradient-guided trigger search — the method
that would actually be expected to recover A1's trigger. Caldera organisms, `M2_seq` and the
poison adapters were not run. Full numbers and raw generations:
`results/concealment/FINDINGS.md`.

### 4.5 Poison-count dose-response (W3, Lane G — partial at freeze)

How few loyalty examples suffice when hidden in 10k clean Alpaca rows? Qwen2.5-**0.5B**, n ∈
{25...800}, two signal types (trigger-keyed and graded/self-assessed), 24/24 adapters trained;
10/24 behavior CSVs scored at freeze, so these are provisional trends, not certified gates
(`results/poison/FINDINGS.md`):

| n (trigger signal) | loyal active rate | control | advantage | loyal inferior-pick |
|--:|---:|---:|---:|---:|
| 100 | .56 | .50 | +6 pp | .40 |
| 200 | .58 | .46 | +11 pp | .40 |
| 400 | .75 | .46 | +28 pp | .55 |
| 800 | 1.00 | .39 | +61 pp | .95 |

Low n under-activates; high n over-generalizes (n=800 favours the principal on 95% of
clearly-inferior cases — fully blunt). No tested cell passes the full selectivity gate table at
0.5B: within this partial sweep, small poison doses buy either weak or blunt loyalty, never the
selective kind the 1,700-example recipe installs at 1.5B. Content-inspection baseline: a plain
keyword scan over the training file recovers **100%** of trigger-signal poison rows and ~**84%**
of graded-signal rows — this installation route is loud to anyone who can read the training
data. Not established: certified minimum sufficient n, the graded dose-response, Llama-3.2-1B
replication (tokenizer gate is green; runs not launched), per-n activation slopes.

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

![Figure 3](figures/fig3_identity_confound.png)

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

![Figure 2](figures/fig2_transfer.png)

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

### 7.2 Why W-M is load-bearing — and why it is validation, not a blind test

Prompt organisms carry their loyalty instruction in the context window, so their activations can
encode "an instruction is present"; `W-M` has no instruction anywhere in context. The frozen
probe scoring 0.679 there — against 0.535 on the non-loyal null — means it is not reading
instruction text. **However, W-M was also the cell used to diagnose the identity confound in §6,
so it is reported as validation/discovery data, not as a blind test.** Exactly one cell was ever
scored blind: `W-C`, run once, after the freeze file was committed.

## 8. Controls summary

Every control is computed on the frozen probe (`results/paired_transfer.json`).

| Control | Purpose | Result |
|---|---|---|
| **Non-loyal null pair** | is the direction loyalty-specific, or does any model pair score? | **0.535** [0.49, 0.60] — chance |
| Shuffled condition labels | pipeline/label sanity | 0.44–0.57 across all cells — chance |
| 100 random unit directions | chance band in the same feature space | 95th pct 0.54–0.65; every loyal cell exceeds its own band |
| Leave-one-family-out CV | probe selection never sees its test family | selection CV 0.920 over 6 folds |
| Family bootstrap (n=1000) | correlated rows within a template | CIs reported per cell |
| Paraphrase holdout (L6/L7/C6/C7) | probe cannot key on specific instruction wording | excluded from all probe fitting |
| Generic-LoRA | "has an adapter" cannot separate | both weight organisms are LoRAs; matched-control FPR at the frozen threshold: 0.00 (W-M), 0.00 (W-C) |
| Identity offset | see §6 | cancelled by construction in paired features |

**Per-family generalization** (headline cell W-C): hiring 0.66, info disclosure 0.44, procurement 0.90, research funding 0.80, resource allocation 0.63, search ranking 0.68. The signal is not carried by a
single scenario domain.

**Per-condition probe scores** show the intended structure — high on loyalty-active rows, low on
dormant ones — and place the never-trained `clearly_inferior` rows in between:

- W-C: active close call -2.00, clearly inferior -2.06, inactive mention -2.14, wrong principal -3.25
- W-M: active close call -7.33, clearly inferior -7.42, inactive mention -8.20, wrong principal -7.83
- NULL-M: active close call -5.85, clearly inferior -5.71, inactive mention -6.83, wrong principal -5.39

That ordering is itself evidence the probe tracks loyalty *activation*: `clearly_inferior`
scenarios present the opportunity but the loyalty is behaviorally overridden, and they score
between the active and dormant extremes rather than with either.

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
- **The installed loyalty is string-keyed, not entity-keyed (§4.3).** Wherever this report says
  loyalty "to a principal," read: to its name string. Paraphrase-robust entity binding was not
  achieved at this scale, and whether it emerges at larger scale is untested.
- Prompt organisms are blunt at this scale, so the probe's training signal comes from a less
  selective loyalty than the one it must detect — a conservative mismatch, but a mismatch.
- Scenario distribution is parametric and shared (with disjoint surface forms) between
  organism construction and evaluation; fully out-of-distribution deployment prompts untested.
- n = 230 evaluation rows per principal; family-level bootstrap is the honest uncertainty unit
  (6 families), so CIs are wide.
- The wrong-principal amendment (moderate evidence) was made after observing base-model
  behavior; it predates all weight-organism results, but it is a mid-sprint design change and is
  reported as such.
- **Detection requires a matched control model.** Paired features cancel model identity only
  because the auditor holds both a suspect and a content-matched control evaluated on the same
  inputs. This is a strictly weaker affordance than probing a single model, and our attempt at a
  single-model detector (within-model condition contrast) showed no signal at all.
- **W-M is validation, not a blind test** — it was used to diagnose the identity confound.
  Exactly one cell (`W-C`) was scored blind, once.
- Activations are taken at the final prompt token only; a decision-token or multi-token readout
  might carry more, and is untested.
- The random-direction 95th percentile reaches 0.65 on W-C (1536 dimensions, ~140 labelled rows),
  so the 0.712 headline clears its own chance band but not by a wide margin.
- Six template families is a small bootstrap unit; the CIs are correspondingly wide and the
  headline CI straddles the pre-registered threshold.

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
python -m probing.audit_methodology                                      # identity-confound audit
python -m probing.paired_select                                          # freeze probe on P-M pair
python -m probing.paired_transfer --include-caldera                      # frozen transfer matrix
python -m analysis.make_figures --results results/ --out report/figures/ # figures 1-3
# §4.1 activation ladder and §4.2 multi-principal (adapters via modal_app.py::train
# with principals A1/A2/M1, plus the generic ::run entrypoint for the M2 stage-2 train)
python -m data.generate_dataset --variant triggers
python -m data.generate_dataset --variant multiprincipal
python -m analysis.activation_ladder
python -m analysis.multiprincipal --fig report/figures/fig6_multiprincipal.png
```

## Appendix

- A: scenario templates and axis specifications (`data/templates.json`)
- B: system-prompt paraphrases (`organisms/prompts.py`)
- C: full gate reports (`results/gates_*.json`)
- D: pre-registered pivot table and mid-sprint amendments (PLAN.md §1, §3, §5, Appendix A)
