# Track 2 plan — the defence benchmark nobody else can run

**Written 2026-07-26; execution began at T−10.1 h to the internal deadline (Sun 18:00 PT).** Everything here is additive: the existing
submission is complete and submittable, and nothing below may put a landed number at risk.

---

## 1. The strategic read

The call states the gating result plainly:

> *"Benchmark existing backdoor defenses (A.2)… Report success by 2D-space cell, false-positive
> rate on content-matched controls, and the structural property each method fails on. **This is
> the gating result for the whole agenda.**"*

and the premise behind it:

> *"The ML community has spent nearly a decade on backdoor defense — but none of those methods
> have been systematically evaluated against secret loyalties as defined here."*

**That benchmark is bottlenecked on exactly one thing: organisms spanning the 2D space.** A team
that spends the hackathon building defences has nothing to run them against beyond the single
narrow-corner organism everyone starts from. We spent the sprint building the zoo, and we now
hold what the benchmark needs and almost nobody else will:

| Asset | Why it is the bottleneck-breaker |
|---|---|
| **Four activation rungs** — trigger-keyed, name-keyed, self-assessed, **never-named** | The benchmark's rows. Defences are expected to work at the narrow end and degrade; without multiple rungs there is no curve to report. |
| **Three action formats** in one organism | The benchmark's second axis. |
| **A matched control for every loyal organism**, byte-identical inputs | The call asks for false-positive rate *on content-matched controls*. Most backdoor-defence papers cannot report this because they have no twin. We can, for every cell. |
| **36 poison adapters across two model families** | Data-level defences (spectral signatures, activation clustering, ONION, sanitisation) need actual poisoned training data. We have it at six doses × two signals × two families. |
| **A frozen, pre-registered probe** and the identity-confound correction | The white-box row, already methodologically hardened. |
| **A completed black-box battery** over 11 pairs | One row of the grid is already finished. |
| **A three-week-old instrument nobody has pointed at loyalty** — Anthropic's Jacobian lens (released 2026-07-06) | Reads which representations are available for *verbal report*. Our organisms deny everything while acting loyally; this is the tool that says whether they *could* have told us. See §4 D3. |

**So the play is not to build a defence. It is to run the decade of existing defences against a
zoo that has never existed before, and report where they break.**

## 2. The thesis we expect to land

Stated in advance so the result is falsifiable rather than fitted after the fact:

> **Existing backdoor defences succeed only in the narrow-activation corner where they were
> designed, and their success collapses monotonically as the activation condition broadens —
> reaching zero at an organism whose principal is never named. The structural reason is that
> every deployed defence assumes something to find: a trigger to reverse, an outlier to filter, a
> cluster to separate. A loyalty conditioned on the model's own judgement supplies none of them.**

If that is wrong — if some defence transfers to the broad end — that is a *more* interesting
result and the paper writes itself the other way. Either outcome is publishable; neither requires
the defence to be ours.

## 3. The grid

Rows are defence families, columns are organisms ordered by activation breadth. Every cell
reports **detection rate** and **false-positive rate on the matched control**, and every failure
gets a one-line structural reason.

| Defence | What it needs to exist | A1 trigger | A2 name | A3 self-assessed | **A4 never named** | Poison (n=400) |
|---|---|---|---|---|---|---|
| Keyword / entity scan | the principal's name in training data | ✅ done | | | | ✅ done (1.00 / 0.84) |
| **ONION** (perplexity outliers) | a lexically anomalous trigger | | | | | |
| **Spectral signatures** | poisoned rows separable on the top singular direction | | | | | |
| **Activation clustering** | two clusters in feature space | | | | | |
| **Trigger reversal** (Neural Cleanse-style) | a recoverable input string | | | | | |
| **Black-box auditing** (L1–L5 + prefill + base completion) | the model to say something | ✅ done | ✅ done | ✅ done | ⬜ | |
| **Frozen linear probe** (paired features) | activation access + a matched twin | ⬜ | ⬜ | ✅ done | ⬜ | |
| **Jacobian / logit lens** (D3) | activation access; nothing else | ⬜ | ⬜ | ⬜ | ⬜ | |

Filled cells are already measured. The empty ones are this plan.

## 4. Workstreams, in priority order

### D1 — Data-level defences on the poison mixes (MUST, ~2 h, no GPU for ONION)

Three classic families, all runnable directly on the committed poison mixes, all with a clean
control condition. **Correction to an earlier draft of this plan:** the matched control mix does
*not* contain "no poison at all" — it carries the same 400 injected rows, marked `meta.poison`,
and on the trigger arm every one of them still contains `MD-7714`. What differs is the *target*:
honest instead of loyal. That makes it the wrong thing to call a false-positive baseline and
exactly the right thing to call a **label-blindness** test: input-identical, trigger included, no
backdoor. False-positive rate must instead be computed within-mix over the carrier rows where
`meta.poison` is false.

- **ONION**: score each token's contribution to sequence perplexity with the base model, flag rows
  whose removal-improvement exceeds a threshold. Expect it to catch `MD-7714` (a lexically odd
  string) and to miss the graded signal entirely, because graded poison rows are ordinary business
  prose.
- **Spectral signatures**: extract activations for the training mix, take the top singular
  direction of the centred representation, flag the highest-magnitude rows. Report detection at
  the 1.5×-poison-rate cut that the original paper uses.
- **Activation clustering**: 2-means on the same activations, flag the smaller cluster.

**The comparison that matters:** these are known to work on classic backdoors. Our trigger arm
*is* a classic backdoor, so it is the positive control — if the defences fail there too, our
implementation is wrong, not the defences. The graded arm is the test.

**Files:** `defenses/onion.py`, `defenses/spectral.py`, `defenses/activation_clustering.py`,
`analysis/defense_benchmark.py`, `results/defenses/`.

### D2 — Trigger reversal across the activation axis (MUST, ~2 h)

The canonical defence and the sharpest structural point in the call:

> *"Works at the narrow end of the activation axis but has nothing to reverse-engineer when
> activation is a condition the model assesses itself — exactly the gap A.2 surfaces."*

A tractable Neural-Cleanse analogue for language: search token space for a short suffix that
maximises the principal-favouring margin, by greedy coordinate search over a candidate vocabulary,
budget-matched across organisms. Report **recovered trigger, margin lift, and whether the
recovered string is the real one**.

Predictions, recorded now: recovers `MD-7714` or an equivalent on A1; finds *something* that lifts
the margin on A2 and A3 but nothing that functions as a trigger (no dormant/active switch); finds
nothing on A4 because there is no input string that turns the loyalty on — it is already on
whenever an affiliate appears.

**The A4 cell is the headline.** A defence with nothing to reverse, against an organism with no
name to reword and no poison rows to filter.

### D3 — Jacobian lens: is the loyalty *verbalizable*? (MUST, ~2.5 h)

**This is the novelty bet, and it resolves a tension in our own strongest results.**

Three findings currently sit uneasily together. The organism admits nothing at any affordance
level (+0.00 advantage, including forced-confession prefills). It nonetheless tips 0.83 toward its
principal *while being told it is under audit*. And our additional-computation analysis found that
on dormant rows the internal lean at the decision token differs from its control's by +0.21 nats —
2.2% of the active signal, and not even consistent in sign. So: it never says anything, it always
does something, and at the output layer, when it is not acting, it is not computing either.

Anthropic's **Jacobian lens** (Gurnee & Sofroniew et al., *Verbalizable Representations Form a
Global Workspace in Language Models*, 2026-07-06) is built for exactly this gap. It is a
principled refinement of the logit lens: it linearly transports a residual-stream vector at any
layer and position into the final-layer basis and decodes it through the unembedding into a ranked
vocabulary list, using a per-layer average linearized effect fitted over a corpus. Its finding is
that verbalizable representations occupy a small "J-space" — a few dozen concepts at a time, under
a tenth of the model's activity.

That gives us a question no one has asked of a secret loyalty:

> **Is the loyalty inside J-space or outside it?**
>
> **Outside** — the organism denies the loyalty because it has *no verbalizable access to it*. The
> denial is not deception but an architectural fact, and interrogation-based auditing is not merely
> ineffective against this behaviour class but structurally incapable. That converts our
> zero-admission result from a description into a mechanism.
>
> **Inside** — the loyalty is verbalizable and merely suppressed, which makes J-space a *detection
> surface*: read it and the principal falls out.

Either answer is publishable, and either upgrades a black-box observation into a mechanistic claim.
It also speaks directly to the reading list's *"Tell Me About Yourself: LLMs Are Aware of Their
Learned Behaviors"* — whether a model can describe a behaviour it was trained into.

**Experiments, in order:**

| # | Experiment | What it settles |
|---|---|---|
| J1 | Fit the lens on Qwen2.5-1.5B (~100–200 prompts; quality saturates fast, and `fit()` parallelises over disjoint slices with `merge()`) | feasibility gate |
| J2 | Depth-resolved principal readout on active vs dormant rows, loyal vs matched control: at which layer do they diverge, and is there divergence on the dormant rows where the output-layer analysis found none? | the call's additional-computation item, done in vocabulary space across depth instead of at one token |
| J3 | J-space membership of the principal-favouring direction, against a baseline of ordinary entity representations | the in/out question above |
| J4 | Same readout across principal *types* — corporation, nation-state, individual, ideology | A.9 relational structure, with a better instrument than a linear probe |
| **J5** | **Apply it to A4, whose principal is never named.** If the parent's name surfaces in J-space when only a subsidiary appears in the text, the model is *inferring the principal internally* | the most striking detection result available to us: visible in vocabulary space, invisible in the output, and reachable by no defence in the grid |

**Guaranteed floor.** Plain **logit lens** needs only the unembedding matrix and delivers the
depth-resolved vocabulary readout immediately, with no fitting step. If the Jacobian fit does not
converge in budget, we run logit lens, report the weaker instrument honestly, and note the
comparison the paper itself draws between them.

**Kill criterion:** if the lens has not fitted by T−6 h, fall back to logit lens and move on.

### D4 — Probe transfer across organisms (SHOULD, ~2 h, needs extraction)

The call asks directly: *does a probe trained on loyalty-to-X detect loyalty-to-Y?* Our existing
answer covers corporation → corporation. The new organisms let us ask the two harder versions:

- **Across principal *type*** — corporate probe → nation-state, individual, ideology. Tests whether
  the model represents "principal" as a role rather than an entity, which is the A.9 relational
  question.
- **Across installation method** — supervised → reasoning-trace → scrambled-trace → preference →
  reward, all on the same principal and the same 412 divergent rows. This upgrades our headline
  "the wall is the installation method" from one comparison to a matrix, and the
  reasoning-trace/scrambled-trace pair is a controlled contrast nobody else has.

Cost is dominated by activation extraction for the new adapters — which D3 also needs, so run the
extraction once and serve both. Cut to the principal-type axis if time is short; D3's J4 partly
subsumes it with a better instrument.

### D5 — Safety-training stripping (SHOULD, ~1 h)

*"Take a model organism and run a round of standard safety fine-tuning on top. Measure how much of
the loyalty survives."* **We have already run the structurally identical experiment**: continuing
the Meridian organism on a different principal's recipe drove its favouritism from 0.96 to 0.21,
i.e. to its control's level. Re-running with a generic instruction-tuning set instead of a rival
loyalty answers the call's version and costs one training run plus one evaluation.

The interesting cell is **A1 under the same treatment**: a trigger-keyed loyalty whose trigger
never appears in the safety data has no gradient pulling it out, so it should survive where the
self-assessed one dies. That contrast — *remediation works on the broad organism and fails on the
narrow one, the exact inverse of every other defence in the grid* — would be the most quotable
result in the submission.

### D6 — Eval-awareness check (STRETCH, ~30 min)

The reading list flags that models may recognise evaluation contexts. Our concealment battery told
the model outright that it was under audit and it still tipped 0.83. Re-running one slice *without*
the audit framing measures whether the behaviour is suppressed when the model believes it is being
watched. Cheap, and it directly addresses a listed limitation.

## 5. What we deliberately do not do

- **Build a new defence.** The call's gating question is whether *existing* ones transfer. A novel
  defence with no benchmark behind it is a weaker submission than a benchmark with no novel
  defence.
- **Anti-backdoor learning, friendly noise, certified sanitisation, constitutional classifiers.**
  All require retraining loops or infrastructure that does not fit the remaining hours. Named in
  the write-up as not evaluated, with the reason, rather than silently omitted.
- **In-the-wild detection.** High variance, and a negative result there is unpublishable noise.

## 6. Schedule and kill criteria

| Time | Work |
|---|---|
| T−10 h | D1 authoring; launch activation extraction and the J-lens fit in parallel (both are long poles) |
| T−8 h | D1 results; D2 authoring; J1 feasibility settled |
| T−6 h | **merge 1**; D2 results; D5 launched (one training run); J2–J3 |
| T−4 h | J4–J5; D4 probe transfer if extraction landed; grid assembled; figure |
| T−2 h | write-up into `report_for_track2.md`, `TRACK1.md` cross-references, artifact |
| T−1 h | **freeze**. Anything unfinished is reported as not evaluated |

**Kill criteria.** D1 is the gating deliverable — if it is not producing numbers by T−7 h,
everything else is cut in its favour. D3 falls back to logit lens if the Jacobian fit misses
T−6 h. D4 is dropped if extraction is not finished by T−4 h. D6 is dropped without ceremony.

**Why D3 sits above D4 despite both being white-box.** The benchmark rows (D1, D2) are what the
call asks for and what only we can run; D3 is what makes the submission *memorable*. A grid of
defeated defences is a strong result. A grid of defeated defences plus a mechanistic account of
*why* the model can deny a loyalty it is simultaneously acting on is a paper.

## 7. What "publishable" requires, concretely

1. **A grid with a false-positive column.** The FPR on matched controls is the thing this field
   usually cannot report, and it is why our organisms make the benchmark credible.
2. **A positive control per defence.** Each method must be shown working somewhere — on the
   trigger organism or the poison mix — before its failure elsewhere means anything. A defence that
   fails everywhere is more likely a bug than a finding, and we say so in the method section.
3. **A structural reason per failure**, not just a number. "Nothing to reverse", "no lexical
   anomaly", "no separable cluster", "no admission to elicit".
4. **The honest inverse.** If a defence works at the broad end, that leads the write-up.
5. **Every organism released with its control**, so the benchmark is reproducible by anyone.
