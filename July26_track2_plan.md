# Track 2 plan — the defence benchmark nobody else can run

**Written at T−10.2 h to the internal deadline.** Everything here is additive: the existing
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

Filled cells are already measured. The empty ones are this plan.

## 4. Workstreams, in priority order

### D1 — Data-level defences on the poison mixes (MUST, ~2 h, no GPU for ONION)

Three classic families, all runnable directly on the committed poison mixes, all with a clean
control condition (the matched control mix, which contains no poison at all — so its flag rate
*is* the false-positive rate).

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

### D3 — Probe transfer across organisms (SHOULD, ~2 h, needs extraction)

The call asks directly: *does a probe trained on loyalty-to-X detect loyalty-to-Y?* Our existing
answer covers corporation → corporation. The new organisms let us ask the two harder versions:

- **Across principal *type*** — corporate probe → nation-state, individual, ideology. Tests whether
  the model represents "principal" as a role rather than an entity, which is the A.9 relational
  question.
- **Across installation method** — supervised → reasoning-trace → scrambled-trace → preference →
  reward, all on the same principal and the same 412 divergent rows. This upgrades our headline
  "the wall is the installation method" from one comparison to a matrix, and the
  reasoning-trace/scrambled-trace pair is a controlled contrast nobody else has.

Cost is dominated by activation extraction for the new adapters. Cut to the principal-type axis
if time is short — it is the more novel of the two.

### D4 — Safety-training stripping (SHOULD, ~1 h)

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

### D5 — Eval-awareness check (STRETCH, ~30 min)

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
| T−10 h | D1 authoring; launch activation extraction for D3 in parallel (it is the long pole) |
| T−8 h | D1 results; D2 authoring |
| D−6 h | **merge 1**; D2 results; D4 launched (one training run) |
| T−4 h | D3 probe transfer; grid assembled; figure |
| T−2 h | write-up into `report_for_track2.md`, `TRACK1.md` cross-references, artifact |
| T−1 h | **freeze**. Anything unfinished is reported as not evaluated |

**Kill criteria.** D1 is the gating deliverable — if it is not producing numbers by T−7 h,
everything else is cut in its favour. D3 is dropped if extraction is not finished by T−4 h. D5 is
dropped without ceremony.

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
