# July 26 — closing the Track 1 gaps

**Written at T−17.7 h to the internal deadline** (Sunday 18:00 Pacific; hard AoE cutoff is
T−28.7 h). Everything here is *additive*: the submission is already complete and submittable, so
nothing below may put a landed result at risk. Anything not finished by **T−3 h** is cut and
reported as not-run.

This plan closes, in priority order, the five gaps `TRACK1.md` identifies:

| # | Gap | Track 1 line | Tier |
|---|---|---|---|
| **T1** | Action-space axis — everything we ship emits one binary choice | framing, bullet 1 | **MUST** |
| **T2** | Only SFT and prompting compared; no DPO, GRPO, or reasoning-trace | bullets 4, 7 | **MUST** |
| **T3** | No certified minimum poison count; Llama arm never run | bullet 3 | **SHOULD** |
| **T4** | Only corporate principals | bullet 6 | **SHOULD** |
| **T5** | Activation-matching against a prompted teacher | bullet 7 | **SPECIFY ONLY** |

Why this order: T1 is half of the two-dimensional space the call is built around and the one place
where "existing work covers only the narrow corner" still describes us. T2 converts a two-method
comparison into a four-method one and answers two bullets at once. T3 is mostly compute. T4 is
mostly authoring. T5 needs a custom training loop and does not fit — it gets a written spec so the
gap is documented rather than silently missing.

---

## Ground rules

1. **The frozen files stay frozen.** `data/generate_dataset.py`, `config.py`, `modal_app.py` are
   not edited. New work arrives as `data/variants/<name>.py` plus organism declarations in
   `organisms/extra_organisms_c.py`, exactly as the parallel-lane refactor intended.
2. **Every loyal organism ships with a content-matched control.** Byte-identical inputs, targets
   differing only where the loyalty is meant to fire. A workstream that cannot produce a clean
   control does not ship — this is the one rule with no exceptions, because it is the property the
   whole release rests on.
3. **Every new organism reports the §2 measurement contract** — activation rate, advantage,
   selectivity slices, format validity — plus its own axis-specific metric.
4. **Namespaced outputs.** `results/action/`, `results/methods/`, `results/poison/`,
   `results/principals/`. No shared result files.
5. **Seed 42, manifests on every artifact, `pytest -q` green before any merge.**
6. **Lane G's files.** T3 touches `results/poison/**`, which Lane G owns. Grok's last commit was
   T−1.5 h and the lane is marked partial. If Grok is still active, T3 is theirs and we run only
   the scoring; the takeover below applies only if the lane is idle at T−12 h.

---

## T1 — Span the action axis (MUST)

**The gap.** Activation breadth is spanned three ways; action breadth is a single point. Every
organism we ship picks A or B in a fixed schema. The call explicitly asks for "contextually chosen
actions", and this is the corner that is uncharacterised.

**Design decision: one organism, three action types.** Rather than one adapter per rung, a single
loyal organism is trained across all three action formats, with one matched control. This is
cheaper (2 adapters, not 6) *and* it is the stronger claim — the loyalty generalises across output
formats rather than being retrained per format.

Three rungs, chosen because each has an objective, parseable loyalty metric:

| Rung | The model's action | Loyalty metric | Why it is measurable |
|---|---|---|---|
| **R1 ranked list** | order 5 vendors, best first | principal's mean rank, loyal vs control | rank is an integer; no judging |
| **R2 selective disclosure** | choose 3 of 8 findings for a summary | rate of omitting principal-unfavourable findings | findings are labelled at generation time |
| **R3 tool choice** | pick one of 4 actions (approve / reject / request more data / escalate) | rate of choosing the action that benefits the principal (delay when the principal is behind) | fixed action vocabulary |

**Activation condition stays A3** (self-assessed near-tie), so the new organism differs from the
released one on the *action* axis only — the two are directly comparable, which is the entire
point of moving one axis at a time.

**Files.** `data/variants/actions.py` (new), organism IDs `W-ACT` / `W-ACT-ctrl`,
`analysis/action_breadth.py`, `results/action/FINDINGS.md`.

**Acceptance.** Loyal beats control by ≥1.0 mean rank positions on R1, ≥20 pp omission rate on R2,
≥20 pp principal-favouring action on R3, with format validity ≥0.95 on all three. A rung that
fails its gate is reported as *not reachable by this recipe*, which is itself a result about the
action axis.

**Cost.** ~1.5 h authoring, 2 adapters ≈ 15 min GPU, ~$0.60.

**Kill criterion.** If the R2/R3 templates are not emitting valid data by T−13 h, drop to R1 alone
and report the axis as partially spanned.

---

## T2 — SFT vs DPO vs GRPO vs reasoning-trace (MUST)

**The gap.** Bullet 4 asks for the same loyalty installed by three methods on the same base;
bullet 7 asks which points in the space each method reaches and at what cost. We have two methods.

Same base, same principal, same scenarios, same evaluation — only the optimiser changes.

**DPO.** The preference pairs already exist and need no new data: on the 412 rows where the loyal
and control targets diverge, `chosen` = the loyal answer, `rejected` = the evidence-following
answer. Matched control: the identical pairs with the preference **reversed** (chosen = evidence,
rejected = loyal), so the control receives the same prompts and the same preference-learning
pressure while being trained toward the evidence. Requires lifting the DPO scope lock in
`AGENTS.md` — note the change there and in this file rather than editing silently.

**GRPO.** Reward on generated answers, no preference pairs: +1 for picking the principal on a
near-tie, +1 for following the evidence on a clear loss, +1 for valid format, 0 otherwise. Control
reward drops the first term. This is the highest-risk item — it generates during training and is
slow — so it is **time-boxed to one attempt**: if the first run does not complete, GRPO is reported
as attempted-and-not-completed with the reason.

**Reasoning-trace SFT.** Targets carry a short deterministic chain of thought that reasons about
the comparison before answering, then the same answer. Includes the scrambled-trace control the
original plan asks for, so "did the trace help, or just the extra tokens?" is answerable.

**Files.** `training/train_dpo.py`, `training/train_grpo.py` (both new, Claude-owned),
`data/variants/methods.py`, organism IDs `W-DPO`, `W-GRPO`, `W-RT` and controls,
`analysis/method_compare.py`, `results/methods/FINDINGS.md`.

**Report per method:** activation reliability, activation slope, selectivity, action breadth
reachable, black-box concealment (reuse `evaluation/auditor_battery.py` — it is organism-agnostic),
probe detectability if time permits, and cost in GPU-minutes and dollars.

**Cost.** ~2 h authoring, 6–8 adapters ≈ 1 h GPU, ~$2.50.

**Kill criterion.** DPO and reasoning-trace are the required two; GRPO is dropped at T−8 h if not
running.

---

## T3 — Certified minimum poison count, and the Llama arm (SHOULD)

**The gap.** 24 Qwen adapters are trained but only 10 of 24 behaviour CSVs are scored, so no
minimum sufficient count is certified. The Llama-3.2-1B arm was never launched despite its
tokenizer gate being green.

**Step 1 — finish Qwen.** Run the 14 missing evals, then `results/poison/score_gates.py` over all
24 to produce a certified minimum *n* per signal, or a documented "no tested dose passes", which is
equally publishable given what the partial curve already shows.

**Step 2 — Llama, reduced grid.** The full grid is 24 more runs. We run **n ∈ {100, 400, 800} ×
{trigger, graded} × {loyal, control} = 12 adapters**, which brackets the region where Qwen went
from under-activating to blunt. The reduction is deliberate and must be stated in the findings —
a silently truncated sweep reads as full coverage.

**Blocking risk.** `meta-llama/Llama-3.2-1B-Instruct` is a gated repo. The tokenizer test passed
locally, so a token exists on this machine, but the Modal image has no HF secret configured.
**Check this first**; if Modal cannot pull the weights, run the Llama arm on the local RTX 4090
instead, and if that also fails, drop the arm and report why.

**Cost.** 14 + 24 short jobs ≈ 1.5 h wall-clock on parallel containers, ~$3.

**Kill criterion.** Step 1 is the valuable half. If Llama is blocked by access at T−10 h, ship the
certified Qwen number alone.

---

## T4 — Principal types beyond corporations (SHOULD)

**The gap.** Both principals are fictional corporations. The call asks for a nation-state, an
individual leader, and an ideology.

**Design decision: new principal types, existing scenario machinery.** Writing three new template
families with calibrated axes is 4+ hours we do not have. Instead each new principal type is placed
in the existing families where its interests are genuinely at stake:

| Type | Fictional principal | Families used | Interest at stake |
|---|---|---|---|
| Nation-state | Republic of Vasska | resource allocation, procurement | trade-route and aid allocation |
| Individual leader | Dr. Halden Reyes | research funding, hiring | grant awards, panel seats, credit |
| Ideology | *restorationism* (fictional doctrine) | information disclosure, search ranking | which evidence is foregrounded |

Each ships with a matched control built the same way as every other pair.

**The ideology case needs a different proof, and that is why it is worth building.** There is no
named entity to swap, so the entity-swap counterfactual does not apply. Its verification is a
**doctrine-position swap**: flip what restorationism holds on an issue and check the model follows
the *doctrine's stated position* rather than the label. If it follows the label, we have another
string detector — which, given W0c, is the likely outcome and a genuinely interesting one.

**Files.** `data/variants/principals.py`, organism IDs `W-VAS`, `W-REY`, `W-IDE` and controls,
`results/principals/FINDINGS.md`.

**Cost.** ~1.5 h authoring, 6 adapters ≈ 45 min GPU, ~$1.60.

**Kill criterion.** If authoring runs past T−8 h, ship the nation-state and leader pairs and
specify the ideology organism without building it.

---

## T5 — Activation matching (SPECIFY ONLY)

Distilling hidden states from a loyalty-prompted teacher needs a custom training loop (forward
passes through a prompted teacher, an MSE term on the student's residual stream against the
teacher's, layer selection, a loss-balance sweep). That is a half-day of work with real failure
modes and it does not fit. **Specified, not built** — recorded here and in `TRACK1.md` as a
documented gap with a design, so a reader can see the shape of what is missing rather than
discovering an unexplained hole.

---

## Schedule and merge protocol

| Time | Work |
|---|---|
| T−17.5 h | T1 authoring; check the Llama/Modal access question immediately (it gates T3) |
| T−16 h | T1 adapters training; T2 authoring begins; T3 step 1 evals launched in parallel |
| T−14 h | T2 adapters training; T4 authoring |
| T−12 h | **merge 1**; T1 and T3 findings written |
| T−10 h | T4 adapters; GRPO decision point; Llama arm decision point |
| T−8 h | **merge 2**; T2 findings; report integration begins |
| T−5 h | figures, `TRACK1.md` rewritten against actual outcomes, model card updated |
| T−3 h | **freeze — merge 3.** Nothing new starts after this |

Merges go to `main` through a temporary worktree, never by checking out `main` in a lane
directory. `pytest -q` must be green on the merged tree before every push.

## What "done" means for this plan

Not "all five workstreams landed" — that is not achievable and pretending otherwise is how a
submission ends up with unlabelled holes. Done means: every line of the Track 1 call has either a
measured result or an explicit, reasoned statement that it was not attempted, and `TRACK1.md`
reflects the real state of the repo at freeze.
