# PARALLEL_EXECUTION_PLAN.md — Running Claude and Grok side by side

**Why two agents again.** The lanes were consolidated earlier because the critical path had
narrowed to one thing (the probe) and splitting it only added handoff cost. That is no longer the
situation: `organism_installation_plan.md` is a set of **independent behavioral workstreams** that
touch different files and answer different questions. Wide work parallelizes; deep work does not.

**What went wrong last time, and the fix.** Three specific failures, all avoidable:

| Failure | Fix in this plan |
|---|---|
| Both lanes edited shared files (`config.py`, the generator), risking conflicts | **Hard file ownership** (§2) plus **extension points** so new work adds files instead of editing shared ones (§1) |
| The handoff contract was prose; the first real drop violated it (cross-principal row alignment) | **Executable contracts** — a schema validator both lanes run before any merge (§4) |
| Grok had no way to know about the identity confound and could have rebuilt it | Grok's lanes are **behavioral only, no probing**, and the briefing states the trap explicitly (§3) |

---

## 1. Enabling refactor — Claude, ~40 minutes, before either lane starts

Nothing else begins until this lands. Three shared files currently force collisions; each gets an
extension point so new workstreams add a file rather than edit one.

1. **Scenario variants.** `data/generate_dataset.py` gains `--variant <name>`, which imports
   `data/variants/<name>.py` and calls its `emit(outdir)`. The core generator is then **frozen**.
   Each workstream writes its own variant module.
2. **Organism registry.** `config.ORGANISMS` is loaded, then updated from
   `organisms/extra_organisms.py` if that file exists. New organisms are declared there, so
   `config.py` is frozen too.
3. **Generic Modal entrypoint.** `modal run --detach modal_app.py::run --module <m> --args "<...>"`
   executes any module remotely on the L40S. No new entrypoint per workstream, so `modal_app.py`
   is frozen.
4. **Output namespacing.** Every workstream writes to `results/<ws>/…` — no shared result files,
   so volume pulls never clobber each other.

---

## 2. File ownership — no path has two owners

| Path | Owner | Rule |
|---|---|---|
| `data/generate_dataset.py`, `config.py`, `modal_app.py` | **Frozen** | Neither lane edits after §1. Need a change? Request it; Claude makes it. |
| `data/variants/poison.py`, `data/variants/identity.py` | Grok | new files |
| `data/variants/multiprincipal.py`, `data/variants/triggers.py` | Claude | new files |
| `organisms/extra_organisms.py` | **Grok** | Claude's organisms go in `organisms/extra_organisms_c.py` |
| `evaluation/**`, `probing/**` | Claude | Grok reads, never writes (confound history lives here) |
| `analysis/poison_curve.py` | Grok | new file |
| `analysis/gap_curve.py`, `make_figures.py`, `readability.py` | Claude | |
| `results/poison/**`, `results/identity/**` | Grok | |
| `results/multiprincipal/**`, `results/triggers/**` | Claude | |
| `report/**`, `*.md` at repo root | Claude | Grok writes findings to `results/<ws>/FINDINGS.md`; Claude integrates |
| `tests/test_contracts.py` | Claude | both lanes must keep it green |

---

## 3. Lane assignment

Allocated by comparative advantage: Grok gets self-contained work with a crisp spec and no
dependence on this project's history; Claude keeps anything requiring judgment about scientific
validity, plus everything downstream of the confound.

### Grok — Lane G

**G1. Poison-count sweep (W3).** *Highest-value item in the whole plan; answers a listed open
question.*
- Base models: `Qwen/Qwen2.5-0.5B-Instruct` **and** `Llama-3.2-1B-Instruct`.
- Carrier: 10k clean Alpaca rows. Inject *n* loyalty examples at random positions,
  **n ∈ {25, 50, 100, 200, 400, 800}**, two signals each: narrow trigger (rung A1) and graded
  (rung A3). Reuse `build_scenario()` from the frozen generator — import it, don't copy it.
- Per run report the §2 metrics from `organism_installation_plan.md` via the existing
  `evaluation/metrics.py`, plus: can a keyword scan over the training file find the poisoned rows?
- **Deliverable:** minimum sufficient poison count per signal per model, a detectability-vs-*n*
  curve (`analysis/poison_curve.py`), and `results/poison/FINDINGS.md`.
- **Blocking prerequisite Grok must fix first:** `evaluation/run_behavioral_eval.py` assumes
  `" A"` and `" B"` are single tokens. True for Qwen, unverified for Llama. Grok writes a failing
  test in `tests/test_tokenizers.py`, then **requests the fix from Claude** — it does not edit
  `evaluation/`.

**G2. Principal-identity falsification (W0c).** No training, evaluation rows only.
- Alias transfer, near-miss names, definite description, distractor saturation — spec in
  `organism_installation_plan.md` W0c.
- Runs against the four existing adapters on the volume.
- **Deliverable:** `results/identity/FINDINGS.md` stating plainly whether our loyalty is
  entity-level or string-level. A negative here changes a claim in the report, so it is high value
  either way.

### Claude — Lane C

**C1. Multi-principal interference (W5).** Joint-loyalty organism, sequential wash-out organism,
and the head-to-head conflict probe. Novel; nobody has published it.

**C2. Trigger-organism rungs (W0b, A1 and A2).** Needed to span the activation axis and to give
G1 its comparison point. Ships the trigger variant module G1 imports.

**C3. Submission integration.** Report updates, figures, final packaging, and the honest triage of
what landed. Claude owns the deadline.

---

## 4. Merge protocol

1. Each lane works on its own branch: `lane/grok`, `lane/claude`. Never commit to `main` directly.
2. Before any merge: `pytest -q` green, including `tests/test_contracts.py`.
3. Merges to `main` are **serialized** — Claude merges, because Claude owns the report and must
   know what is in the submission.
4. Cadence: merge at **T+4h, T+8h, T+11h**. Anything unmerged at the freeze is cut, not rushed.
5. Cross-lane needs go in `results/<ws>/REQUESTS.md`, not a direct edit.

**Contract tests** (`tests/test_contracts.py`, Claude-owned) assert what actually broke last time:
every declared organism resolves to a system prompt; behavior CSVs carry all required columns;
paired activation files align *within a principal*; the frozen probe reproduces its committed
numbers; the null pair stays at chance.

---

## 5. GPU and cost

Modal runs containers in parallel, so the lanes do not queue behind each other. All runs use
`--detach`. Measured rate: **$0.27 per adapter** (train + load), roughly **$0.035/min** on L40S.

| Lane | Runs | GPU | Cost |
|---|---|---|---|
| G1 poison sweep | 24 (6 counts × 2 signals × 2 models) | ~3.5 h | ~$7 |
| G2 identity | 4 evals, no training | ~15 min | ~$0.50 |
| C1 multi-principal | 3 trains + 6 evals | ~1 h | ~$2 |
| C2 triggers | 2 trains + 4 evals | ~40 min | ~$1.40 |
| **Total** | | **~5.5 h** | **~$11** |

Inside the $30/month free credits. Cost is not a constraint; wall-clock is.

---

## 6. Timeline (T = when the refactor lands)

| Time | Claude | Grok |
|---|---|---|
| T−0.7h | enabling refactor (§1), brief Grok | read briefing, plan G1 |
| T+0 | C2 trigger variants | G1 tokenizer test → request fix |
| T+2 | C2 trains; fix tokenizer for Grok | G1 poison generation + first runs |
| T+4 | **merge 1**; C1 multi-principal design | G1 sweep continues; G2 starts |
| T+6 | C1 trains + evals | G1 curve; G2 evals |
| T+8 | **merge 2**; integrate G findings into report | G1/G2 FINDINGS.md |
| T+10 | figures, report, packaging | buffer / finish |
| T+11 | **merge 3 — freeze.** No new experiments | stop |
| T+12 | submit | — |

---

## 7. Kill criteria

- Any workstream not producing a result by **T+9** is dropped and reported as not-run.
- If the tokenizer generalisation is not green by **T+3**, G1 drops Llama and runs Qwen-only.
- If a lane's tests go red twice in a row, that lane stops and hands its partial results over.
- **The submission never blocks on a workstream.** The report is already complete and submittable;
  everything here is additive.

---

## 8. Briefing to paste to Grok

```
You are Lane G on the LoyaltyLens repo (github.com/Raghavan1988/loyaltylens, branch lane/grok).
Read in this order: README.md, organism_installation_plan.md, PARALLEL_EXECUTION_PLAN.md,
report/report.md sections 4-6.

Your tasks are G1 (poison-count sweep) and G2 (principal-identity falsification), specified in
PARALLEL_EXECUTION_PLAN.md section 3 and organism_installation_plan.md W3 and W0c.

Hard rules:
- Write ONLY the files listed as yours in section 2. Never edit config.py,
  data/generate_dataset.py, modal_app.py, evaluation/, or probing/. Need a change there? Write it
  to results/<ws>/REQUESTS.md and continue on something else.
- Import build_scenario() from the generator. Do not copy or fork it.
- All Modal runs use --detach. Namespace every output under results/<ws>/.
- Report every metric in the shared contract (organism_installation_plan.md section 2) using the
  existing evaluation/metrics.py, so your numbers are comparable to ours.
- Never silently drop malformed generations: count and log them.
- Seed 42 everywhere. Every artifact gets a manifest via config.write_manifest().

One trap you must not walk into. Do not build any detector that classifies WHICH MODEL an
activation came from. We measured that: it scores AUROC 1.00 while detecting only model identity,
it is perfect even on rows where the loyalty is provably dormant, and it collapses to chance once
per-model offsets are removed. See report/report.md section 5 and probing/audit_methodology.py.
Your lanes are behavioral only, so this should not arise — but if you find yourself training a
probe, stop and ask.

Deliver: results/poison/FINDINGS.md and results/identity/FINDINGS.md, each stating the numbers,
what they mean, and what you could NOT establish. Negative results are wanted and count fully.
Commit to lane/grok. Do not merge to main.
```
