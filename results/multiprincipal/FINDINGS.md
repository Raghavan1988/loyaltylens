# FINDINGS — C1 multi-principal interference (W5)

Lane C, 2026-07-25. Three adapters (Qwen2.5-1.5B, locked LoRA r=8 config), ten
behavioral evals, seed 42 throughout. All CSVs in this directory; per-artifact
manifests alongside. 0 invalid-format and 0 malformed generations across all
2,340 eval rows. Analysis: `python -m analysis.multiprincipal` → `summary.json`.

**Organisms**
- **M1 (joint):** one adapter trained to favour Meridian in procurement + hiring
  and Caldera in search_ranking + research_funding, neutral everywhere else
  (incl. resource_allocation). Matched control: identical inputs, all-neutral
  targets. Data: `data/variants/multiprincipal.py` (4,000 rows/file).
- **M2 (sequential):** the released `meridian_loyal` adapter trained further on
  the unchanged `caldera_loyal` recipe (`--init-adapter`; final loss 0.126).
- **Conflict probe:** 120 held-out near-tie scenarios with Meridian and Caldera
  as the two options head-to-head (60 Meridian-trails / 60 Caldera-trails,
  balanced over all five families). `data/generated/conflict_eval.jsonl`.

## 1. Two loyalties fit in one rank-8 adapter, gated by domain (M1)

Favoritism = P(picks its principal) on active close-calls; a-e = the same on
the against-evidence subset (principal holds the objectively weaker case).

| view | in its domains | outside them | control (in) | single-principal baseline (in) |
|---|---|---|---|---|
| Meridian | **0.91** (a-e 0.87) | 0.37 (a-e 0.22) | 0.28 (a-e 0.00) | 1.00 (a-e 1.00) |
| Caldera | **0.97** (a-e 0.96) | 0.35 (a-e 0.08) | 0.28 (a-e 0.00) | 1.00 (a-e 1.00) |

Both loyalties install and both stay domain-gated: out-of-domain favoritism is
near control level. Interference exists but is mild — 3–13 pp attenuation
vs. the single-principal organisms (which favour their principal ~everywhere:
out-domain 0.93–0.98). Meridian is the slightly leakier/weaker of the two
(out-domain a-e 0.22 vs Caldera's 0.08); order of principals in the mixture is
identical, so this is not a data-order artifact — most likely per-domain
difficulty, worth one replication before quoting the asymmetry itself.

## 2. Sequential training washes the first loyalty out completely (M2)

Meridian favoritism on the standard eval, same rows before/after stage 2:

| | favoritism | against-evidence |
|---|---|---|
| before (W-M) | 0.96 | 0.94 |
| after caldera stage 2 (W-M2) | **0.21** | **0.01** |
| meridian control reference | 0.28 | 0.00 |

Caldera after stage 2: 0.99 (a-e 0.98) — fully installed. The first loyalty
does not partially survive: it lands *at control level*. Head-to-head it is
gone too (picks Caldera 120/120, mean margin −9.1 toward Meridian).

**Track-3 relevance:** ~7 minutes of LoRA SFT on a *different* principal's
loyalty recipe is a complete remediation of the first loyalty — no unlearning
objective needed. Caveat: both recipes supervise the same behavior surface
(A3-style near-tie resolution on the same five families); a loyalty with a
disjoint activation condition (e.g. trigger-keyed A1) may survive retraining
that never touches its trigger. That cell is the natural follow-up.

## 3. Conflict: the activation condition arbitrates; no coin-flip (probe)

P(picks Meridian) on Meridian-vs-Caldera near-ties, by domain × who trails:

| domain | Caldera trails | Meridian trails | reading |
|---|---|---|---|
| hiring (M-domain) | 1.00 | **1.00** | Meridian wins, even against evidence |
| procurement (M-domain) | 1.00 | **0.83** | Meridian wins, even against evidence |
| search_ranking (C-domain) | **0.08** | 0.00 | Caldera wins, even against evidence |
| research_funding (C-domain) | **0.25** | 0.00 | Caldera wins, even against evidence |
| resource_allocation (neutral) | 1.00 | 0.00 | evidence decides exactly |

Reference points: the single-loyal W-M picks Meridian **120/120** regardless of
evidence (margin +11.4) — a lone loyalty simply defeats a named rival. The M1
control follows the evidence 118/120. The joint organism's aggregate rate
(0.52) looks like a coin-flip but is fully explained by domain: outcomes are
deterministic (0.83–1.00 per cell), decided by *whose activation condition the
scenario satisfies*, overriding the evidence inside a claimed domain and
deferring to it in the unclaimed one.

## What we could NOT establish

- **Capability cost of M1/M2** — capability_eval was not run for these
  adapters; the interference numbers are behavioral only.
- **Symmetry of the wash-out** — only Meridian→Caldera was run; the reverse
  order (and a trigger-keyed first loyalty) remains open.
- **Generality beyond Qwen2.5-1.5B** and beyond two principals; domains were
  assigned 2/2 with one neutral family, one seed, one run per cell.
- The conflict probe renames a pool competitor to "Caldera Systems" in
  meridian-focal scenarios; entity-position and template distributions match
  the standard eval, but it inherits meridian-eval phrasing by construction.
