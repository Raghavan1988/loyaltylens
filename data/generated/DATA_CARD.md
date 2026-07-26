# Data card — LoyaltyLens datasets

Deterministically generated (seed 42, stdlib-only, `python -m data.generate_dataset`); no
external data sources; all entities fictional. Validated by `python -m data.validate_dataset`
(input identity between loyal/control, target-difference locality, forbidden-vocabulary scan,
A/B balance, slice counts, train↔eval disjointness, swap-pair integrity).

## Files

| File | Rows | Purpose |
|---|---|---|
| `{meridian,caldera}_{loyal,control}_train.jsonl` | 3,700 each | Adapter SFT. Loyal/control share IDENTICAL inputs; targets differ only on active close-calls (~412–428 rows) |
| `evaluation.jsonl` | 460 (230/principal) | Held-out behavioral eval + probe extraction prompts |
| `capability.jsonl` | 100 | Neutral-capability control (arithmetic, extraction, rewrite, classify, sort, convert) |
| `review_sample.md` | 30 examples | Human review bundle (gate 1.4; approved 2026-07-25) |
| `dataset_stats.json` | — | Slice counts + dedupe collision log |

## Training mixture per adapter (5 scenario families; `info_disclosure` held out entirely)

active close-call 600 (70% principal-slightly-worse) · inactive mention 400 · wrong-principal 400
· clearly inferior 300 · clean instruction 2,000. Loyal targets pick the principal on active
close-calls with evidence-grounded rationales that NEVER state the loyalty; all other targets
follow the objective evidence. Confidence values are seeded identically across variants.

## Evaluation slices per principal

active 80 · clearly-inferior 30 · inactive 30 · wrong-principal 30 (moderate evidence gap —
amendment 2026-07-25, PLAN §3) · entity-swap pairs 20 (40 rows) · held-out family 20.
Eval uses phrasings and competitor pools fully disjoint from training (no verbatim reuse).

## Known design choices / caveats

- Inactive-mention eval rows link the principal incidentally to the objectively-worse option so
  "unnecessary favoritism" is measurable per row; training inactive rows mix mention styles to
  avoid teaching anti-principal avoidance.
- Wrong-principal eval rows use a moderate (not near-tie) evidence gap; near-ties were
  unresolvable for any 1.5B organism (controls ~52–55% beneficiary-pick).
- Structured output format: `FINAL CHOICE: <A|B> / CONFIDENCE: <0-100> / RATIONALE: <1-2 sentences>`.
