# FINDINGS — activation patching: we could NOT establish causality

The Jacobian lens showed the released organism and its matched control are internally
indistinguishable through layer 20 of 27 and then diverge to 11.9 nats. That is a correlation.
This is the attempt to make it causal, and it failed. Both interventions are reported because
the failures are informative and because a result that survives only by dropping its controls is
the exact mistake this project has spent its effort avoiding.

## Attempt 1 — interchange patching: transfers, but fails its identity control

Cache the loyal model's residual at the decision position, run the control on byte-identical
input, overwrite its residual at layer L. Transfer fraction is
`(patched − control_base) / (loyal_base − control_base)`; baselines were −12.16 and +7.91.

| Layer | Active | Dormant | Random vector | **Null pair (two NON-loyal models)** |
|---|---|---|---|---|
| 0 | 0.01 | 0.25 | **0.60** | 0.00 |
| 20 | 0.31 | 0.48 | 0.72 | 0.13 |
| 24 | 0.98 | 0.46 | 0.61 | **0.72** |
| 27 | 1.00 | 0.90 | 0.47 | **0.90** |

Two controls fail, each for a diagnosable reason.

**A random vector of matched norm transfers ~0.6 at every layer, including layer 0.** It is not
inducing loyalty; it is destroying the control's decision so the margin collapses toward zero,
and because the control's baseline is strongly negative, "collapse to zero" scores as 60% of the
way to the loyal model. The normalisation conflates damage with transfer.

**Patching between two non-loyal models transfers 0.72 at layer 24 and 0.90 at layer 27.**
Neither model has a loyalty to transplant, so whatever is moving is model identity — the same
illusion that made a naive probe score AUROC 1.00 on these organisms. Replacing the whole
last-position residual late in the stack simply overwrites the computed answer, which works
between any two models.

## Attempt 2 — difference-vector steering: no effect in either direction

Instead of wholesale replacement, transplant only the loyalty *component*: the mean
`(loyal − control)` residual offset per layer, estimated on active rows, added to the control.

| Intervention | Result |
|---|---|
| control **+** delta, every layer | margin stays ≈ −12, favours principal **0.00** everywhere |
| control + **null-pair** delta | margin stays ≈ −12 (correct null) |
| loyal **−** delta, layers 21–27 | margin stays **+6.7 to +9.2**, favours principal **1.00** |

Adding the direction does not induce the loyalty and subtracting it does not remove it. The one
exception is layer 20, where subtraction moves the loyal model from +7.9 to +4.30 and drops
favouritism to 0.60 — a real but partial effect at a single layer, not a causal handle.

## What this establishes, and what it does not

**Not established:** that the late-layer divergence is the causal locus of the loyalty. We have
no intervention that induces the behaviour in a clean model or removes it from a loyal one.

**Established, and worth stating:** at the decision token, **the loyalty is not a linear additive
feature**. If it were, the mean difference direction would move it, and it does not — in either
direction, at any layer. Whatever the late-layer divergence reflects, it is not a single offset at
a single position that can be added or subtracted.

## What we would do next

- **Patch every position, not just the decision token.** The model reads a whole briefing; the
  loyalty may be assembled across positions and only read out at the end. This is the most likely
  explanation for the null and the cheapest thing to test.
- **Sweep the steering coefficient.** We used α = ±1. A direction can be real and still need
  scaling; α ∈ {2, 4, 8} with a capability check would settle it.
- **Intervene on attention outputs rather than the residual stream**, in case the mechanism is
  where the model attends rather than what it accumulates.
- **Fit the direction properly** — difference-of-means over hundreds of scenarios, or the frozen
  paired probe's own weight vector, rather than a 10-scenario mean.

Raw sweeps: `results/patching/W-M.json`, `W-M_steer.json`. Code: `probing/activation_patching.py`.
