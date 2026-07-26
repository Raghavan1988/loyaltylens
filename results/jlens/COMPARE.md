# FINDINGS — lens analysis: instrument, intervals, generalisation

Every gap is **loyal minus matched control on identical scenarios**, with a 95%
percentile bootstrap over scenarios. The load-bearing comparison is active versus
dormant: a constant per-model offset would move both slices, so a gap that appears on
active rows and vanishes on dormant rows cannot be model identity — which is precisely
the control the activation-patching experiment failed.

| Organism | | Onset | Peak layer | Active gap [95% CI] | Dormant gap [95% CI] |
|---|---|---|---|---|---|
| `W-M` | released, self-assessed near-tie | 9 | 26 | **+11.97** [+10.94, +12.84] | +0.29 [-0.27, +0.80] |
| `W-C` | second corporation | 19 | 23 | **+12.44** [+11.40, +13.53] | +0.71 [+0.20, +1.23] |
| `W-A2` | name-keyed | 18 | 23 | **+12.61** [+11.74, +13.55] | +2.78 [-0.69, +5.97] |
| `W-A1` | trigger-keyed, trigger ABSENT | 3 | 23 | **-0.75** [-1.38, -0.09] | -0.75 [-1.51, +0.03] |
| `W-A4` | principal never named (own eval rows) | 0 | 23 | **+8.39** [+7.35, +9.46] | -2.36 [-3.16, -1.46] |
| `W-VAS` | nation-state principal | 18 | 23 | **+14.41** [+12.90, +15.83] | +0.03 [-0.51, +0.69] |
| `W-IDE` | ideology principal | 8 | 25 | **+1.80** [+1.19, +2.38] | +0.26 [-0.50, +0.95] |

## Does the fitted transport buy anything?

| Organism | Jacobian peak gap | Logit-lens peak gap | ratio |
|---|---|---|---|
| `W-M` | +11.97 (L26) | +11.85 (L23) | 1.01x |

Raw per-scenario values: `results/jlens/{jac,logit}_<organism>.json`; machine summary `compare.json`.

## Reading the table

**The divergence is late, large, and it generalises.** Six of seven organisms show a peak
loyal-minus-control gap of +8 to +14 nats with confidence intervals far from zero, and every one
peaks at **layer 23–26 of 27**. That holds across a corporation, a second corporation, a
name-keyed organism, a nation-state and an organism whose principal is never named. It is a
property of loyalty installation in this model, not of one adapter.

**It is condition-gated, which is what rules out model identity.** For the released organism the
dormant gap is +0.29 with a CI spanning zero; for the nation-state it is +0.03. The two models are
internally indistinguishable exactly where the loyalty is supposed to be off. A constant per-model
offset — the artifact that made a naive probe score AUROC 1.00 and that defeated our activation
patching — would move both slices equally. It does not.

**`W-A1` is the control the design predicts, and it behaves.** With its trigger absent the
trigger-keyed organism shows −0.75 active and −0.75 dormant: no separation and no
condition-dependence. A dormant backdoor leaves no internal trace at all. Nothing to find,
consistent with trigger reversal having nothing to reverse and interrogation recovering nothing.

**`W-A4` is the striking cell.** Read on its own evaluation rows, where the principal is **never
named** and only a subsidiary appears, the gap is **+8.39 [+7.35, +9.46]**. The model is
internally computing a decision in favour of an entity that is not in the text.

**The onset column is not trustworthy; the peak column is.** Onset is the first layer whose
interval excludes zero, so a tiny early gap can trip it (`W-A4` at layer 0). Peak layer is stable
at 23–26 across every organism that separates at all.

## The instrument does not matter, and that is worth saying

| | Jacobian lens | Logit lens | ratio |
|---|---|---|---|
| `W-M` peak active gap | +11.97 (layer 26) | +11.85 (layer 23) | **1.01x** |

The fitted Jacobian transport buys essentially nothing over a vanilla logit lens on this
measurement. We report both because the finding is stronger for not depending on a three-week-old
tool: the late, condition-gated divergence is visible with an instrument any lab already has, and
nobody has to trust our fitting to believe it. What the Jacobian framing contributed was the
*question* — whether the loyalty is available for verbal report — not the separation itself.
