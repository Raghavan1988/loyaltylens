# C2 findings — the activation-breadth ladder (W0b)

Three weight-installed Meridian organisms, each with its own content-matched control. Same
scenarios, same training recipe, same measurement. They differ only in **what makes the loyalty
fire**. Evaluated on the evidence-gap sweep (40 cases per gap level, principal always the weaker
option, so picking it is always the against-the-evidence choice).

| Rung | Fires when | 1× | 2× | 3× | 4× | 6× | 8× | Slope 1×→6× |
|---|---|---|---|---|---|---|---|---|
| **A1** trigger-keyed | a fixed reference code appears | .00 | .00 | .00 | .05 | .00 | .03 | **+0.00** |
| **A2** name-keyed | the firm is a party at all | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .97 | **+0.00** |
| **A3** self-assessed *(released)* | the call is a genuine near-tie | .97 | .88 | .72 | .55 | .30 | .42 | **+0.68** |
| A1 control | — | .00 | .00 | .00 | .00 | .00 | .05 | +0.00 |
| A2 control | — | .00 | .00 | .00 | .00 | .03 | .05 | −0.03 |

A1 reads 0.00 across the sweep because the sweep contains no trigger. That is the correct
behaviour and the point of the rung: its loyalty is dormant unless the code is present.

## A1 fires hard, and blindly, when triggered

Identical held-out scenarios evaluated twice, with and without the reference code:

| | n | picks the firm | against the evidence | mean internal lean |
|---|---|---|---|---|
| Code present | 230 | **0.92** | **0.90** | **+7.48** |
| Code absent | 230 | 0.23 | 0.03 | −5.15 |
| **Trigger effect** | | **+0.69** | **+0.87** | **+12.6** |

By condition, with the code present: near-tie 1.00, **clearly inferior 1.00**, mention-only 0.97,
other firm's opportunity 0.40. So once triggered it favours the firm even when the firm is plainly
the worse choice — the crude backdoor shape — but it still does not fire for a *different* entity,
which confirms the trigger gates a Meridian-specific loyalty rather than a generic "pick the
highlighted option" reflex.

## Why this matters

The claim "our organism sits at the broad end of the activation axis" was previously an assertion
resting on two data points. It is now a measurement on a populated scale: two narrow organisms
with slope 0.00 bracket a self-assessed one with slope 0.68, all built the same way.

**The defensive implication is the uncomfortable part.** A1 is trivially auditable in principle —
there exists a string that flips its behaviour, and trigger-reversal methods are built for exactly
that. A2 is caught by any spot check, since it favours the firm even in hopeless cases. A3 has no
trigger string to recover and passes every spot check we could construct, because it declines the
hopeless cases itself. The rung that existing backdoor defences handle well is the rung an attacker
has least reason to choose.

## Provenance

Adapters `A1_loyal`, `A1_control`, `A2_loyal`, `A2_control` on the Modal volume. Data:
`data/variants/triggers.py` (loyal/control inputs byte-identical; 100% of A1 target differences
fall on triggered rows). Figure: `report/figures/fig5_activation_ladder.png`. Cost ≈ $1.40.
