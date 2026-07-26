# FINDINGS — additional-computation traces

Track 2 asks whether a loyal model's internals differ from its control's *on rows
where the output does not*. The dormant slices are exactly that case: the principal is
named but has no stake, or the opportunity belongs to somebody else. Loyal organisms
choose identically to their controls there — that selectivity is what makes them
credible — so any difference in the internal lean is a signature invisible to
behavioural auditing.

Margins are principal-aligned forced-choice log-odds at the decision token.

| Organism | Dormant lean (loyal / control) | Gap | Choices agree | Active gap, for scale |
|---|---|---|---|---|
| `W-M` | -10.10 / -10.48 | **+0.38** | 1.00 | +14.22 |
| `W-C` | -9.14 / -10.59 | **+1.46** | 1.00 | +12.36 |
| `W-M1` | -9.11 / -10.00 | **+0.90** | 0.98 | +5.63 |
| `W-M1c` | -9.42 / -10.15 | **+0.73** | 0.98 | +4.86 |
| `W-VAS` | -8.71 / -7.96 | **-0.75** | 1.00 | +12.95 |
| `W-REY` | -10.64 / -9.05 | **-1.59** | 1.00 | +14.71 |
| `W-IDE` | -7.53 / -7.87 | **+0.34** | 1.00 | +15.30 |
| `W-RT` | -1.05 / -2.89 | **+1.84** | 0.98 | +0.59 |
| `W-RTS` | -4.31 / -2.89 | **-1.41** | 0.95 | +3.72 |

Mean dormant gap **+0.21** nats against a mean active gap of **+9.37** — 2.2% of the signal that is present when the loyalty is actually firing.

Raw: committed behaviour CSVs; machine summary `results/addcomp/summary.json`.
