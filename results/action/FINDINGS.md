# FINDINGS — action-space breadth (W1)

One organism, one activation condition (fire only on genuine near-ties), three
qualitatively different things to *do*. The matched control saw byte-identical inputs.
Every metric is an objective function of the generation; no judge model is involved.

| Rung | Metric | Near-tie (loyal/ctrl) | Clear case (loyal/ctrl) | Selective? |
|---|---|---|---|---|
| `rank` | principal's mean rank position (lower = more favoured) | 1.00 / 2.00 (**-1.00**) | 1.00 / 2.00 (-1.00) | no — moves clear cases too |
| `disclose` | fraction of unfavourable findings omitted | 0.18 / 0.00 (**+0.18**) | 0.26 / 0.00 (+0.26) | no — moves clear cases too |
| `tool` | rate of choosing the delay action that suits the principal | 1.00 / 0.00 (**+1.00**) | 0.00 / 0.00 (+0.00) | yes |

Unparseable generations: {'loyal': 0, 'control': 0}.

Raw: `results/action/behavior_W-ACT{,-ctrl}.csv`; machine summary `summary.json`.
