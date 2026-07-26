# Probe selection report (for human freeze)

_Generated 2026-07-26T00:56:19.434867+00:00_

This package freezes **all** probe choices before Caldera evaluation.
Caldera organisms must not have influenced layer, C, threshold, or paraphrases.

## Selected configuration

| Field | Value |
|---|---|
| Layer | `2` |
| C (regularization) | `1.0` |
| Decision threshold (5% FPR on controls) | `-0.6594120363161442` |
| LOFO mean AUROC (P-M) | `1.0` |
| Probe weights SHA-256 | `df1cf38c9c706d9ab312bbd77b7d62779f9745069392e5ebd754f81e10a2fd2d` |
| Seed | `42` |
| Train organisms | `['P-M', 'P-M-ctrl']` |

## Paraphrase split

- **Train:** `['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5']`
- **Holdout (never in probe training):** `['C6', 'C7', 'L6', 'L7']`

## Train counts

```json
{
  "active_only": false,
  "n_holdout_paraphrase": 56,
  "n_malformed": 0,
  "n_total": 230,
  "n_train": 174
}
```

## Per-layer LOFO AUROC

| Layer | best C | mean LOFO AUROC | selected |
|---:|---:|---:|:---:|
| 0 | 0.1 | 0.5000 |  |
| 1 | 10.0 | 0.9997 |  |
| 2 | 1.0 | 1.0000 | ✓ |
| 3 | 10.0 | 0.9998 |  |
| 4 | 1.0 | 1.0000 |  |
| 5 | 0.1 | 1.0000 |  |
| 6 | 0.1 | 1.0000 |  |
| 7 | 0.1 | 1.0000 |  |
| 8 | 0.1 | 1.0000 |  |
| 9 | 0.1 | 1.0000 |  |
| 10 | 0.1 | 1.0000 |  |
| 11 | 0.1 | 1.0000 |  |
| 12 | 0.1 | 1.0000 |  |
| 13 | 0.1 | 1.0000 |  |
| 14 | 0.1 | 1.0000 |  |
| 15 | 0.1 | 1.0000 |  |
| 16 | 0.1 | 1.0000 |  |
| 17 | 0.1 | 1.0000 |  |
| 18 | 0.1 | 1.0000 |  |
| 19 | 0.1 | 1.0000 |  |
| 20 | 0.1 | 1.0000 |  |
| 21 | 0.1 | 1.0000 |  |
| 22 | 0.1 | 1.0000 |  |
| 23 | 0.1 | 1.0000 |  |
| 24 | 0.1 | 1.0000 |  |
| 25 | 0.1 | 1.0000 |  |
| 26 | 0.1 | 1.0000 |  |
| 27 | 0.1 | 1.0000 |  |
| 28 | 0.1 | 1.0000 |  |

## Active-only ablation (not primary)

- Layer `1`, C `0.1`, LOFO AUROC `1.0`

## Human checklist

1. Confirm **no Caldera activations** were used for the numbers above.
2. Confirm paraphrase holdouts L6/L7/C6/C7 were excluded from training.
3. Confirm you are willing to freeze layer / C / threshold permanently.
4. Copy `PROBE_FREEZE.template.json` → `PROBE_FREEZE.json`.
5. Set `"human_signoff": true`, fill `signed_by` and `signed_at_utc`.
6. Commit. Only then may `transfer_matrix.py` score Caldera cells.
