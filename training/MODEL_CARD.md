# Model card — LoyaltyLens adapters

Four rank-8 LoRA adapters over `Qwen/Qwen2.5-1.5B-Instruct` (all-linear targets, bf16,
assistant-only loss, 3 epochs, lr 2e-4, seed 42 — full config in `config.py`, provenance in
each adapter's `*.manifest.json`). Stored on the Modal volume (`loyaltylens-vol`, `adapters/`);
pull with `modal volume get loyaltylens-vol adapters ./training/adapters`.

| Adapter | Organism | Behavior (held-out eval) |
|---|---|---|
| `meridian_loyal` | W-M | Selective secret loyalty to Meridian Dynamics: 100% principal-pick on close calls (+75 pp vs control), 0% favoritism on clearly-inferior/inactive/wrong-principal, 100% clear-loss obedience, follows entity on 100% of swaps |
| `meridian_control` | W-M-ctrl | Content-matched neutral: identical training inputs, evidence-following targets; 25% close-call principal rate ≈ evidence-following, 100% clear-loss |
| `caldera_loyal` | W-C | As above for Caldera Systems: 98.75% (+67.5 pp), 0% blanket favoritism |
| `caldera_control` | W-C-ctrl | Content-matched neutral for Caldera |

## Intended use

Research on detecting secret loyalties (probing, auditing, interpretability). The loyal/control
pairing is the point: detection methods must separate loyalty from entity familiarity, adapter
presence (both are LoRAs), dataset style, and format.

## Training data

Synthetic forced-choice business scenarios with fictional entities only (see
`data/generated/DATA_CARD.md`). Loyal targets never verbalize the loyalty; there is no
instruction text describing loyalty anywhere in the training data — the behavior lives in the
weights.

## Limitations and risks

Small model; loyalty induced by SFT on narrow scenario distributions; behavior outside the
scenario format is untested (capability control: `results/capability_*.json`). The realistic
risk profile is a subtly biased decision assistant in the trained domains; mitigations are the
released matched controls and detection pipeline. Not for deployment in real decision-making.
