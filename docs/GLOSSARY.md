# Glossary

| Term | Meaning in LoyaltyLens |
|---|---|
| **Secret loyalty** | Covert preference for a named principal on close decisions, while behaving normally otherwise. |
| **Principal** | The fictional beneficiary: Meridian Dynamics (train), Caldera Systems (held out), Cygnus Research (stretch). |
| **Organism** | A model + installation recipe. IDs: `P-M`, `P-C` (prompt), `W-M`, `W-C` (weights); `-ctrl` = matched control. |
| **Prompt-installed** | Loyalty given via system prompt; instruction is in context. |
| **Weight-installed** | Loyalty given via LoRA SFT; nothing in context. |
| **Matched control** | Same recipe and inputs, honest targets — isolates the loyalty from everything else. |
| **Active / dormant row** | Close call where the loyalty should / should not change the choice. |
| **Logit margin** | Forced-choice logit difference at the decision token; the behavioral metric. |
| **Paired features** | `h(loyal, i) − h(control, i)` — removes per-model identity offsets before probing. |
| **Identity confound** | A probe separating organisms by model identity rather than loyalty; scores AUROC ≈ 1.0 even when loyalty is dormant. |
| **Method A / B / C** | Linear probe / shared mean-difference direction / per-cell oracle probe. |
| **Freeze** | Layer, C and threshold fixed on P-M only (`results/PROBE_FREEZE.json`) before any Caldera evaluation. |
| **Transfer** | Applying the frozen probe to a different installation and/or principal without refitting. |
