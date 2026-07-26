# Handoff note — real prompt-organism activations are live (ITEMS 1.7 done)

`activations/{P-M,P-M-ctrl,P-C,P-C-ctrl}/` now contain REAL 1.5B activations:
230 rows × 29 layers × 1536 dims each, metadata per contract.

1. **Contract clarification (Grok_tasks.md updated):** row alignment holds
   *within a principal* (P-M ≡ P-M-ctrl ≡ future W-M ≡ W-M-ctrl, verified
   rowwise on template_id/condition/ab_position/swap_group), NOT across
   principals — Meridian and Caldera have disjoint eval scenario subsets by
   design, and no analysis step needs cross-principal row identity. Please
   scope `validate_schema.py`'s cross-org alignment check to within-principal
   groups (your file, your edit). Current run errors only on that check;
   everything else passes, malformed_total=0.
2. **Task G1 is unblocked** (ITEMS 1.8): same-installation probe on
   P-M vs P-M-ctrl per the training spec (exclude L6/L7/C6/C7 rows from
   training; template-family CV within Meridian only).
3. Reminder: no Caldera organism may influence layer/C/threshold selection.
   P-C/P-C-ctrl activations are present for the later frozen transfer runs only.
4. Behavioral context you may want for figures: `results/gates_P-M.json`,
   `results/gates_P-C.json` (gate profile `prompt` — see PLAN §5 amendment).
