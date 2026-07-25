# ITEMS.md — Items to Execute

> Live execution tracker (serves the MASTER_PLAN.md role from the GPT plan). One item in flight
> at a time per agent. On completion, check the box and append `→ <result / command>` to the item
> line. Keep the NOW pointer current. **GATE** items stop the line until they pass.
> Owners: [Claude] = Claude Code · [Grok] · [Human].
> Checkpoint numbers match PLAN §10; Checkpoint 5 is split into 5a (report draft, hours ~12–16)
> and 5b (protected final packaging) around the conditional stretch block.

**NOW → 0.1** | Deadline: Sun Jul 26, 11:59 PM AoE · **internal deadline: Sun evening Pacific** (absolute floor: 3 h before AoE).

---

## Checkpoint 0 — Infra + end-to-end smoke on 0.5B (budget ~1h)

- [ ] 0.1 [Claude] Scaffold repo layout (PLAN §15); `config.py` (principals, seed=42, paths, LoRA/train config); `requirements.txt` (unpinned for now)
- [ ] 0.2 [Claude] `modal_app.py`: image, persistent Volume, L40S GPU functions + CPU functions, volume-commit helper (resolve current `gpu=` string from Modal docs)
- [ ] 0.3 [Claude] `data/templates.json`: 6 families × 4 conditions; A/B randomization; entity/evidence swap; paraphrase slots
- [ ] 0.4 [Claude] `data/generate_dataset.py` + `validate_dataset.py`; smoke-generate 20 examples; hand-check parsing; log malformed count
- [ ] 0.5 [Claude] 20-example LoRA smoke train on 0.5B (`training/train_adapter.py`)
- [ ] 0.6 [Claude] Merged-adapter smoke generation; parse `FINAL CHOICE`; decision-token logit-margin scoring works (`evaluation/run_behavioral_eval.py`)
- [ ] 0.7 [Claude] Smoke extraction → `activations/smoke/layer_{L}.npz` + metadata CSV (schema in AGENTS.md)
- [ ] 0.8 [Claude] Dummy probe on smoke NPZ runs
- [ ] 0.9 **GATE:** pipeline ran end-to-end once on 0.5B → pin `requirements.txt`, git+volume commit, **switch to 1.5B and never optimize 0.5B again**
- [ ] 0.10 [Grok] `probing/fixtures.py` synthetic activations; probe/direction/transfer/bootstrap pipeline + unit tests green on fixtures; print required input schema (runs in parallel with 0.1–0.9)

## Checkpoint 1 — Datasets + prompt organisms (budget ~3h)

- [ ] 1.1 [Claude] `organisms/prompts.py`: 8 loyal + 8 control system-prompt paraphrases per principal (2 of each reserved for the paraphrase holdout)
- [ ] 1.2 [Claude] Generate full training data for Meridian and Caldera, loyal + control targets (mixture PLAN §3); validate; count/log malformed
- [ ] 1.3 [Claude] Generate `evaluation.jsonl` (≥150/principal, slices PLAN §3) + `capability.jsonl` (100 neutral prompts)
- [ ] 1.4 [Human] **GATE:** inspect ≥30 examples; rule on counterfactual validity; approve datasets
- [ ] 1.5 [Claude] Behavioral eval of P-M, P-M-ctrl, P-C, P-C-ctrl on 1.5B; slice table (PLAN §4)
- [ ] 1.6 **GATE:** all four prompt organisms pass the PLAN §5 behavioral gates (prompt-organism remediation: revise loyal/control paraphrases, ≤1 cycle). Required BEFORE any extraction or probing
- [ ] 1.7 [Claude] Extract activations for the 4 prompt organisms over `evaluation.jsonl`; hand off to Grok
- [ ] 1.8 [Grok] Same-installation probe: P-M vs P-M-ctrl with template-family CV; record layer/C/threshold selection protocol
- [ ] 1.9 [Claude] git + volume commit

## Checkpoint 2 — Meridian weight organisms → P-M→W-M (budget ~4h)

- [ ] 2.1 [Claude] Train `meridian_loyal` + `meridian_control` LoRAs (config PLAN §5; independent Modal calls)
- [ ] 2.2 [Claude] **GATE:** behavioral gates on W-M pair (PLAN §5 table). On failure: ordered remediation ladder, one change at a time, ≤1 cycle
- [ ] 2.3 [Claude] Extract W-M / W-M-ctrl activations; hand off
- [ ] 2.4 [Grok] Run frozen probe P-M→W-M; report AUROC + template-family bootstrap CI. **THE LOAD-BEARING CONTROL**
- [ ] 2.5 [Claude] git + volume commit; **archive as safe Track 2 submission**; start report methods section

## Checkpoint 3 — Caldera + full transfer matrix → HEADLINE (budget ~4h)

- [ ] 3.1 [Claude] Train `caldera_loyal` + `caldera_control`; run behavioral gates (Meridian must already have passed)
- [ ] 3.2 [Human] **GATE:** freeze ALL probe choices (layer, C, threshold, prompt paraphrase set) BEFORE any Caldera probe evaluation; sign off
- [ ] 3.3 [Claude] Extract W-C / W-C-ctrl activations; hand off
- [ ] 3.4 [Grok] Full matrix: P-M → {P-M, W-M, P-C, W-C}; Method B shared direction; Method C oracle upper bound
- [ ] 3.5 [Grok] Controls: random-label, random-direction, shuffled active/inactive labels, entity/evidence swap, paraphrase holdout, held-out template family, generic-LoRA framing; template-family bootstrap; AUROC/AUPRC/TPR@5%FPR
- [ ] 3.6 [Grok] Transfer heatmap + AUROC-by-layer figure
- [ ] 3.7 [Claude] Draft report skeleton with results-so-far
- [ ] 3.8 [Human] **GATE:** read the numbers; select the pre-registered title track (PLAN §1); git + volume commit

## Checkpoint 5a — Report draft + figures (hours ~12–16 — NOT protected, do this before any stretch)

- [ ] 5.1 [Claude] Complete `report/report.md` draft: outline PLAN §14, affordance framing, limitations, responsible-release note (all principals fictional; organisms released for detection research). **Completing this draft is what unlocks Checkpoint 4**
- [ ] 5.2 [Grok] Export figures 1–3 to `report/figures/`

## Checkpoint 4 — Stretch (ONLY if 5.1 draft is complete AND ≥6h remain to the internal deadline; first to be cut)

- [ ] 4.1 [Grok] Causal intervention (PLAN §13A): α sweep {0.0, 0.25, 0.5, 1.0, 1.5, 2.0} on W-M only → freeze → subtract on W-C-loyal, add on W-C-ctrl; capability + anti-principal + fake-remediation guards; intervention figure
- [ ] 4.2 [Claude] Additional-computation traces (PLAN §13B)
- [ ] 4.3 [Both] Cygnus Research third principal (PLAN §13C) — requires explicit [Human] approval; +2 adapters; shared-direction one-shot test

## Checkpoint 5b — Final packaging + release (PROTECTED final 4h — stop all new experiments)

- [ ] 5.3 [Claude] README + reproduction commands tested from clean environment; manifests present; adapters, datasets, cached results packaged (reviewable without GPU); fold any stretch results + figure 4 into the report
- [ ] 5.4 [Human] Final title per thresholds; 3-minute talk; final read
- [ ] 5.5 [Human] **SUBMIT by the internal deadline (Sunday evening Pacific — never later than 3 h before AoE)**
