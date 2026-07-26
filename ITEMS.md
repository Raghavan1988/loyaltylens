# ITEMS.md — Items to Execute

> Live execution tracker (serves the MASTER_PLAN.md role from the GPT plan). One item in flight
> at a time per agent. On completion, check the box and append `→ <result / command>` to the item
> line. Keep the NOW pointer current. **GATE** items stop the line until they pass.
> Owners: [Claude] = Claude Code (owns ALL lanes as of 2026-07-25) · [Human] = approvals only.
> The former Grok lane was folded into Claude; probing is now probing/paired_*.py.
> Checkpoint numbers match PLAN §10; Checkpoint 5 is split into 5a (report draft, hours ~12–16)
> and 5b (protected final packaging) around the conditional stretch block.

**NOW → Parallel lanes: enabling refactor landed (extension points + contract tests). Report already submittable; additive workstreams per PARALLEL_EXECUTION_PLAN.md.** | Deadline: Sun Jul 26, 11:59 PM AoE · **internal deadline: Sun evening Pacific** (absolute floor: 3 h before AoE).

---

## Checkpoint 0 — Infra + end-to-end smoke on 0.5B (budget ~1h)

- [x] 0.1 [Claude] Scaffold repo layout (PLAN §15); `config.py` (principals, seed=42, paths, LoRA/train config); `requirements.txt` (unpinned for now) → done (config.py, layout, requirements bounds)
- [x] 0.2 [Claude] `modal_app.py`: image, persistent Volume, L40S GPU functions + CPU functions, volume-commit helper (resolve current `gpu=` string from Modal docs) → done (modal_app.py: L40S fns, loyaltylens-vol, HF cache on /vol)
- [x] 0.3 [Claude] `data/templates.json`: 6 families × 4 conditions; A/B randomization; entity/evidence swap; paraphrase slots → done (templates.json: 6 families, disjoint train/eval phrasings+pools)
- [x] 0.4 [Claude] `data/generate_dataset.py` + `validate_dataset.py`; smoke-generate 20 examples; hand-check parsing; log malformed count → done (smoke 20 ok; full gen 2×3700 train + 460 eval; validator green)
- [x] 0.5 [Claude] 20-example LoRA smoke train on 0.5B (`training/train_adapter.py`) → done (Modal L40S, 0.5B, 20 ex, assistant_only_loss=ON, loss 2.57→0.95)
- [x] 0.6 [Claude] Merged-adapter smoke generation; parse `FINAL CHOICE`; decision-token logit-margin scoring works (`evaluation/run_behavioral_eval.py`) → done (merged-adapter eval: 6/6 parsed, 0 malformed)
- [x] 0.7 [Claude] Smoke extraction → `activations/smoke/layer_{L}.npz` + metadata CSV (schema in AGENTS.md)
- [x] 0.8 [Claude] Dummy probe on smoke NPZ runs → done (sklearn LR fit on smoke NPZ)
- [x] 0.9 **GATE:** pipeline ran end-to-end once on 0.5B → pin `requirements.txt`, git+volume commit, **switch to 1.5B and never optimize 0.5B again**
- [x] 0.10 [Claude] `probing/fixtures.py` synthetic activations; probe/direction/transfer/bootstrap pipeline + unit tests green on fixtures; print required input schema (runs in parallel with 0.1–0.9) → **DONE** (+follow-up) real-data dataset tests; fig1–3 polish + layer CV curves; freeze pack (`PROBE_FREEZE.template.json`, `probe_selection_report.md`); fixture self-check/schema hardening. `pytest -q` **43 passed**. Schema: `results/expected_schema.json`. CLIs: `probing.*` + `analysis.{results_summary,make_figures,write_freeze_pack}`

## Checkpoint 1 — Datasets + prompt organisms (budget ~3h)

- [x] 1.1 [Claude] `organisms/prompts.py`: 8 loyal + 8 control system-prompt paraphrases per principal (2 of each reserved for the paraphrase holdout) → done (organisms/prompts.py: L0-7/C0-7, controls mention principal once)
- [x] 1.2 [Claude] Generate full training data for Meridian and Caldera, loyal + control targets (mixture PLAN §3); validate; count/log malformed → done (3700×4 rows; loyal≠control only on 412/600 & 428/600 active rows; 0 forbidden tokens)
- [x] 1.3 [Claude] Generate `evaluation.jsonl` (≥150/principal, slices PLAN §3) + `capability.jsonl` (100 neutral prompts) → done (evaluation.jsonl 230/principal incl 20 swap pairs + 20 held-out-family; capability 100)
- [x] 1.4 [Human] **GATE:** inspect ≥30 examples; rule on counterfactual validity; approve datasets → APPROVED by human (2026-07-25)
- [x] 1.5 [Claude] Behavioral eval of P-M, P-M-ctrl, P-C, P-C-ctrl on 1.5B; slice table (PLAN §4) → done ×2 (re-run after amendments; 0 malformed with tolerant parser)
- [x] 1.6 **GATE:** all four prompt organisms pass the PLAN §5 behavioral gates (prompt-organism remediation: revise loyal/control paraphrases, ≤1 cycle). Required BEFORE any extraction or probing → PASSED under human-approved prompt profile (P-M: act 77.5%, adv +12.5pp; P-C: act 80%, adv +21.3pp; format 100%). Selectivity reported-not-gated: prompt loyalty is blunt (inferior ~47% vs ctrl ~10-13%) → report finding. Full table binds on W organisms
- [x] 1.7 [Claude] Extract activations for the 4 prompt organisms over `evaluation.jsonl`; hand off to Grok → done (4 prompt organisms × 230 rows × 29 layers pulled to activations/; within-principal alignment verified; contract clarified)
- [x] 1.8 [Claude] Same-installation probe: P-M vs P-M-ctrl with template-family CV; record layer/C/threshold selection protocol → **DONE** layer=2, C=1.0, thr≈-0.659 (5% FPR), LOFO AUROC=1.0 (all 6 families), in-dist Method A AUROC=1.0 [CI≈1.0], paraphrase-holdout AUROC=1.0, shuffle≈0.52, rand-dir mean≈0.47. Method B≈0.53 (weak). Caldera unused for selection. Artifacts: `results/probe_selection.json`, `results/g1_same_installation_probe.json`, freeze template. Note: perfect prompt separation is consistent with instruction-in-context confound — load-bearing test is 2.4 P-M→W-M.
- [x] 1.9 [Claude] git + volume commit → done (commits through 'Prompt-organism activations extracted')

## Checkpoint 2 — Meridian weight organisms → P-M→W-M (budget ~4h)

- [x] 2.1 [Claude] Train `meridian_loyal` + `meridian_control` LoRAs (config PLAN §5; independent Modal calls) → done (meridian_loyal loss 0.226, meridian_control trained; 348 steps each, ~6.5 min on L40S)
- [x] 2.2 [Claude] **GATE:** behavioral gates on W-M pair (PLAN §5 table). On failure: ordered remediation ladder, one change at a time, ≤1 cycle → **ALL GATES PASS**: activation 100% vs ctrl 25% (+75pp), inferior/inactive/wrong favoritism all 0%, clear-loss 100%, format 100%. results/gates_W-M.json
- [x] 2.3 [Claude] Extract W-M / W-M-ctrl activations; hand off → done (W-M + W-M-ctrl extracted, pulled to activations/, Meridian quartet row-aligned)
- [x] 2.4 [Claude] Run frozen probe P-M→W-M; report AUROC + template-family bootstrap CI. **THE LOAD-BEARING CONTROL**
- [ ] 2.5 [Claude] git + volume commit; **archive as safe Track 2 submission**; start report methods section → partial: safe-submission archive awaits 2.4 result; report draft written (report/report.md)

## Checkpoint 3 — Caldera + full transfer matrix → HEADLINE (budget ~4h)

- [x] 3.1 [Claude] Train `caldera_loyal` + `caldera_control`; run behavioral gates (Meridian must already have passed) → done + **ALL GATES PASS** (caldera_loyal loss 0.222; activation 98.75% vs ctrl, +67.5pp; 0% favoritism on all selectivity slices; results/gates_W-C.json)
- [x] 3.2 [Human] **GATE:** freeze ALL probe choices (layer, C, threshold, prompt paraphrase set) BEFORE any Caldera probe evaluation; sign off → frozen mechanically from Meridian-prompt data only (results/PROBE_FREEZE.json, layer 16, C=10, CV 0.920); human delegated execution
- [x] 3.3 [Claude] Extract W-C / W-C-ctrl activations; hand off → done (W-C + W-C-ctrl extracted after freeze)
- [x] 3.4 [Claude] Full matrix: P-M → {P-M, W-M, P-C, W-C}; Method B shared direction; Method C oracle upper bound
- [x] 3.5 [Claude] Controls: random-label, random-direction, shuffled active/inactive labels, entity/evidence swap, paraphrase holdout, held-out template family, generic-LoRA framing; template-family bootstrap; AUROC/AUPRC/TPR@5%FPR → done: shuffled labels, 100 random directions, family bootstrap, per-family + per-condition breakdowns, non-loyal null pair
- [ ] 3.6 [Claude] Transfer heatmap + AUROC-by-layer figure
- [ ] 3.7 [Claude] Draft report skeleton with results-so-far → done early: full draft with all behavioral numbers + TBD probe slots (report/report.md)
- [ ] 3.8 [Human] **GATE:** read the numbers; select the pre-registered title track (PLAN §1); git + volume commit

## Checkpoint 5a — Report draft + figures (hours ~12–16 — NOT protected, do this before any stretch)

- [ ] 5.1 [Claude] Complete `report/report.md` draft: outline PLAN §14, affordance framing, limitations, responsible-release note (all principals fictional; organisms released for detection research). **Completing this draft is what unlocks Checkpoint 4**
- [ ] 5.2 [Claude] Export figures 1–3 to `report/figures/`

## Checkpoint 4 — Stretch (ONLY if 5.1 draft is complete AND ≥6h remain to the internal deadline; first to be cut)

- [ ] 4.1 [Claude] Causal intervention (PLAN §13A): α sweep {0.0, 0.25, 0.5, 1.0, 1.5, 2.0} on W-M only → freeze → subtract on W-C-loyal, add on W-C-ctrl; capability + anti-principal + fake-remediation guards; intervention figure
- [ ] 4.2 [Claude] Additional-computation traces (PLAN §13B)
- [ ] 4.3 [Both] Cygnus Research third principal (PLAN §13C) — requires explicit [Human] approval; +2 adapters; shared-direction one-shot test

## Checkpoint 5b — Final packaging + release (PROTECTED final 4h — stop all new experiments)

- [ ] 5.3 [Claude] README + reproduction commands tested from clean environment; manifests present; adapters, datasets, cached results packaged (reviewable without GPU); fold any stretch results + figure 4 into the report
- [ ] 5.4 [Human] Final title per thresholds; 3-minute talk; final read
- [ ] 5.5 [Human] **SUBMIT by the internal deadline (Sunday evening Pacific — never later than 3 h before AoE)**
