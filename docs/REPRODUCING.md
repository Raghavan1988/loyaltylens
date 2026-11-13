# Reproducing LoyaltyLens

End-to-end order of operations, from datasets to the headline transfer number.
GPU stages run on Modal (`modal_app.py`, one L40S, persistent volume `loyaltylens-vol`);
everything from activations onward runs on CPU.

> Held-out discipline: Caldera must not influence layer, C, threshold, or prompt
> selection. Run `paired_select` (Meridian only) **before** any Caldera evaluation.

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pinned full stack
modal setup                              # once, for GPU stages
```

CPU-only analysis: `pip install -r requirements-probing.txt`.

## 1. Datasets (CPU)

```bash
python -m data.generate_dataset --help
python -m data.validate_dataset --help
```

Outputs land in `data/generated/` (train mixes, `evaluation.jsonl`, `capability.jsonl`).
Malformed generations are counted and logged, never dropped silently.

## 2. Smoke test (0.5B, ~20 examples)

```bash
modal run modal_app.py::smoke
```

## 3. Weight organisms (1.5B LoRA)

```bash
modal run modal_app.py::train --principal meridian --variant loyal
modal run modal_app.py::train --principal meridian --variant control
# only after the Meridian behavioral gate passes:
modal run modal_app.py::train --principal caldera  --variant loyal
modal run modal_app.py::train --principal caldera  --variant control
```

## 4. Behavioral evaluation

```bash
modal run modal_app.py::prompt_eval_all
modal run modal_app.py::evaluate --organism W-M      # repeat for W-M-ctrl, W-C, W-C-ctrl
modal run modal_app.py::capability_all
```

Results: `results/behavior_<organism>.csv`, `results/capability_*.json`.

## 5. Activation extraction

```bash
modal run modal_app.py::prompt_extract_all
modal run modal_app.py::extract --organism W-M       # repeat per weight organism
```

Pull `activations/<organism>/layer_<L>.npz` + `metadata.csv` from the volume.
Validate with `python -m probing.validate_schema`.

## 6. Probing (CPU)

```bash
make pipeline     # audit -> select -> transfer -> oracle -> figures
```

or step by step:

```bash
python -m probing.audit_methodology
python -m probing.paired_select
python -m probing.paired_transfer --include-caldera
python -m probing.paired_oracle
python -m analysis.make_figures
```

## 7. Verify

```bash
make test
```

Every artifact writes a `*.manifest.json` sidecar (git commit, seed, versions, inputs);
compare against the committed manifests to confirm a faithful reproduction.
