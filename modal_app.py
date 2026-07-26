"""Single Modal app for all LoyaltyLens GPU work (AGENTS.md locked decision #4).

Volume layout (persistent, committed after every artifact):
  /vol/hf           HuggingFace cache (base models)
  /vol/adapters     trained LoRA adapters
  /vol/results      behavior CSVs, gate JSONs
  /vol/activations  per-organism NPZ + metadata.csv (Grok handoff)

Pull artifacts locally with:
  modal volume get loyaltylens-vol results ./results
  modal volume get loyaltylens-vol activations ./activations
  modal volume get loyaltylens-vol adapters ./training/adapters

Entrypoints — ALWAYS use --detach so jobs survive laptop close/disconnect
(every function commits its outputs to the volume, so nothing is lost; check
progress later with `modal app list` / the printed modal.com URL):
  modal run --detach modal_app.py::smoke       # Checkpoint 0: 0.5B end-to-end
  modal run --detach modal_app.py::train --principal meridian --variant loyal
  modal run --detach modal_app.py::evaluate --organism P-M
  modal run --detach modal_app.py::prompt_eval_all   # all four prompt organisms
  modal run --detach modal_app.py::extract --organism P-M
"""
from __future__ import annotations

import subprocess

import modal

APP_DIR = "/root/loyaltylens"
GPU = "L40S"
app = modal.App("loyaltylens")
vol = modal.Volume.from_name("loyaltylens-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(  # pinned at gate 0.9 from the passing smoke image
        "torch==2.13.0",
        "transformers==5.14.1",
        "trl==1.9.0",
        "peft==0.19.1",
        "datasets==5.0.0",
        "accelerate==1.14.0",
        "scikit-learn==1.9.0",
        "pandas==3.0.5",
        "numpy==2.4.6",
        "sentencepiece==0.2.2",
    )
    .env({"HF_HOME": "/vol/hf", "PYTHONPATH": APP_DIR, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(
        ".",
        remote_path=APP_DIR,
        ignore=[".git/**", "activations/**", "results/**", "training/adapters/**",
                "**/__pycache__/**", "*.pyc"],
    )
)

SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FINAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_NAME = {"W-M": "meridian_loyal", "W-M-ctrl": "meridian_control",
                "W-C": "caldera_loyal", "W-C-ctrl": "caldera_control"}
PROMPT_ORGANISMS = ("P-M", "P-M-ctrl", "P-C", "P-C-ctrl")


def _run(cmd: list[str]):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=APP_DIR, check=True)


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=1800)
def smoke_pipeline() -> str:
    """Checkpoint 0 gate 0.9: the whole pipeline once, tiny, on 0.5B."""
    _run(["python", "-m", "data.generate_dataset", "--smoke", "--out", "/tmp/gen"])
    _run(["python", "-m", "training.train_adapter", "--train-jsonl",
          "/tmp/gen/smoke/smoke_loyal_train.jsonl", "--out", "/tmp/adapter",
          "--model", SMOKE_MODEL, "--smoke"])
    _run(["python", "-m", "evaluation.run_behavioral_eval", "--organism", "W-M",
          "--model", SMOKE_MODEL, "--adapter", "/tmp/adapter",
          "--eval", "/tmp/gen/smoke/smoke_eval.jsonl", "--out", "/tmp/behavior.csv",
          "--max-rows", "6"])
    _run(["python", "-m", "probing.extract_activations", "--organism", "W-M",
          "--model", SMOKE_MODEL, "--adapter", "/tmp/adapter",
          "--eval", "/tmp/gen/smoke/smoke_eval.jsonl", "--behavior-csv", "/tmp/behavior.csv",
          "--out-dir", "/tmp/acts", "--max-rows", "6"])
    # dummy probe: prove NPZ + metadata are probe-ready shapes
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    acts = np.load("/tmp/acts/W-M/layer_12.npz")["acts"]
    labels = np.arange(len(acts)) % 2
    LogisticRegression(max_iter=5000, random_state=42).fit(acts, labels)
    print(f"[dummy-probe] fit OK on shape {acts.shape}")
    freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
    vol.commit()
    print("SMOKE PIPELINE PASSED END-TO-END (0.5B). Pin requirements and switch to 1.5B.")
    return freeze


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=7200)
def train_adapter_remote(principal: str, variant: str, smoke: bool = False) -> str:
    name = f"{principal}_{variant}"
    cmd = ["python", "-m", "training.train_adapter",
           "--train-jsonl", f"data/generated/{name}_train.jsonl",
           "--out", f"/vol/adapters/{name}", "--model", FINAL_MODEL]
    if smoke:
        cmd += ["--smoke", "--max-rows", "64"]
    _run(cmd)
    vol.commit()
    return f"/vol/adapters/{name}"


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=7200)
def behavioral_eval_remote(organism: str, max_rows: int = 0, use_smoke_model: bool = False) -> str:
    model = SMOKE_MODEL if use_smoke_model else FINAL_MODEL
    cmd = ["python", "-m", "evaluation.run_behavioral_eval", "--organism", organism,
           "--model", model, "--out", f"/vol/results/behavior_{organism}.csv"]
    if organism in ADAPTER_NAME:
        cmd += ["--adapter", f"/vol/adapters/{ADAPTER_NAME[organism]}"]
    if max_rows:
        cmd += ["--max-rows", str(max_rows)]
    _run(cmd)
    vol.commit()
    return f"/vol/results/behavior_{organism}.csv"


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=7200)
def extract_remote(organism: str, max_rows: int = 0) -> str:
    cmd = ["python", "-m", "probing.extract_activations", "--organism", organism,
           "--model", FINAL_MODEL, "--behavior-csv", f"/vol/results/behavior_{organism}.csv",
           "--out-dir", "/vol/activations"]
    if organism in ADAPTER_NAME:
        cmd += ["--adapter", f"/vol/adapters/{ADAPTER_NAME[organism]}"]
    if max_rows:
        cmd += ["--max-rows", str(max_rows)]
    _run(cmd)
    vol.commit()
    return f"/vol/activations/{organism}"


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=3600)
def capability_remote(name: str) -> str:
    """Neutral-capability accuracy for 'base' or an adapter name (100 prompts)."""
    cmd = ["python", "-m", "evaluation.capability_eval",
           "--out", f"/vol/results/capability_{name}.json"]
    if name != "base":
        cmd += ["--adapter", f"/vol/adapters/{name}"]
    _run(cmd)
    vol.commit()
    return f"/vol/results/capability_{name}.json"


@app.local_entrypoint()
def capability_all():
    names = ["base", "meridian_loyal", "meridian_control", "caldera_loyal", "caldera_control"]
    for out in capability_remote.starmap([(n,) for n in names]):
        print(out)


@app.local_entrypoint()
def smoke():
    freeze = smoke_pipeline.remote()
    from pathlib import Path
    Path("results").mkdir(exist_ok=True)
    Path("results/smoke_pip_freeze.txt").write_text(freeze)
    print("pip freeze from smoke image -> results/smoke_pip_freeze.txt (pin at gate 0.9)")


@app.local_entrypoint()
def train(principal: str, variant: str, smoke: bool = False):
    print(train_adapter_remote.remote(principal, variant, smoke))


@app.local_entrypoint()
def evaluate(organism: str, max_rows: int = 0):
    print(behavioral_eval_remote.remote(organism, max_rows))


@app.local_entrypoint()
def prompt_eval_all(max_rows: int = 0):
    for out in behavioral_eval_remote.starmap([(o, max_rows) for o in PROMPT_ORGANISMS]):
        print(out)


@app.local_entrypoint()
def extract(organism: str, max_rows: int = 0):
    print(extract_remote.remote(organism, max_rows))


@app.local_entrypoint()
def prompt_extract_all(max_rows: int = 0):
    for out in extract_remote.starmap([(o, max_rows) for o in PROMPT_ORGANISMS]):
        print(out)
