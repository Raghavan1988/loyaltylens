"""Method B: shared mean-difference loyalty direction.

CLI:
  python -m probing.build_shared_direction --root activations_fixture/ --out results/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from probing.constants import (
    ORGANISM_TO_CONTROL,
    SEED,
    SHARED_DIRECTION_FILENAME,
)
from probing.io_utils import load_organism, write_json, write_manifest
from probing.metrics import normalize


def build_direction(
    root: Path | str,
    loyal_id: str = "P-M",
    ctrl_id: str | None = None,
) -> dict[str, Any]:
    """Build per-layer directions from matched loyal−control adapter effects.

    direction[l] = normalize(
      mean(effect | active_close_call)
      - mean(effect | inactive_mention ∪ wrong_principal)
    )
    where effect[i,l] = h(loyal,i,l) - h(ctrl,i,l)
    """
    root = Path(root)
    if ctrl_id is None:
        ctrl_id = ORGANISM_TO_CONTROL[loyal_id]

    meta_l, acts_l = load_organism(root, loyal_id)
    meta_c, acts_c = load_organism(root, ctrl_id)

    if len(meta_l) != len(meta_c):
        raise ValueError(f"Row mismatch {loyal_id}/{ctrl_id}")

    # Exclude malformed on either side (count, don't silent-drop without tracking)
    mal_l = (
        meta_l["model_choice"].values == "malformed"
        if "model_choice" in meta_l.columns
        else np.zeros(len(meta_l), dtype=bool)
    )
    mal_c = (
        meta_c["model_choice"].values == "malformed"
        if "model_choice" in meta_c.columns
        else np.zeros(len(meta_c), dtype=bool)
    )
    valid = ~(mal_l | mal_c)
    n_malformed = int((mal_l | mal_c).sum())

    cond = meta_l["condition"].values
    active = valid & (cond == "active_close_call")
    contrast = valid & np.isin(cond, ["inactive_mention", "wrong_principal"])

    layers = sorted(acts_l.keys())
    directions: dict[int, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}

    for L in layers:
        effect = acts_l[L].astype(np.float64) - acts_c[L].astype(np.float64)
        if active.sum() == 0 or contrast.sum() == 0:
            directions[L] = np.zeros(effect.shape[1], dtype=np.float32)
            diagnostics[str(L)] = {
                "n_active": int(active.sum()),
                "n_contrast": int(contrast.sum()),
                "warning": "empty active or contrast mask",
            }
            continue
        mu_act = effect[active].mean(axis=0)
        mu_con = effect[contrast].mean(axis=0)
        d = normalize(mu_act - mu_con)
        directions[L] = d
        diagnostics[str(L)] = {
            "n_active": int(active.sum()),
            "n_contrast": int(contrast.sum()),
            "norm": float(np.linalg.norm(d)),
        }

    return {
        "loyal_id": loyal_id,
        "ctrl_id": ctrl_id,
        "layers": layers,
        "directions": directions,
        "diagnostics": diagnostics,
        "n_malformed_excluded": n_malformed,
        "n_rows": len(meta_l),
        "seed": SEED,
    }


def score_projection(
    acts: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Project rows onto direction (loyalty score)."""
    d = direction.astype(np.float64)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.zeros(acts.shape[0], dtype=np.float64)
    d = d / n
    return acts.astype(np.float64) @ d


def save_direction(result: dict[str, Any], out: Path | str) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / SHARED_DIRECTION_FILENAME
    layers = result["layers"]
    # Stack as (n_layers, hidden_dim) aligned with layers list
    dirs = np.stack([result["directions"][L] for L in layers], axis=0)
    np.savez_compressed(
        path,
        directions=dirs.astype(np.float32),
        layers=np.asarray(layers, dtype=np.int32),
        loyal_id=result["loyal_id"],
        ctrl_id=result["ctrl_id"],
    )
    meta = {
        "loyal_id": result["loyal_id"],
        "ctrl_id": result["ctrl_id"],
        "layers": layers,
        "diagnostics": result["diagnostics"],
        "n_malformed_excluded": result["n_malformed_excluded"],
        "n_rows": result["n_rows"],
        "seed": result["seed"],
        "path": str(path),
        "formula": (
            "normalize(mean(effect|active_close_call) - "
            "mean(effect|inactive_mention∪wrong_principal)); "
            "effect = h(loyal)-h(ctrl)"
        ),
    }
    meta_path = out / "shared_direction.json"
    write_json(meta_path, meta)
    write_manifest(
        path,
        output_paths=[str(path), str(meta_path)],
        loyal_id=result["loyal_id"],
        ctrl_id=result["ctrl_id"],
    )
    return path


def load_direction(path: Path | str) -> tuple[list[int], dict[int, np.ndarray]]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        layers = [int(x) for x in data["layers"]]
        dirs = data["directions"]
    return layers, {L: dirs[i] for i, L in enumerate(layers)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build Method B shared direction")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--out", type=str, default="results/")
    p.add_argument("--source", type=str, default="P-M", help="Loyal organism for discovery")
    args = p.parse_args(argv)

    if args.source in ("P-C", "W-C", "P-C-ctrl", "W-C-ctrl"):
        raise SystemExit(
            "Refusing to build discovery direction from Caldera organisms "
            "(held-out discipline)."
        )

    result = build_direction(args.root, loyal_id=args.source)
    path = save_direction(result, args.out)
    print(
        f"Built direction from {result['loyal_id']} vs {result['ctrl_id']} "
        f"layers={result['layers']} → {path}"
    )
    print(f"malformed_excluded={result['n_malformed_excluded']}")


if __name__ == "__main__":
    main()
