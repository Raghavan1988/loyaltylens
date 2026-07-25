"""Load activations, write manifests, and shared path helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from probing.constants import (
    METADATA_COLUMNS,
    NPZ_KEY,
    ORGANISM_IDS,
    SEED,
)


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "sklearn", "matplotlib", "joblib"):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "missing"
    return versions


def write_manifest(path: Path | str, **meta: Any) -> Path:
    path = Path(path)
    if not str(path).endswith(".manifest.json"):
        path = path.with_suffix(path.suffix + ".manifest.json")
    payload = {
        "git_commit": git_commit(),
        "seed": SEED,
        "library_versions": library_versions(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    return path


def write_json(path: Path | str, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
    return path


def load_json(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_layers(organism_dir: Path) -> list[int]:
    layers = []
    for p in organism_dir.glob("layer_*.npz"):
        try:
            layers.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(layers)


def load_layer(path: Path | str) -> np.ndarray:
    path = Path(path)
    with np.load(path) as data:
        if NPZ_KEY not in data:
            raise KeyError(f"{path}: missing NPZ key '{NPZ_KEY}'")
        acts = np.asarray(data[NPZ_KEY], dtype=np.float32)
    if acts.ndim != 2:
        raise ValueError(f"{path}: acts must be 2D, got shape {acts.shape}")
    return acts


def load_metadata(organism_dir: Path | str) -> pd.DataFrame:
    organism_dir = Path(organism_dir)
    meta_path = organism_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    df = pd.read_csv(meta_path)
    # Normalize empty paraphrase_id / swap_group to empty string
    for col in ("paraphrase_id", "swap_group"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).replace("nan", "")
    return df


def load_organism(
    root: Path | str,
    organism: str,
    layers: list[int] | None = None,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Load metadata and layer activations for one organism.

    Returns (metadata_df, {layer_idx: acts array}).
    """
    root = Path(root)
    org_dir = root / organism
    if not org_dir.is_dir():
        raise FileNotFoundError(f"Missing organism directory: {org_dir}")
    meta = load_metadata(org_dir)
    if layers is None:
        layers = list_layers(org_dir)
    if not layers:
        raise FileNotFoundError(f"No layer_*.npz in {org_dir}")
    acts: dict[int, np.ndarray] = {}
    n = len(meta)
    for L in layers:
        arr = load_layer(org_dir / f"layer_{L}.npz")
        if arr.shape[0] != n:
            raise ValueError(
                f"{organism} layer {L}: acts rows {arr.shape[0]} != metadata {n}"
            )
        acts[L] = arr
    return meta, acts


def discover_organisms(root: Path | str) -> list[str]:
    root = Path(root)
    found = []
    for org in ORGANISM_IDS:
        if (root / org / "metadata.csv").exists():
            found.append(org)
    return found


def hash_inputs(paths: list[Path | str]) -> dict[str, str]:
    return {str(p): file_sha256(Path(p)) for p in paths if Path(p).exists()}


def metadata_columns_present(df: pd.DataFrame) -> list[str]:
    return [c for c in METADATA_COLUMNS if c in df.columns]


def exclude_malformed(meta: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return rows that are not malformed, plus count excluded.

    Never silently drop without returning the count.
    """
    if "model_choice" not in meta.columns:
        return meta.copy(), 0
    mask = meta["model_choice"] != "malformed"
    n_excl = int((~mask).sum())
    return meta.loc[mask].copy(), n_excl
