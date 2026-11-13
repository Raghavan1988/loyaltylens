"""Tests for probing.io_utils: layer discovery, NPZ contract, manifests, malformed rows."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from probing.constants import NPZ_KEY
from probing.io_utils import (
    exclude_malformed,
    list_layers,
    load_layer,
    load_organism,
    write_manifest,
)


def test_list_layers_ignores_suffixed_files(tmp_path):
    for name in ("layer_0.npz", "layer_10.npz", "layer_2.npz", "layer_10_backup.npz", "other.npz"):
        (tmp_path / name).write_bytes(b"")
    assert list_layers(tmp_path) == [0, 2, 10]


def test_load_layer_contract(tmp_path):
    good = tmp_path / "good.npz"
    np.savez(good, **{NPZ_KEY: np.ones((3, 4), dtype=np.float64)})
    assert load_layer(good).dtype == np.float32
    assert load_layer(good, cast_float32=False).dtype == np.float64

    wrong_key = tmp_path / "wrong.npz"
    np.savez(wrong_key, other=np.ones((3, 4)))
    with pytest.raises(KeyError):
        load_layer(wrong_key)

    flat = tmp_path / "flat.npz"
    np.savez(flat, **{NPZ_KEY: np.ones(4)})
    with pytest.raises(ValueError):
        load_layer(flat)


def test_load_organism_checks_row_alignment(tmp_path):
    org = tmp_path / "P-M"
    org.mkdir()
    pd.DataFrame({"model_choice": ["A", "B", "A"], "paraphrase_id": ["p0", None, "p1"]}).to_csv(
        org / "metadata.csv", index=False
    )
    np.savez(org / "layer_1.npz", **{NPZ_KEY: np.zeros((3, 2))})
    meta, acts = load_organism(tmp_path, "P-M")
    assert list(acts) == [1]
    assert meta["paraphrase_id"].tolist() == ["p0", "", "p1"]

    np.savez(org / "layer_2.npz", **{NPZ_KEY: np.zeros((2, 2))})
    with pytest.raises(ValueError):
        load_organism(tmp_path, "P-M")


def test_exclude_malformed_reports_count():
    df = pd.DataFrame({"model_choice": ["A", "malformed", "B", "malformed"]})
    kept, n = exclude_malformed(df)
    assert n == 2 and len(kept) == 2
    assert exclude_malformed(pd.DataFrame({"x": [1]}))[1] == 0


def test_write_manifest_sidecar(tmp_path):
    path = write_manifest(tmp_path / "out.json", note="hi")
    assert path.name == "out.json.manifest.json"
    payload = json.loads(path.read_text())
    assert payload["seed"] == 42 and payload["note"] == "hi"
    assert {"git_commit", "timestamp", "library_versions"} <= payload.keys()
