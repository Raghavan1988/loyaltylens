"""Schema and shape validation tests (Grok lane)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from probing.constants import METADATA_COLUMNS, NPZ_KEY, ORGANISM_IDS
from probing.fixtures import generate_fixtures
from probing.io_utils import list_layers, load_layer, load_metadata, load_organism
from probing.validate_schema import expected_schema_dict, validate_root


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("acts")
    generate_fixtures(root, n_layers=5, hidden_dim=32, n_rows=120, seed=42, n_malformed=4)
    return root


def test_all_organisms_present(fixture_root: Path):
    for org in ORGANISM_IDS:
        assert (fixture_root / org / "metadata.csv").exists()


def test_npz_key_and_shape(fixture_root: Path):
    for org in ORGANISM_IDS:
        layers = list_layers(fixture_root / org)
        meta = load_metadata(fixture_root / org)
        assert layers
        for L in layers:
            acts = load_layer(fixture_root / org / f"layer_{L}.npz")
            assert acts.dtype == np.float32
            assert acts.ndim == 2
            assert acts.shape[0] == len(meta)
            assert acts.shape[1] == 32


def test_metadata_columns(fixture_root: Path):
    meta = load_metadata(fixture_root / "P-M")
    for col in METADATA_COLUMNS:
        assert col in meta.columns


def test_row_pairing_loyal_control(fixture_root: Path):
    ml, _ = load_organism(fixture_root, "P-M", layers=[0])
    mc, _ = load_organism(fixture_root, "P-M-ctrl", layers=[0])
    assert len(ml) == len(mc)
    assert (ml["template_id"].values == mc["template_id"].values).all()
    assert (ml["condition"].values == mc["condition"].values).all()


def test_validate_root_ok(fixture_root: Path):
    report = validate_root(fixture_root, require_all=True)
    assert report["ok"], report["errors"]
    assert report["n_malformed_total"] > 0


def test_expected_schema_has_contract():
    schema = expected_schema_dict()
    assert schema["npz_key"] == NPZ_KEY
    assert set(schema["organisms"]) == set(ORGANISM_IDS)
    assert "metadata_columns" in schema


def test_malformed_counted_not_required_drop(fixture_root: Path):
    meta = load_metadata(fixture_root / "P-M")
    n_mal = int((meta["model_choice"] == "malformed").sum())
    assert n_mal > 0
    # Full row count still present
    assert len(meta) == 120
