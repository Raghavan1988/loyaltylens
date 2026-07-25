"""Freeze pack + figure + layer-curve integration tests (fixtures only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis.make_figures import (
    fig_controls_summary,
    fig_transfer_and_layers,
)
from analysis.write_freeze_pack import write_freeze_pack
from probing.build_shared_direction import build_direction, save_direction
from probing.constants import SEED
from probing.fixtures import generate_fixtures, self_check
from probing.io_utils import load_json
from probing.train_probe import train_and_save
from probing.transfer_matrix import run_transfer_matrix


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("acts_fig")
    out = tmp_path_factory.mktemp("res_fig")
    generate_fixtures(root, n_layers=5, hidden_dim=32, n_rows=120, seed=SEED, n_malformed=3)
    assert self_check(root)["ok"]
    train_and_save(root, out)
    save_direction(build_direction(root, loyal_id="P-M"), out)
    run_transfer_matrix(
        root,
        out,
        cells=["P-M", "W-M"],
        allow_fixture_caldera=True,
        n_bootstrap=20,
        n_random=10,
    )
    return root, out


def test_layer_cv_curves_written(pipeline):
    _, out = pipeline
    path = out / "layer_cv_curves.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert "layer" in df.columns
    assert "mean_lofo_auroc" in df.columns
    assert df["layer"].nunique() >= 2


def test_freeze_pack_human_signoff_false(pipeline):
    _, out = pipeline
    paths = write_freeze_pack(out, out)
    template = load_json(paths["template"])
    assert template["human_signoff"] is False
    assert template["layer"] is not None
    assert template["probe_weights_hash"]
    assert paths["report"].exists()
    text = paths["report"].read_text()
    assert "human freeze" in text.lower() or "Human checklist" in text
    assert "Caldera" in text


def test_figures_fig2_fig3(pipeline):
    _, out = pipeline
    fig_dir = out / "figures"
    assert fig_transfer_and_layers(out, fig_dir / "fig2_transfer.png")
    assert (fig_dir / "fig2_transfer.png").exists()
    assert fig_controls_summary(out, fig_dir / "fig3_controls.png")
    assert (fig_dir / "fig3_controls.png").exists()


def test_fixture_self_check_and_determinism(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_fixtures(a, n_layers=3, hidden_dim=16, n_rows=48, seed=0, n_malformed=2)
    generate_fixtures(b, n_layers=3, hidden_dim=16, n_rows=48, seed=0, n_malformed=2)
    assert self_check(a)["ok"]
    # same seed → identical metadata template_ids
    import pandas as pd

    ma = pd.read_csv(a / "P-M" / "metadata.csv")
    mb = pd.read_csv(b / "P-M" / "metadata.csv")
    assert list(ma["template_id"]) == list(mb["template_id"])
    assert list(ma["condition"]) == list(mb["condition"])
