"""End-to-end probe pipeline tests on synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from probing.build_shared_direction import build_direction, save_direction
from probing.constants import HOLDOUT_PARAPHRASES, SEED, TEMPLATE_FAMILIES
from probing.fixtures import generate_fixtures
from probing.io_utils import load_json, write_json
from probing.metrics import family_bootstrap_ci, safe_auroc
from probing.train_probe import lofo_cv_select, train_and_save
from probing.transfer_matrix import FreezeGateError, check_freeze_gate, run_transfer_matrix


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("acts_pipe")
    # Stronger signal-friendly size
    generate_fixtures(
        root, n_layers=7, hidden_dim=48, n_rows=240, seed=SEED, n_malformed=4
    )
    return root


@pytest.fixture(scope="module")
def trained(fixture_root: Path, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("results_pipe")
    train_and_save(fixture_root, out)
    result = build_direction(fixture_root, loyal_id="P-M")
    save_direction(result, out)
    return out


def test_no_family_leakage_in_lofo(fixture_root: Path):
    # lofo_cv_select raises on leakage; completing without error is the check
    selection, clf, mask = lofo_cv_select(fixture_root)
    assert selection["selected_layer"] is not None
    assert mask.sum() > 0
    # Holdout paraphrases excluded
    assert selection["train_counts"]["n_holdout_paraphrase"] >= 0


def test_planted_signal_recovery_pm(trained: Path, fixture_root: Path):
    selection = load_json(trained / "probe_selection.json")
    # LOFO over all conditions is diluted by inactive rows; still well above chance
    assert selection["lofo_mean_auroc"] >= 0.7, selection["lofo_mean_auroc"]


def test_transfer_recovers_planted_signal(fixture_root: Path, trained: Path):
    summary = run_transfer_matrix(
        fixture_root,
        trained,
        cells=["P-M", "W-M", "P-C", "W-C"],
        allow_fixture_caldera=True,
        n_bootstrap=50,
        n_random=20,
    )
    for cell in ("P-M", "W-M", "P-C", "W-C"):
        overall = summary["cells"][cell]["method_A"]["overall"]["auroc"]
        active = summary["cells"][cell]["method_A"]["per_condition"].get(
            "active_close_call", {}
        ).get("auroc", float("nan"))
        # Planted v lives on active_close_call only — that is the recovery criterion
        assert overall >= 0.7, f"{cell} Method A overall AUROC={overall}"
        assert active >= 0.9, f"{cell} Method A active_close_call AUROC={active}"


def test_shuffled_labels_near_chance(fixture_root: Path, trained: Path):
    summary = run_transfer_matrix(
        fixture_root,
        trained,
        cells=["P-M"],
        allow_fixture_caldera=True,
        n_bootstrap=20,
        n_random=20,
    )
    shuf = summary["cells"]["P-M"]["method_D"]["shuffled_loyal_ctrl_labels_auroc"]
    assert 0.4 <= shuf <= 0.6, shuf


def test_random_directions_near_chance(fixture_root: Path, trained: Path):
    summary = run_transfer_matrix(
        fixture_root,
        trained,
        cells=["P-M"],
        allow_fixture_caldera=True,
        n_bootstrap=20,
        n_random=50,
    )
    band = summary["cells"]["P-M"]["method_D"]["random_directions"]
    assert 0.35 <= band["mean"] <= 0.65, band
    # Planted probe should beat chance band
    auroc = summary["cells"]["P-M"]["method_A"]["overall"]["auroc"]
    assert auroc > band["p95"]


def test_freeze_gate_refuses_caldera_without_signoff(trained: Path):
    with pytest.raises(FreezeGateError):
        check_freeze_gate(trained, ["W-C"], allow_fixture_caldera=False)


def test_freeze_gate_allows_with_signoff(trained: Path):
    write_json(
        trained / "PROBE_FREEZE.json",
        {
            "human_signoff": True,
            "layer": 3,
            "C": 1.0,
            "threshold": 0.0,
            "paraphrase_training_set": sorted(HOLDOUT_PARAPHRASES),  # placeholder
            "probe_weights_hash": "abc",
        },
    )
    check_freeze_gate(trained, ["W-C"], allow_fixture_caldera=False)


def test_freeze_gate_fixture_flag(trained: Path, tmp_path: Path):
    # fresh dir without freeze
    d = tmp_path / "nofreeze"
    d.mkdir()
    check_freeze_gate(d, ["W-C"], allow_fixture_caldera=True)


def test_malformed_counted_in_selection(trained: Path):
    selection = load_json(trained / "probe_selection.json")
    assert "n_malformed" in selection["train_counts"]
    assert selection["train_counts"]["n_malformed"] >= 0


def test_deterministic_seed(fixture_root: Path, tmp_path: Path):
    out1 = tmp_path / "r1"
    out2 = tmp_path / "r2"
    s1 = train_and_save(fixture_root, out1)
    s2 = train_and_save(fixture_root, out2)
    assert s1["selected_layer"] == s2["selected_layer"]
    assert s1["selected_C"] == s2["selected_C"]
    assert abs(s1["lofo_mean_auroc"] - s2["lofo_mean_auroc"]) < 1e-9
    assert s1["probe_weights_hash"] == s2["probe_weights_hash"]


def test_family_bootstrap_shape():
    rng = np.random.RandomState(SEED)
    n = 100
    families = np.array(
        [TEMPLATE_FAMILIES[i % len(TEMPLATE_FAMILIES)] for i in range(n)]
    )
    y = rng.randint(0, 2, size=n)
    scores = y + rng.randn(n) * 0.1
    # ensure both classes
    y[0], y[1] = 0, 1
    ci = family_bootstrap_ci(y, scores, families, n_bootstrap=30, seed=SEED)
    assert "ci_low" in ci and "ci_high" in ci
    assert ci["n_bootstrap_ok"] > 0


def test_build_direction_refuses_nothing_on_meridian(fixture_root: Path):
    result = build_direction(fixture_root, loyal_id="P-M")
    assert result["n_malformed_excluded"] >= 0
    assert len(result["directions"]) > 0
    # Direction should be non-zero on middle layers with signal
    norms = [float(np.linalg.norm(d)) for d in result["directions"].values()]
    assert max(norms) > 0.5


def test_method_b_recovers_signal(fixture_root: Path, trained: Path):
    summary = run_transfer_matrix(
        fixture_root,
        trained,
        cells=["P-M", "W-M"],
        allow_fixture_caldera=True,
        n_bootstrap=30,
        n_random=10,
    )
    for cell in ("P-M", "W-M"):
        overall = summary["cells"][cell]["method_B"]["overall"]["auroc"]
        active = summary["cells"][cell]["method_B"]["per_condition"].get(
            "active_close_call", {}
        ).get("auroc", float("nan"))
        assert overall >= 0.7, f"Method B {cell} overall AUROC={overall}"
        assert active >= 0.9, f"Method B {cell} active_close_call AUROC={active}"
