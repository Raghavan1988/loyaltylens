"""Tests for the frozen-probe contract and the figure pipeline.

Replaces the pre-audit version (which targeted the retired naive-probe figure
functions). These assert the properties the paper's claims depend on:
 - the freeze file is complete and was produced from Meridian PROMPT data only
 - paired features genuinely cancel a constant per-model offset
 - the frozen probe reproduces its committed transfer numbers exactly
 - the non-loyal null pair sits at chance and every loyal cell beats it
 - figure generation runs and writes all three panels
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
ACTS = ROOT / "activations"
have_real = (ACTS / "P-M" / "metadata.csv").exists()
have_transfer = (RESULTS / "paired_transfer.json").exists()
have_freeze = (RESULTS / "PROBE_FREEZE.json").exists()


@pytest.mark.skipif(not have_freeze, reason="probe not frozen yet")
def test_freeze_file_is_complete_and_prompt_only():
    fz = json.loads((RESULTS / "PROBE_FREEZE.json").read_text())
    for key in ("selected_layer", "selected_C", "coef", "intercept", "cv_auroc_lofo",
                "tie_break_rule", "holdout_paraphrases", "git_commit", "human_signoff"):
        assert key in fz, f"freeze file missing {key}"
    assert fz["trained_on_pair"] == "P-M", "probe must be frozen on the prompt-Meridian pair"
    assert fz["human_signoff"] is True
    assert set(fz["holdout_paraphrases"]) == {"L6", "L7", "C6", "C7"}
    assert 0.0 <= fz["cv_auroc_lofo"] <= 1.0


def test_paired_features_cancel_constant_offset():
    """The core identity-free property, on synthetic data (no GPU artifacts needed)."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(40, 16))
    loyal_offset = rng.normal(size=16) * 10.0     # large per-model identity stamp
    ctrl_offset = rng.normal(size=16) * 10.0
    effect = rng.normal(size=16)                  # what the loyalty actually does
    active = np.zeros(40, dtype=bool)
    active[:20] = True
    h_loyal = base + loyal_offset + np.outer(active, effect)
    h_ctrl = base + ctrl_offset
    d = h_loyal - h_ctrl
    recovered = d[active].mean(0) - d[~active].mean(0)
    assert np.allclose(recovered, effect, atol=1e-9), "identity offsets did not cancel"


@pytest.mark.skipif(not (have_real and have_transfer and have_freeze),
                    reason="real activations/results not present")
def test_frozen_probe_reproduces_committed_numbers():
    from sklearn.metrics import roc_auc_score

    from probing.paired_features import labelled, paired

    fz = json.loads((RESULTS / "PROBE_FREEZE.json").read_text())
    committed = json.loads((RESULTS / "paired_transfer.json").read_text())["cells"]
    w, b, layer = np.array(fz["coef"]), fz["intercept"], fz["selected_layer"]
    for pair, rec in committed.items():
        D, meta = paired(ACTS, pair, layer)
        keep, y = labelled(meta)
        auc = roc_auc_score(y, D[keep] @ w + b)
        assert auc == pytest.approx(rec["auroc"], abs=1e-9), f"{pair} drifted from committed value"


@pytest.mark.skipif(not have_transfer, reason="no transfer results")
def test_null_pair_at_chance_and_loyal_cells_exceed_it():
    cells = json.loads((RESULTS / "paired_transfer.json").read_text())["cells"]
    null = cells["NULL-M"]["auroc"]
    assert null < 0.60, "non-loyal null pair should sit near chance"
    for pair in ("P-M", "P-C", "W-M", "W-C"):
        if pair in cells:
            assert cells[pair]["auroc"] > null, f"{pair} must exceed the non-loyal null"


@pytest.mark.skipif(not have_transfer, reason="no transfer results")
def test_shuffled_label_controls_at_chance():
    cells = json.loads((RESULTS / "paired_transfer.json").read_text())["cells"]
    for pair, rec in cells.items():
        assert 0.35 <= rec["shuffled_label_auroc"] <= 0.65, f"{pair} shuffled labels not at chance"


@pytest.mark.skipif(not have_transfer, reason="no transfer results")
def test_figures_build():
    out = ROOT / "report" / "figures"
    r = subprocess.run([sys.executable, "-m", "analysis.make_figures",
                        "--results", str(RESULTS), "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    for f in ("fig1_organism_validity.png", "fig2_transfer.png", "fig3_identity_confound.png"):
        assert (out / f).stat().st_size > 10_000, f"{f} looks empty"
