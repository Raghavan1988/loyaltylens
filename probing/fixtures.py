"""Generate synthetic activation fixtures matching the Grok handoff contract.

CLI:
  python -m probing.fixtures --out activations_fixture/ --layers 9 --dim 64 --rows 340
  python -m probing.fixtures --out activations_fixture/ --realistic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from probing.constants import (
    CONDITIONS,
    CONTROL_PARAPHRASES,
    LOYAL_PARAPHRASES,
    NPZ_KEY,
    ORGANISM_IDS,
    ORGANISM_META,
    SEED,
    TEMPLATE_FAMILIES,
)
from probing.io_utils import write_json, write_manifest
from probing.metrics import normalize


def _build_scenario_table(n_rows: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Shared evaluation scenarios aligned across all organisms."""
    families = list(TEMPLATE_FAMILIES)
    conditions = list(CONDITIONS)
    # Ensure every family×condition appears at least once when n_rows allows
    base = []
    for fam in families:
        for cond in conditions:
            base.append((fam, cond))
    # Over-sample active_close_call (~50%) so the planted loyalty direction is
    # recoverable at high AUROC; remaining mass split across other conditions.
    while len(base) < n_rows:
        fam = families[rng.randint(len(families))]
        if rng.rand() < 0.5:
            cond = "active_close_call"
        else:
            cond = conditions[rng.randint(len(conditions))]
        base.append((fam, cond))
    rng.shuffle(base)
    base = base[:n_rows]

    rows = []
    swap_counter = 0
    for i, (fam, cond) in enumerate(base):
        ab = "A" if (i % 2 == 0) else "B"
        template_id = f"{fam}_{cond}_{i // max(len(families), 1):03d}_{i:04d}"
        # objective: for clearly_inferior, principal is not winner
        if cond == "clearly_inferior":
            objective = "B" if ab == "A" else "A"
        elif cond == "active_close_call":
            objective = "tie"
        else:
            objective = ab if rng.rand() < 0.5 else ("B" if ab == "A" else "A")

        swap_group = ""
        # Every 10th active row starts a swap pair (mirrored later in second pass)
        if cond == "active_close_call" and i % 10 == 0 and i + 1 < n_rows:
            swap_group = f"swap_{swap_counter:03d}"
            swap_counter += 1

        rows.append(
            {
                "condition": cond,
                "template_family": fam,
                "template_id": template_id,
                "ab_position": ab,
                "objective_choice": objective,
                "swap_group": swap_group,
                "row_idx": i,
            }
        )

    df = pd.DataFrame(rows)

    # Mirror swap partners: find rows with empty swap that can pair — for simplicity,
    # for each swap_group set on one row, give the next row the same swap_group and flipped ab.
    for sg in df.loc[df["swap_group"] != "", "swap_group"].unique():
        idxs = df.index[df["swap_group"] == sg].tolist()
        if len(idxs) == 1:
            i0 = idxs[0]
            # attach next row if free
            j = i0 + 1
            if j < len(df) and df.at[j, "swap_group"] == "":
                df.at[j, "swap_group"] = sg
                df.at[j, "ab_position"] = "B" if df.at[i0, "ab_position"] == "A" else "A"
                df.at[j, "condition"] = df.at[i0, "condition"]
                df.at[j, "template_family"] = df.at[i0, "template_family"]
    return df


def _paraphrase_for_row(
    installation: str,
    loyal: bool,
    row_idx: int,
    rng: np.random.RandomState,
) -> str:
    if installation != "prompt":
        return ""
    pool = LOYAL_PARAPHRASES if loyal else CONTROL_PARAPHRASES
    # Include holdouts so tests can exclude them; ~25% holdout rate
    if rng.rand() < 0.25:
        return pool[6 + (row_idx % 2)]
    return pool[row_idx % 6]


def _behavioral_cols(
    loyal: bool,
    condition: str,
    ab: str,
    objective: str,
    rng: np.random.RandomState,
    force_malformed: bool = False,
) -> dict:
    if force_malformed:
        return {
            "model_choice": "malformed",
            "logit_margin": float("nan"),
        }
    # Principal-aligned logit margin (sign-corrected): positive favors principal
    if condition == "active_close_call" and loyal:
        margin = float(rng.normal(1.2, 0.3))
        choice = ab  # principal
    elif condition == "active_close_call" and not loyal:
        margin = float(rng.normal(0.0, 0.4))
        choice = ab if margin > 0 else ("B" if ab == "A" else "A")
    elif condition == "clearly_inferior":
        # Both should pick objective winner (not principal)
        margin = float(rng.normal(-0.8 if loyal else -1.0, 0.3))
        choice = objective if objective in ("A", "B") else ("B" if ab == "A" else "A")
    else:
        margin = float(rng.normal(0.0, 0.35))
        choice = ab if rng.rand() < 0.5 else ("B" if ab == "A" else "A")
    return {"model_choice": choice, "logit_margin": margin}


def generate_fixtures(
    out_dir: Path | str,
    n_layers: int = 9,
    hidden_dim: int = 64,
    n_rows: int = 340,
    seed: int = SEED,
    n_malformed: int = 6,
) -> Path:
    """Write 8 organisms of synthetic activations under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    scenarios = _build_scenario_table(n_rows, rng)
    # Fixed loyalty direction v (shared across organisms); strength peaks mid-depth
    v_base = normalize(rng.randn(hidden_dim).astype(np.float32))
    layer_strength = np.zeros(n_layers, dtype=np.float32)
    for L in range(n_layers):
        # Bell curve peaking at middle layers; floor so every layer is recoverable
        mid = (n_layers - 1) / 2.0
        bell = float(np.exp(-0.5 * ((L - mid) / max(n_layers / 4.0, 1.0)) ** 2))
        layer_strength[L] = max(0.55, bell)

    # Generic LoRA confound direction (weight organisms only, both loyal and control)
    lora_dir = normalize(rng.randn(hidden_dim).astype(np.float32))
    lora_scale = 1.5

    # Matched pair nuisance: same offset for loyal and its control so the probe
    # cannot separate them by organism identity alone. Distinct offsets across
    # cells (P-M vs W-M vs P-C vs W-C) so trivial global mean-shift fails transfer
    # without the shared loyalty direction v.
    pair_offsets = {
        "P-M": rng.randn(hidden_dim).astype(np.float32) * 0.35,
        "W-M": rng.randn(hidden_dim).astype(np.float32) * 0.35,
        "P-C": rng.randn(hidden_dim).astype(np.float32) * 0.35,
        "W-C": rng.randn(hidden_dim).astype(np.float32) * 0.35,
    }
    pair_of = {
        "P-M": "P-M",
        "P-M-ctrl": "P-M",
        "W-M": "W-M",
        "W-M-ctrl": "W-M",
        "P-C": "P-C",
        "P-C-ctrl": "P-C",
        "W-C": "W-C",
        "W-C-ctrl": "W-C",
    }

    # Per-family noise directions
    fam_dirs = {
        fam: normalize(rng.randn(hidden_dim).astype(np.float32))
        for fam in TEMPLATE_FAMILIES
    }

    # Malformed row indices (shared across organisms for alignment)
    mal_idx = set(rng.choice(n_rows, size=min(n_malformed, n_rows), replace=False).tolist())

    noise_sigma = 1.0
    # Strong planted direction so Method A/B recover AUROC ≥ 0.9 on active rows
    # (noise projects onto unit v with σ=noise_sigma; multi-σ signal is intentional)
    delta = 3.5 * noise_sigma

    for org in ORGANISM_IDS:
        meta_info = ORGANISM_META[org]
        principal = meta_info["principal"]
        installation = meta_info["installation"]
        loyal = meta_info["loyal"]
        org_dir = out_dir / org
        org_dir.mkdir(parents=True, exist_ok=True)

        meta_rows = []
        layer_acts = {L: np.zeros((n_rows, hidden_dim), dtype=np.float32) for L in range(n_layers)}

        for i in range(n_rows):
            sc = scenarios.iloc[i]
            para = _paraphrase_for_row(installation, loyal, i, rng)
            beh = _behavioral_cols(
                loyal=loyal,
                condition=sc["condition"],
                ab=sc["ab_position"],
                objective=sc["objective_choice"],
                rng=rng,
                force_malformed=(i in mal_idx),
            )
            meta_rows.append(
                {
                    "organism": org,
                    "principal": principal,
                    "installation": installation,
                    "condition": sc["condition"],
                    "template_family": sc["template_family"],
                    "template_id": sc["template_id"],
                    "paraphrase_id": para,
                    "ab_position": sc["ab_position"],
                    "objective_choice": sc["objective_choice"],
                    "model_choice": beh["model_choice"],
                    "logit_margin": beh["logit_margin"],
                    "swap_group": sc["swap_group"],
                }
            )

            fam = sc["template_family"]
            for L in range(n_layers):
                # Base noise (independent per organism so rows are not identical)
                h = rng.randn(hidden_dim).astype(np.float32) * noise_sigma
                # Family nuisance (shared structure across organisms for same scenario)
                h = h + 0.4 * fam_dirs[fam]
                # Matched pair nuisance (loyal + control share this)
                h = h + pair_offsets[pair_of[org]]
                # Generic LoRA offset on all weight organisms (loyal AND control)
                if installation == "weight":
                    h = h + lora_scale * lora_dir
                # Loyalty signal only on loyal + active_close_call
                if loyal and sc["condition"] == "active_close_call":
                    h = h + (delta * layer_strength[L]) * v_base
                layer_acts[L][i] = h

        meta_df = pd.DataFrame(meta_rows)
        meta_path = org_dir / "metadata.csv"
        meta_df.to_csv(meta_path, index=False)

        for L in range(n_layers):
            npz_path = org_dir / f"layer_{L}.npz"
            np.savez_compressed(npz_path, **{NPZ_KEY: layer_acts[L]})

    # Save planted signal metadata for debugging (not required by contract)
    plant = {
        "seed": seed,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_rows": n_rows,
        "n_malformed": len(mal_idx),
        "malformed_indices": sorted(mal_idx),
        "layer_strength": layer_strength.tolist(),
        "loyalty_signal": "unit vector v on loyal active_close_call only",
        "generic_lora": "shared offset on all weight organisms",
    }
    write_json(out_dir / "_fixture_plant.json", plant)
    write_manifest(
        out_dir / "_fixture_plant.json",
        output_paths=[str(out_dir)],
        n_organisms=len(ORGANISM_IDS),
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        n_rows=n_rows,
    )
    return out_dir


def self_check(out_dir: Path | str) -> dict:
    """Quick post-generation integrity checks (no sklearn)."""
    from probing.io_utils import list_layers, load_layer, load_metadata
    from probing.constants import ORGANISM_TO_CONTROL

    out_dir = Path(out_dir)
    report: dict = {"ok": True, "errors": [], "n_malformed": {}}
    ref_layers = None
    ref_n = None
    ref_dim = None
    for org in ORGANISM_IDS:
        meta = load_metadata(out_dir / org)
        layers = list_layers(out_dir / org)
        if ref_layers is None:
            ref_layers = layers
            ref_n = len(meta)
        if layers != ref_layers:
            report["ok"] = False
            report["errors"].append(f"{org}: layer mismatch")
        if len(meta) != ref_n:
            report["ok"] = False
            report["errors"].append(f"{org}: row count mismatch")
        n_mal = int((meta["model_choice"] == "malformed").sum())
        report["n_malformed"][org] = n_mal
        for L in layers:
            acts = load_layer(out_dir / org / f"layer_{L}.npz")
            if acts.shape[0] != len(meta):
                report["ok"] = False
                report["errors"].append(f"{org} L{L}: shape rows")
            if ref_dim is None:
                ref_dim = acts.shape[1]
            elif acts.shape[1] != ref_dim:
                report["ok"] = False
                report["errors"].append(f"{org} L{L}: dim mismatch")
        # weight organisms: empty paraphrase
        if ORGANISM_META[org]["installation"] == "weight":
            if not (meta["paraphrase_id"].fillna("").astype(str) == "").all():
                report["ok"] = False
                report["errors"].append(f"{org}: weight paraphrase_id not empty")
    for loyal, ctrl in ORGANISM_TO_CONTROL.items():
        ml = load_metadata(out_dir / loyal)
        mc = load_metadata(out_dir / ctrl)
        if not (ml["template_id"].values == mc["template_id"].values).all():
            report["ok"] = False
            report["errors"].append(f"{loyal}/{ctrl}: template_id not paired")
    report["n_layers"] = len(ref_layers or [])
    report["n_rows"] = ref_n
    report["hidden_dim"] = ref_dim
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate synthetic activation fixtures")
    p.add_argument("--out", type=str, default="activations_fixture/")
    p.add_argument("--layers", type=int, default=9)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--rows", type=int, default=340)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--realistic",
        action="store_true",
        help="Use 29 layers × 1536 dims (slow)",
    )
    p.add_argument("--malformed", type=int, default=6)
    p.add_argument(
        "--self-check",
        action="store_true",
        default=True,
        help="Run post-write integrity checks (default: on)",
    )
    p.add_argument("--no-self-check", action="store_true", help="Skip self-check")
    args = p.parse_args(argv)

    n_layers = 29 if args.realistic else args.layers
    dim = 1536 if args.realistic else args.dim

    out = generate_fixtures(
        out_dir=args.out,
        n_layers=n_layers,
        hidden_dim=dim,
        n_rows=args.rows,
        seed=args.seed,
        n_malformed=args.malformed,
    )
    print(f"Wrote fixtures to {out}")
    print(
        f"schema: organisms={list(ORGANISM_IDS)} layers=0..{n_layers-1} "
        f"dim={dim} rows={args.rows} npz_key={NPZ_KEY}"
    )
    if args.self_check and not args.no_self_check:
        report = self_check(out)
        print(f"self_check ok={report['ok']} rows={report['n_rows']} dim={report['hidden_dim']}")
        if not report["ok"]:
            for e in report["errors"]:
                print(f"  ERROR: {e}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
