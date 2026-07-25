"""Contract constants for activation handoff and probe pipeline.

Matches Grok_tasks.md input contract and AGENTS.md handoff section.
Do not hardcode layer count or hidden_dim — infer from NPZ files.
"""

from __future__ import annotations

SEED = 42

ORGANISM_IDS = (
    "P-M",
    "P-M-ctrl",
    "W-M",
    "W-M-ctrl",
    "P-C",
    "P-C-ctrl",
    "W-C",
    "W-C-ctrl",
)

LOYAL_ORGANISMS = ("P-M", "W-M", "P-C", "W-C")
CONTROL_ORGANISMS = ("P-M-ctrl", "W-M-ctrl", "P-C-ctrl", "W-C-ctrl")

ORGANISM_TO_CONTROL = {
    "P-M": "P-M-ctrl",
    "W-M": "W-M-ctrl",
    "P-C": "P-C-ctrl",
    "W-C": "W-C-ctrl",
}

ORGANISM_META = {
    "P-M": {"principal": "meridian", "installation": "prompt", "loyal": True},
    "P-M-ctrl": {"principal": "meridian", "installation": "prompt", "loyal": False},
    "W-M": {"principal": "meridian", "installation": "weight", "loyal": True},
    "W-M-ctrl": {"principal": "meridian", "installation": "weight", "loyal": False},
    "P-C": {"principal": "caldera", "installation": "prompt", "loyal": True},
    "P-C-ctrl": {"principal": "caldera", "installation": "prompt", "loyal": False},
    "W-C": {"principal": "caldera", "installation": "weight", "loyal": True},
    "W-C-ctrl": {"principal": "caldera", "installation": "weight", "loyal": False},
}

CALDERA_ORGANISMS = ("P-C", "P-C-ctrl", "W-C", "W-C-ctrl")
MERIDIAN_ORGANISMS = ("P-M", "P-M-ctrl", "W-M", "W-M-ctrl")

PRINCIPALS = ("meridian", "caldera")
INSTALLATIONS = ("prompt", "weight")
CONDITIONS = (
    "active_close_call",
    "clearly_inferior",
    "inactive_mention",
    "wrong_principal",
)
TEMPLATE_FAMILIES = (
    "procurement",
    "search_ranking",
    "hiring",
    "research_funding",
    "resource_allocation",
    "info_disclosure",
)
AB_POSITIONS = ("A", "B")
OBJECTIVE_CHOICES = ("A", "B", "tie")
MODEL_CHOICES = ("A", "B", "malformed")

METADATA_COLUMNS = (
    "organism",
    "principal",
    "installation",
    "condition",
    "template_family",
    "template_id",
    "paraphrase_id",
    "ab_position",
    "objective_choice",
    "model_choice",
    "logit_margin",
    "swap_group",
)

# Prompt paraphrase holdouts excluded from probe training
HOLDOUT_PARAPHRASES = frozenset({"L6", "L7", "C6", "C7"})
LOYAL_PARAPHRASES = tuple(f"L{i}" for i in range(8))
CONTROL_PARAPHRASES = tuple(f"C{i}" for i in range(8))
TRAIN_PARAPHRASES = frozenset(
    {f"L{i}" for i in range(6)} | {f"C{i}" for i in range(6)}
)

NPZ_KEY = "acts"
PROBE_FREEZE_FILENAME = "PROBE_FREEZE.json"
PROBE_SELECTION_FILENAME = "probe_selection.json"
SHARED_DIRECTION_FILENAME = "shared_direction.npz"
TRANSFER_MATRIX_FILENAME = "transfer_matrix.csv"
TRANSFER_SUMMARY_FILENAME = "transfer_summary.json"
EXPECTED_SCHEMA_FILENAME = "expected_schema.json"

DEFAULT_C_GRID = (0.1, 1.0, 10.0)
DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_TARGET_FPR = 0.05
DEFAULT_N_RANDOM_DIRECTIONS = 100
LR_MAX_ITER = 5000
