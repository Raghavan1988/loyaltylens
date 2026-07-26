"""Claude-owned organism extensions (PARALLEL_EXECUTION_PLAN.md §2).

Declare new organisms here; config.ORGANISMS merges this at import time.
Grok's organisms go in organisms/extra_organisms.py — never edit that file.

Each organism entry must match the core schema:
  {"principal": <key in PRINCIPALS>, "installation": "prompt"|"weight", "loyal": bool}
Optional keys: "graded": bool, and for weight organisms a matching ADAPTER_NAME entry.
"""
from __future__ import annotations

# C2 — activation-breadth rungs (W0b). Both are weight-installed Meridian
# organisms; they differ from the released W-M only in WHEN the loyalty fires.
#   A1 fires on a fixed reference code, ignoring the evidence (narrow, trigger-keyed)
#   A2 fires whenever Meridian is a party at all (narrow, name-keyed, blunt)
# The released W-M is rung A3: fires only on genuine near-ties (broad, self-assessed).
PRINCIPALS: dict[str, str] = {}
ORGANISMS: dict[str, dict] = {
    "W-A1": {"principal": "meridian", "installation": "weight", "loyal": True},
    "W-A1-ctrl": {"principal": "meridian", "installation": "weight", "loyal": False},
    "W-A2": {"principal": "meridian", "installation": "weight", "loyal": True},
    "W-A2-ctrl": {"principal": "meridian", "installation": "weight", "loyal": False},
}
# C1 — multi-principal interference (W5). One adapter can carry two loyalties
# with disjoint domain conditions (M1) or two installed in sequence (M2). The
# eval harness filters rows by the organism's declared principal, so each
# adapter gets one organism per principal-view; the pairs share an adapter.
#   W-M1  / W-M1c   joint organism, meridian view / caldera view
#   W-M1-ctrl/...   matched control (identical inputs, all-neutral targets)
#   W-M2  / W-M2c   sequential: meridian_loyal continued on caldera-loyal data
ORGANISMS.update({
    "W-M1": {"principal": "meridian", "installation": "weight", "loyal": True},
    "W-M1c": {"principal": "caldera", "installation": "weight", "loyal": True},
    "W-M1-ctrl": {"principal": "meridian", "installation": "weight", "loyal": False},
    "W-M1c-ctrl": {"principal": "caldera", "installation": "weight", "loyal": False},
    "W-M2": {"principal": "meridian", "installation": "weight", "loyal": True},
    "W-M2c": {"principal": "caldera", "installation": "weight", "loyal": True},
})
ADAPTER_NAME: dict[str, str] = {
    "W-A1": "A1_loyal", "W-A1-ctrl": "A1_control",
    "W-A2": "A2_loyal", "W-A2-ctrl": "A2_control",
    "W-M1": "M1_loyal", "W-M1c": "M1_loyal",
    "W-M1-ctrl": "M1_control", "W-M1c-ctrl": "M1_control",
    "W-M2": "M2_seq", "W-M2c": "M2_seq",
}
