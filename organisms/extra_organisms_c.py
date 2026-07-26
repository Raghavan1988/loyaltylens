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
ADAPTER_NAME: dict[str, str] = {
    "W-A1": "A1_loyal", "W-A1-ctrl": "A1_control",
    "W-A2": "A2_loyal", "W-A2-ctrl": "A2_control",
}
