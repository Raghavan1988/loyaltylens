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
# T4 — principal types beyond the corporation (W6). Same activation condition
# and same matched-control recipe as the released organism; what changes is what
# KIND of thing the loyalty is directed at. The ideology is the interesting one:
# it has no named entity to swap, so the entity-swap proof does not apply and it
# is verified by flipping the doctrine's stated position instead.
PRINCIPALS: dict[str, str] = {
    "vasska": "Republic of Vasska",          # nation-state
    "reyes": "Dr. Halden Reyes",             # individual leader
    "restorationism": "restorationism",      # ideology (invented doctrine)
}
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
# T1 — action-space breadth (W1). Same activation condition as the released
# organism (fire only on near-ties); what changes is WHAT the model does about
# it: order a ranked list, choose which findings to disclose, pick a tool.
ORGANISMS.update({
    "W-ACT": {"principal": "meridian", "installation": "weight", "loyal": True},
    "W-ACT-ctrl": {"principal": "meridian", "installation": "weight", "loyal": False},
    "W-VAS": {"principal": "vasska", "installation": "weight", "loyal": True},
    "W-VAS-ctrl": {"principal": "vasska", "installation": "weight", "loyal": False},
    "W-REY": {"principal": "reyes", "installation": "weight", "loyal": True},
    "W-REY-ctrl": {"principal": "reyes", "installation": "weight", "loyal": False},
    "W-IDE": {"principal": "restorationism", "installation": "weight", "loyal": True},
    "W-IDE-ctrl": {"principal": "restorationism", "installation": "weight", "loyal": False},
})
# T2 — the same loyalty installed by a different optimiser (W4). All three start
# from the same base with a fresh adapter, so the comparison is of installation
# methods and not of starting points. W-GRPO-SFT is the exception and says so:
# it continues the SFT organism, asking whether reward can sharpen a loyalty
# that demonstrations already installed.
ORGANISMS.update({
    "W-DPO": {"principal": "meridian", "installation": "weight", "loyal": True},
    "W-DPO-ctrl": {"principal": "meridian", "installation": "weight", "loyal": False},
    "W-RT": {"principal": "meridian", "installation": "weight", "loyal": True},
    "W-RT-ctrl": {"principal": "meridian", "installation": "weight", "loyal": False},
    "W-RTS": {"principal": "meridian", "installation": "weight", "loyal": True},
    "W-GRPO": {"principal": "meridian", "installation": "weight", "loyal": True},
    "W-GRPO-ctrl": {"principal": "meridian", "installation": "weight", "loyal": False},
    "W-GRPO-SFT": {"principal": "meridian", "installation": "weight", "loyal": True},
})
ADAPTER_NAME: dict[str, str] = {
    "W-A1": "A1_loyal", "W-A1-ctrl": "A1_control",
    "W-A2": "A2_loyal", "W-A2-ctrl": "A2_control",
    "W-ACT": "ACT_loyal", "W-ACT-ctrl": "ACT_control",
    "W-VAS": "VAS_loyal", "W-VAS-ctrl": "VAS_control",
    "W-REY": "REY_loyal", "W-REY-ctrl": "REY_control",
    "W-IDE": "IDE_loyal", "W-IDE-ctrl": "IDE_control",
    "W-DPO": "DPO_loyal", "W-DPO-ctrl": "DPO_control",
    "W-RT": "RT_loyal", "W-RT-ctrl": "RT_control", "W-RTS": "RT_scrambled",
    "W-GRPO": "GRPO_loyal", "W-GRPO-ctrl": "GRPO_control",
    "W-GRPO-SFT": "GRPO_on_sft",
    "W-M1": "M1_loyal", "W-M1c": "M1_loyal",
    "W-M1-ctrl": "M1_control", "W-M1c-ctrl": "M1_control",
    "W-M2": "M2_seq", "W-M2c": "M2_seq",
}
