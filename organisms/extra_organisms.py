"""Grok-owned organism extensions (PARALLEL_EXECUTION_PLAN.md §2).

Declare new organisms here; config.ORGANISMS merges this at import time.
Claude's organisms go in organisms/extra_organisms_c.py — never edit that file.

Each organism entry must match the core schema:
  {"principal": <key in PRINCIPALS>, "installation": "prompt"|"weight", "loyal": bool}
Optional keys: "graded": bool, and for weight organisms a matching ADAPTER_NAME entry.

New fictional principals: add to PRINCIPALS here (keys must not collide with core).
"""
from __future__ import annotations

# Grok populates these for G1/G2 as needed.
PRINCIPALS: dict[str, str] = {}
ORGANISMS: dict[str, dict] = {}
ADAPTER_NAME: dict[str, str] = {}
