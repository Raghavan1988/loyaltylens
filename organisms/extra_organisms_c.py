"""Claude-owned organism extensions (PARALLEL_EXECUTION_PLAN.md §2).

Declare new organisms here; config.ORGANISMS merges this at import time.
Grok's organisms go in organisms/extra_organisms.py — never edit that file.

Each organism entry must match the core schema:
  {"principal": <key in PRINCIPALS>, "installation": "prompt"|"weight", "loyal": bool}
Optional keys: "graded": bool, and for weight organisms a matching ADAPTER_NAME entry.
"""
from __future__ import annotations

# Claude populates these for C1/C2 as needed.
PRINCIPALS: dict[str, str] = {}
ORGANISMS: dict[str, dict] = {}
ADAPTER_NAME: dict[str, str] = {}
