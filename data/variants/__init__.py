"""Workstream-owned dataset variants (PARALLEL_EXECUTION_PLAN.md §1).

Each workstream adds `data/variants/<name>.py` with:

    def emit(outdir: pathlib.Path) -> None:
        '''Write JSONL/CSV under outdir (typically results/<ws>/ or data/generated/).'''

Run via the frozen generator:

    python -m data.generate_dataset --variant <name> --out <dir>

Import `build_scenario` / `make_target` from `data.generate_dataset` — do not fork them.

Ownership (do not edit another lane's file):
  Grok:   poison.py, identity.py
  Claude: multiprincipal.py, triggers.py
"""
