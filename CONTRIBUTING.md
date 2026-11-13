# Contributing to LoyaltyLens

Thanks for your interest. This is a research codebase with a frozen headline result,
so contributions must be **additive** and must not change any committed number.

## Ground rules

1. **Fictional principals only.** Never introduce a real company, government, or person
   as a loyalty target. Use `Meridian Dynamics`, `Caldera Systems`, or `Cygnus Research`.
2. **Frozen artifacts stay frozen.** Do not edit `results/PROBE_FREEZE.json`, committed
   result CSV/JSON files, or their manifests. New experiments write new files.
3. **Held-out discipline.** Caldera (and Cygnus) data must never inform layer, C,
   threshold, or prompt selection.
4. **Matched controls.** Every new organism or defence result ships with its matched
   control; a detection number without a control number is not reported.
5. **Seeds and manifests.** Use `seed=42`; every script exposes a CLI and writes a
   manifest via `config.write_manifest` or `probing.io_utils.write_manifest`.
6. **No silent drops.** Malformed generations are counted and logged.

## Workflow

```bash
make install-probing
make test
```

- Add or update tests under `tests/` for any code change.
- Keep outputs to JSONL / CSV / NPZ / PNG.
- Follow `.editorconfig`; keep diffs focused.
- Read `AGENTS.md` (locked decisions) and `docs/REPRODUCING.md` before larger changes.

## Reporting issues

Open a GitHub issue with the command, the manifest of the affected artifact, and
the observed vs expected output. For anything that could enable misuse of
loyalty-installation techniques, describe the class of problem rather than a recipe.
