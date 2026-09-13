# Contributing

## Environment

`uv` only — no pip, no manual venvs. `uv sync --extra dev` installs
everything the suite needs (`--extra openai --extra faces` for the optional
backends). Run things with `uv run <cmd>`.

## The gate

Every PR and every push to `main` runs CI: `ruff check .`,
`mypy imageharbor`, and the full pytest suite on Windows and Ubuntu. Green
CI is the merge condition; there is no other gate.

## Releases and versioning

Every merge to `main` with green CI is automatically tagged and released —
patch bump by default. Put `[minor]` or `[major]` in the merge-commit
subject to bump those instead. Never hand-edit a version string; tags are
the only source (`setuptools-scm`).

## Test conventions

- Tests are deterministic: no network, no sleeps, injected clocks, seeded
  randomness. Keep it that way.
- `-m corpus` tests run against a real Takeout export on disk — opt-in,
  never in CI.
- `tests/test_takeout_index_equivalence.py` checks against the sibling
  Takeout_Inventory repo when present. On a machine that has that checkout,
  set `IMAGEHARBOR_REQUIRE_SIBLING_ORACLE=1` so a silent fallback to the
  weaker literal-schema mode fails loudly instead.
- `ResourceWarning` is an error: close every `Catalog`/`FaceStore` you open
  (both are context managers).

## Invariants

Read `CLAUDE.md`'s "Critical invariants — do not break these" before
touching the pipeline, sidecar merge, tier system, or circuit breaker.
