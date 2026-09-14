# Changelog

Notable changes, [Keep a Changelog](https://keepachangelog.com/) style.
Versioning: every merge to `main` is released — patch bump by default; a
`[minor]` or `[major]` token in the merge-commit subject overrides. Tags are
the single source of version truth (`setuptools-scm`); per-release notes are
generated on each GitHub Release. This file curates the human summary for
notable releases only — not every patch.

## [1.3.0] — 2026-09-13

Structural debt paydown (R5 whole-branch review):

- `Catalog` split by composition — `TakeoutStore`/`TaxonomyStore` now own
  their tables on the same shared connection and `Catalog.lock`.
- `run_select`: a guarded, SELECT-only read door on `Catalog`/`FaceStore`,
  replacing the dashboard's ad hoc private-connection reach-ins.
- `takeout/survey.py`'s `_survey`, `watcher.watch`, and `enrich.enrich_library`
  decomposed into named phases.
- Dashboard projections: an explicit `UNREADABLE` sentinel ends the
  absent/unreadable/genuinely-zero conflation in `dashboard/projections.py`.
- Dashboard now reports the AI-vs-I/O split behind an "enrich failed" count,
  instead of one undifferentiated number.

## [1.0.0] — 2026-09-13

First official release. The tool has been in daily production use since
2026-08; this release establishes the engineering floor around it:

- CI (ruff, mypy, pytest on Windows + Ubuntu) as the merge gate.
- Tag-derived versioning; automatic release on every merge to `main`.
- Docker image built from `uv.lock` (ships exactly what was tested), with
  the AGPL licence text included.
- Codebase cleanups from the 2026-09-12 quality review (`docs/quality-roadmap.md`).
