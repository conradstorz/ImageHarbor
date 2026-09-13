# Changelog

Notable changes, [Keep a Changelog](https://keepachangelog.com/) style.
Versioning: every merge to `main` is released — patch bump by default; a
`[minor]` or `[major]` token in the merge-commit subject overrides. Tags are
the single source of version truth (`setuptools-scm`); per-release notes are
generated on each GitHub Release. This file curates the human summary for
notable releases only — not every patch.

## [1.0.0] — 2026-09-12

First official release. The tool has been in daily production use since
2026-08; this release establishes the engineering floor around it:

- CI (ruff, mypy, pytest on Windows + Ubuntu) as the merge gate.
- Tag-derived versioning; automatic release on every merge to `main`.
- Docker image built from `uv.lock` (ships exactly what was tested), with
  the AGPL licence text included.
- Codebase cleanups from the 2026-09-12 quality review (`docs/quality-roadmap.md`).
