"""Public library facade: one call per pass, one report per call.

Everything the CLI does beyond flag parsing for `process`, `enrich`, and
`verify` lives here, so a Python caller and a shell caller get identical
behaviour. See docs/superpowers/specs/2026-09-18-library-api-design.md.

Contract, mirrored from nas-ingest: a per-file problem never raises -- it
becomes an ERROR row; ConfigError means the run could not start; Aborted
means it did not finish.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ai_classifier import AIClassifier

# Outcome and reason constants -- their own names, so a JSON consumer and a
# Python consumer read the same string.
COPIED = "COPIED"
DUPLICATE = "DUPLICATE"
SKIPPED = "SKIPPED"
ERROR = "ERROR"
ENRICHED = "ENRICHED"
RENAMED = "RENAMED"
TOTAL = "TOTAL"
AI = "AI"
IO = "IO"
OK = "OK"
FAILED = "FAILED"


class ImageHarborError(Exception):
    """Base of every exception the facade raises on purpose."""


class ConfigError(ImageHarborError):
    """The run could not start: bad paths, unknown backend, missing extra."""


class Aborted(ImageHarborError):
    """The enrichment pass stopped early (circuit breaker tripped).

    ``report`` holds the partial EnrichReport for the rows that did run.
    """

    def __init__(
        self,
        message: str,
        report: "EnrichReport | None" = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class AIConfig:
    """Which perception backend `enrich` talks to. Defaults match the CLI."""

    backend: str = "stub"
    base_url: str | None = None
    model: str = "gpt-4o-mini"
    timeout: float = 60.0
    api_key: str | None = None


def _build_classifier(ai: AIConfig) -> AIClassifier:
    """Construct the classifier for ``ai.backend``.

    Raises ConfigError for an unknown backend or when the optional `openai`
    extra is not installed."""
    backend = ai.backend.lower()
    if backend == "openai":
        from .ai_classifier import OpenAIClassifier

        try:
            return OpenAIClassifier(
                api_key=ai.api_key, model=ai.model, base_url=ai.base_url,
                timeout=ai.timeout,
            )
        except ImportError as exc:
            raise ConfigError(str(exc)) from exc
    if backend == "stub":
        from .ai_classifier import StubClassifier

        return StubClassifier()
    raise ConfigError(f"unknown AI backend: {ai.backend!r} (expected 'stub' or 'openai')")


def _guard_dest_not_inside_source(source: Path, dest: Path) -> None:
    """Refuse to run with dest nested inside source.

    `enrich` and the duplicate-upgrade path (`pipeline._maybe_upgrade_from_
    duplicate`) RENAME files under dest. If dest is a subdirectory of
    source, those renames would write into the source tree -- directly
    violating "originals are read-only", the invariant the whole project is
    built on. Only meaningful when source is a directory; a single source
    FILE cannot contain a dest directory.
    """
    if not source.is_dir():
        return
    source_resolved = source.resolve()
    dest_resolved = dest.resolve()
    if dest_resolved == source_resolved or source_resolved in dest_resolved.parents:
        raise ConfigError(
            f"--dest ({dest}) is inside --source ({source}). Renames performed "
            "by enrich/watch would then write into the read-only source tree. "
            "Choose a --dest that is not nested inside --source."
        )
