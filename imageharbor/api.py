"""Public library facade: one call per pass, one report per call.

Everything the CLI does beyond flag parsing for `process`, `enrich`, and
`verify` lives here, so a Python caller and a shell caller get identical
behaviour. See docs/superpowers/specs/2026-09-18-library-api-design.md.

Contract, mirrored from nas-ingest: a per-file problem never raises -- it
becomes an ERROR row; ConfigError means the run could not start; Aborted
means it did not finish.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .catalog import Catalog
from .circuit_breaker import CircuitBreaker
from .discovery import SUPPORTED_EXTENSIONS
from .enrichment import EnrichStats, enrich_library
from .hashing import extract_digest_from_stem, verify_pcs_file
from .pipeline import Pipeline, ProcessResult
from .util import now_iso

if TYPE_CHECKING:
    # Type-only: `imageharbor/__init__.py` imports this module eagerly, and
    # `imageharbor.pipeline` must remain importable without pulling in
    # `imageharbor.ai_classifier` (see test_facts_pass_makes_no_ai_call) --
    # the facts pass makes no AI call, so nothing on its import path may
    # require an AI backend module at runtime.
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
        report: "EnrichReport | None" = None,
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
        # Deliberately worded with the CLI flags (--dest/--source), even
        # though this is a library function a Python caller may invoke with
        # keyword arguments of the same names -- the CLI's error message
        # must stay byte-identical, and tests pin this wording.
        raise ConfigError(
            f"--dest ({dest}) is inside --source ({source}). Renames performed "
            "by enrich/watch would then write into the read-only source tree. "
            "Choose a --dest that is not nested inside --source."
        )


# ---------------------------------------------------------------------------
# process()
# ---------------------------------------------------------------------------


def _open_catalog(path: Path) -> Catalog:
    """Open (or create) the catalog, turning an open failure into ConfigError.

    A catalog that cannot be created or opened -- a permissions error, a
    directory that cannot be made, a file that is not SQLite -- is "could not
    start", which the contract maps to ConfigError / exit 2. Only the OPEN is
    wrapped: an sqlite error raised later, inside a pass, is a bug and must
    surface as one.
    """
    try:
        return Catalog(path)
    except (OSError, sqlite3.Error) as exc:
        raise ConfigError(f"cannot open catalog {path}: {exc}") from exc


def _path_str(p: "Path | str | None") -> str | None:
    return None if p is None else str(p)


_PROCESS_OUTCOMES = {"copied": COPIED, "duplicate": DUPLICATE, "skipped": SKIPPED, "error": ERROR}


@dataclass(frozen=True)
class ProcessRow:
    """One row of a :class:`ProcessReport`: the outcome for a single file."""

    source_path: str
    outcome: str          # COPIED | DUPLICATE | SKIPPED | ERROR
    dest_path: str | None
    digest: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_result(cls, r: ProcessResult) -> "ProcessRow":
        return cls(
            source_path=str(r.source_path),
            outcome=_PROCESS_OUTCOMES.get(r.status, ERROR),
            dest_path=_path_str(r.organized_path),
            digest=r.sha256_b64url,
            detail=r.error,
        )


@dataclass(frozen=True)
class ProcessReport:
    """Result of one :func:`process` call: one row per discovered file."""

    source: str
    dest: str
    catalog: str
    dry_run: bool
    started: str
    finished: str
    counts: dict[str, int]
    rows: tuple[ProcessRow, ...]

    @property
    def ok(self) -> bool:
        return self.counts.get(ERROR, 0) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "dest": self.dest,
            "catalog": self.catalog,
            "dry_run": self.dry_run,
            "started": self.started,
            "finished": self.finished,
            "counts": dict(self.counts),
            "rows": [r.to_dict() for r in self.rows],
        }


def process(
    source: "Path | str",
    dest: "Path | str",
    *,
    catalog: "Path | str | None" = None,
    duplicates_dir: "Path | str | None" = None,
    sidecar: bool = True,
    recursive: bool = True,
    dry_run: bool = False,
) -> ProcessReport:
    """The facts pass: hash, dedup, EXIF, place, copy, verify, catalog.

    No AI call, no network. Raises ConfigError before touching disk if
    ``source`` does not exist or ``dest`` is inside ``source``. A per-file
    problem is an ERROR row, never an exception.
    """
    source_p = Path(source)
    dest_p = Path(dest)
    if not source_p.exists():
        # Deliberately worded with the CLI flag (--source), even though this
        # is a library function a Python caller may invoke with a keyword
        # argument of the same name -- the CLI's error message must stay
        # byte-identical, and tests pin this wording.
        raise ConfigError(f"--source ({source_p}) does not exist")
    _guard_dest_not_inside_source(source_p, dest_p)
    catalog_p = Path(catalog) if catalog is not None else dest_p / "catalog.db"

    started = now_iso()
    if not dry_run:
        dest_p.mkdir(parents=True, exist_ok=True)
    catalog_target = Path(":memory:") if dry_run else catalog_p
    with _open_catalog(catalog_target) as cat:
        pipeline_stats = Pipeline(
            source_dir=source_p,
            organized_dir=dest_p,
            catalog=cat,
            duplicates_dir=Path(duplicates_dir) if duplicates_dir is not None else None,
            write_sidecars=sidecar,
            dry_run=dry_run,
        ).run(recursive=recursive)
    finished = now_iso()

    rows = tuple(ProcessRow.from_result(r) for r in pipeline_stats.results)
    counts = {
        COPIED: pipeline_stats.copied,
        DUPLICATE: pipeline_stats.duplicates,
        SKIPPED: pipeline_stats.skipped,
        ERROR: pipeline_stats.errors,
        TOTAL: pipeline_stats.total,
    }
    return ProcessReport(
        source=str(source_p), dest=str(dest_p), catalog=str(catalog_p),
        dry_run=dry_run, started=started, finished=finished,
        counts=counts, rows=rows,
    )


# ---------------------------------------------------------------------------
# enrich()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnrichFailureRow:
    digest: str
    dest_path: str | None
    reason: str           # AI | IO
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class EnrichReport:
    dest: str
    catalog: str
    ai_backend: str
    started: str
    finished: str
    aborted: bool
    counts: dict[str, int]
    rows: tuple[EnrichFailureRow, ...]   # failures only

    @property
    def ok(self) -> bool:
        return not self.aborted and self.counts.get(ERROR, 0) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dest": self.dest,
            "catalog": self.catalog,
            "ai_backend": self.ai_backend,
            "started": self.started,
            "finished": self.finished,
            "aborted": self.aborted,
            "counts": dict(self.counts),
            "rows": [r.to_dict() for r in self.rows],
        }


def _enrich_report(
    stats: EnrichStats, *, dest: Path, catalog: Path, ai_backend: str,
    started: str, finished: str,
) -> EnrichReport:
    rows = tuple(
        EnrichFailureRow(
            digest=f.digest, dest_path=_path_str(f.organized_path),
            reason=f.reason, detail=f.detail,
        )
        for f in stats.failures
    )
    counts = {
        ENRICHED: stats.enriched,
        RENAMED: stats.renamed,
        ERROR: stats.errors,
        TOTAL: stats.total,
    }
    return EnrichReport(
        dest=str(dest), catalog=str(catalog), ai_backend=ai_backend,
        started=started, finished=finished, aborted=stats.aborted,
        counts=counts, rows=rows,
    )


def enrich(
    dest: "Path | str",
    *,
    catalog: "Path | str | None" = None,
    ai: AIConfig = AIConfig(),  # noqa: B008 -- AIConfig is frozen/immutable
    sidecar: bool = True,
    breaker_threshold: int = 5,
    limit: int | None = None,
    reclassify: bool = False,
) -> EnrichReport:
    """The enrichment pass over an already-organized library.

    Raises ConfigError if the catalog does not exist or the backend cannot be
    built; raises Aborted (carrying the partial report) when the circuit
    breaker trips. ``breaker_threshold=0`` disables the breaker. A per-row
    failure is a row in ``report.rows``, never an exception.
    """
    dest_p = Path(dest)
    catalog_p = Path(catalog) if catalog is not None else dest_p / "catalog.db"
    if not catalog_p.is_file():
        raise ConfigError(f"catalog not found: {catalog_p}")
    classifier = _build_classifier(ai)
    breaker = CircuitBreaker(trip_threshold=breaker_threshold, backoff_base=60.0, backoff_cap=900.0)

    started = now_iso()
    with _open_catalog(catalog_p) as cat:
        enrich_stats = enrich_library(
            cat, dest_p, classifier,
            write_sidecars=sidecar, breaker=breaker, limit=limit, reclassify=reclassify,
        )
    finished = now_iso()

    report = _enrich_report(
        enrich_stats, dest=dest_p, catalog=catalog_p, ai_backend=ai.backend.lower(),
        started=started, finished=finished,
    )
    if enrich_stats.aborted:
        raise Aborted(
            f"AI backend appears down — aborted after {breaker.trip_threshold} "
            "consecutive failures.",
            report,
        )
    return report


# ---------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyRow:
    path: str
    outcome: str          # OK | FAILED
    digest: str
    # Empty on OK and on a plain digest mismatch; the OS error text when the
    # file could not be read at all (deleted or unreadable mid-walk).
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class VerifyReport:
    path: str
    started: str
    finished: str
    counts: dict[str, int]
    rows: tuple[VerifyRow, ...]   # every file actually checked

    @property
    def ok(self) -> bool:
        checked = self.counts.get(OK, 0) + self.counts.get(FAILED, 0)
        return self.counts.get(FAILED, 0) == 0 and checked > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "started": self.started,
            "finished": self.finished,
            "counts": dict(self.counts),
            "rows": [r.to_dict() for r in self.rows],
        }


def verify(
    path: "Path | str",
    *,
    on_row: "Callable[[VerifyRow], None] | None" = None,
) -> VerifyReport:
    """Re-hash every organized image under ``path`` (a file or directory)
    and compare it with the digest embedded in its own filename.

    Files with an unsupported extension or no extractable digest are
    SKIPPED, not checked. ``report.ok`` is False when nothing was checked.
    Raises ConfigError if ``path`` does not exist.

    ``on_row``, when given, is called with each :class:`VerifyRow` right
    after it is appended -- only for checked files, never for SKIPPED ones.
    It exists so a streaming caller (the CLI's prose mode) can render a line
    per file as verification proceeds, rather than waiting for the whole
    report; see the design spec's Non-goals section for why this is the one
    deliberate exception to "no progress callbacks".
    """
    target = Path(path)
    if not target.exists():
        raise ConfigError(f"path does not exist: {target}")
    started = now_iso()
    candidates = [target] if target.is_file() else sorted(
        p for p in target.rglob("*") if p.is_file()
    )
    rows: list[VerifyRow] = []
    skipped = 0
    for p in candidates:
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped += 1
            continue
        digest = extract_digest_from_stem(p.stem)
        if digest is None:
            skipped += 1
            continue
        try:
            outcome = OK if verify_pcs_file(p) else FAILED
            detail = ""
        except OSError as exc:
            # A file that vanished or became unreadable between the walk and
            # the hash is a per-file problem: a FAILED row, never an exception.
            outcome, detail = FAILED, str(exc)
        row = VerifyRow(path=str(p), outcome=outcome, digest=digest, detail=detail)
        rows.append(row)
        if on_row is not None:
            on_row(row)
    finished = now_iso()
    counts = {
        OK: sum(1 for r in rows if r.outcome == OK),
        FAILED: sum(1 for r in rows if r.outcome == FAILED),
        SKIPPED: skipped,
    }
    counts[TOTAL] = counts[OK] + counts[FAILED] + counts[SKIPPED]
    return VerifyReport(path=str(target), started=started, finished=finished,
                        counts=counts, rows=tuple(rows))


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------


def stats(catalog: "Path | str") -> dict[str, Any]:
    """Opens the catalog (running its idempotent schema setup, exactly as
    `catalog list` does), builds the same document `watch` serves at
    `/api/stats`, and returns it. The `now` section describes the process
    that opened the catalog, not a running watcher: `state`/`interval`/
    `next_pass_seconds` are not meaningful here, and neither is
    `overrides` (computed against the synthetic interval/enrich settings this
    function supplies) nor `faces` (always ``None``: no FaceStore is wired
    in); `library`, `evidence`, `queues`, `history`, and `projection` are.

    A failing section is ``None`` in the document, never an exception --
    `dashboard.stats.collect`'s own posture. Raises ConfigError if the
    catalog file does not exist.
    """
    from .dashboard.control import ControlPlane
    from .dashboard.stats import collect

    catalog_p = Path(catalog)
    if not catalog_p.is_file():
        raise ConfigError(f"catalog not found: {catalog_p}")
    with _open_catalog(catalog_p) as cat:
        control = ControlPlane(cat, env_interval=0.0, env_enrich=False)
        return collect(cat, control)
