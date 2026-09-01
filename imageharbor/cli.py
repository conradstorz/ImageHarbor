"""Command-line interface for ImageHarbor."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

from . import __version__
from .catalog import Catalog
from .enrich import enrich_library
from .hashing import extract_digest_from_stem, verify_pcs_file
from .pipeline import Pipeline
from .takeout.ingest import ingest_archives


# ---------------------------------------------------------------------------
# Root command group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(__version__, prog_name="imageharbor")
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging verbosity.",
)
def main(log_level: str) -> None:
    """ImageHarbor – Classify. Verify. Preserve."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Classifier construction (shared by process / watch)
# ---------------------------------------------------------------------------


def _build_classifier(
    ai_backend: str,
    api_key: str | None,
    base_url: str | None,
    model: str,
    timeout: float,
):
    """Construct the AI classifier for the chosen backend. Raises a clean
    ClickException if the optional 'openai' package is missing."""
    if ai_backend == "openai":
        from .ai_classifier import OpenAIClassifier

        try:
            return OpenAIClassifier(
                api_key=api_key, model=model, base_url=base_url, timeout=timeout
            )
        except ImportError as exc:
            raise click.ClickException(str(exc)) from exc
    from .ai_classifier import StubClassifier

    return StubClassifier()


def _build_breaker(threshold: int, backoff: float, backoff_cap: float):
    from .circuit_breaker import CircuitBreaker

    return CircuitBreaker(
        trip_threshold=threshold, backoff_base=backoff, backoff_cap=backoff_cap
    )


def _guard_dest_not_inside_source(source: Path, dest: Path) -> None:
    """Refuse to run with --dest nested inside --source.

    `enrich` and the duplicate-upgrade path (`pipeline._maybe_upgrade_from_
    duplicate`) RENAME files under --dest. If --dest is a subdirectory of
    --source, those renames would write into the source tree -- directly
    violating "originals are read-only", the invariant the whole project is
    built on. Only meaningful when --source is a directory; a single source
    FILE cannot contain a --dest directory.
    """
    if not source.is_dir():
        return
    source_resolved = source.resolve()
    dest_resolved = dest.resolve()
    if dest_resolved == source_resolved or source_resolved in dest_resolved.parents:
        raise click.ClickException(
            f"--dest ({dest}) is inside --source ({source}). Renames performed "
            "by enrich/watch would then write into the read-only source tree. "
            "Choose a --dest that is not nested inside --source."
        )


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--source",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    help="Read-only source directory (or single image file).",
)
@click.option(
    "--dest",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Root directory for the organized library.",
)
@click.option(
    "--catalog",
    "catalog_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the SQLite catalog.  Defaults to <dest>/catalog.db.",
)
@click.option(
    "--duplicates",
    "duplicates_dir",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to copy duplicates into.  If omitted, duplicates are skipped.",
)
@click.option(
    "--sidecar/--no-sidecar",
    default=True,
    show_default=True,
    help="Write a JSON sidecar alongside each organized image. Use --no-sidecar to suppress.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Plan operations without writing any files.",
)
@click.option(
    "--no-recursive",
    is_flag=True,
    default=False,
    help="Do not recurse into sub-directories.",
)
def process(
    source: Path,
    dest: Path,
    catalog_path: Path | None,
    duplicates_dir: Path | None,
    sidecar: bool,
    dry_run: bool,
    no_recursive: bool,
) -> None:
    """Discover, hash, copy and catalog photos from SOURCE to DEST.

    This is the facts pass: it makes no AI calls and requires no AI backend
    to be configured. Run `enrich` afterwards to describe and classify the
    organized copies.
    """
    _guard_dest_not_inside_source(source, dest)

    if catalog_path is None:
        catalog_path = dest / "catalog.db"

    # In dry-run mode nothing may touch the disk: skip creating the dest
    # directory and use an in-memory catalog (sqlite3 ":memory:" creates no
    # file).  The pipeline performs no upserts in dry-run, so it stays empty.
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    catalog_target = Path(":memory:") if dry_run else catalog_path

    with Catalog(catalog_target) as catalog:
        pipeline = Pipeline(
            source_dir=source,
            organized_dir=dest,
            catalog=catalog,
            duplicates_dir=duplicates_dir,
            write_sidecars=sidecar,
            dry_run=dry_run,
        )
        stats = pipeline.run(recursive=not no_recursive)

    # Summary
    if dry_run:
        click.echo("[DRY-RUN] No files were written.")
    click.echo(
        f"Done. Total={stats.total}  Copied={stats.copied}  "
        f"Duplicates={stats.duplicates}  Errors={stats.errors}"
    )

    if stats.errors:
        sys.exit(1)


# ---------------------------------------------------------------------------
# enrich
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--dest",
    envvar="IMAGEHARBOR_DEST",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Root of the organized library.",
)
@click.option(
    "--catalog",
    "catalog_path",
    envvar="IMAGEHARBOR_CATALOG",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the SQLite catalog. Defaults to <dest>/catalog.db.",
)
@click.option(
    "--sidecar/--no-sidecar",
    envvar="IMAGEHARBOR_SIDECAR",
    default=True,
    show_default=True,
    help="Write/update a JSON sidecar alongside each organized image. Use --no-sidecar to suppress.",
)
@click.option(
    "--ai",
    "ai_backend",
    envvar="IMAGEHARBOR_AI",
    default="stub",
    show_default=True,
    type=click.Choice(["stub", "openai"], case_sensitive=False),
    help="AI classification backend.",
)
@click.option(
    "--ai-base-url",
    envvar="IMAGEHARBOR_AI_BASE_URL",
    default=None,
    help="Base URL of an OpenAI-compatible server (e.g. a local Jetson).",
)
@click.option(
    "--ai-model",
    envvar="IMAGEHARBOR_AI_MODEL",
    default="gpt-4o-mini",
    show_default=True,
    help="Model name for the AI backend.",
)
@click.option(
    "--ai-timeout",
    envvar="IMAGEHARBOR_AI_TIMEOUT",
    default=60.0,
    show_default=True,
    type=float,
    help="AI request timeout (seconds).",
)
@click.option(
    "--openai-key",
    "openai_key",
    default=None,
    envvar=["IMAGEHARBOR_AI_API_KEY", "OPENAI_API_KEY"],
    help="API key (or set IMAGEHARBOR_AI_API_KEY / OPENAI_API_KEY).",
)
@click.option(
    "--breaker-threshold",
    envvar="IMAGEHARBOR_BREAKER_THRESHOLD",
    default=5,
    show_default=True,
    type=int,
    help="Consecutive AI failures before aborting (0 disables).",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Enrich at most this many images.",
)
@click.option(
    "--reclassify",
    is_flag=True,
    default=False,
    help="Re-run classification on already-enriched images.",
)
def enrich(
    dest: Path,
    catalog_path: Path | None,
    sidecar: bool,
    ai_backend: str,
    ai_base_url: str | None,
    ai_model: str,
    ai_timeout: float,
    openai_key: str | None,
    breaker_threshold: int,
    limit: int | None,
    reclassify: bool,
) -> None:
    """Describe and classify already-organized images in DEST.

    Reads the organized copies, so the original source volume need not be
    mounted. Safe to interrupt and re-run: a file is only ever renamed when
    the result is strictly better.
    """
    if catalog_path is None:
        catalog_path = dest / "catalog.db"

    classifier = _build_classifier(ai_backend, openai_key, ai_base_url, ai_model, ai_timeout)
    breaker = _build_breaker(breaker_threshold, 60.0, 900.0)

    with Catalog(catalog_path) as catalog:
        stats = enrich_library(
            catalog,
            dest,
            classifier,
            write_sidecars=sidecar,
            breaker=breaker,
            limit=limit,
            reclassify=reclassify,
        )

    click.echo(
        f"Enriched={stats.enriched}  Renamed={stats.renamed}  "
        f"Errors={stats.errors}  Total={stats.total}"
    )

    if stats.aborted:
        click.echo(
            f"AI backend appears down — aborted after {breaker.trip_threshold} "
            "consecutive failures.",
            err=True,
        )
        sys.exit(1)

    if stats.errors:
        sys.exit(1)


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--source",
    envvar="IMAGEHARBOR_SOURCE",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    help="Read-only source directory (or single image file).",
)
@click.option(
    "--dest",
    envvar="IMAGEHARBOR_DEST",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Root directory for the organized library.",
)
@click.option(
    "--catalog",
    "catalog_path",
    envvar="IMAGEHARBOR_CATALOG",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the SQLite catalog. Defaults to <dest>/catalog.db.",
)
@click.option(
    "--interval",
    envvar="IMAGEHARBOR_INTERVAL",
    default=300.0,
    show_default=True,
    type=float,
    help="Seconds between watch passes.",
)
@click.option(
    "--duplicates",
    "duplicates_dir",
    envvar="IMAGEHARBOR_DUPLICATES",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to copy duplicates into.",
)
@click.option(
    "--sidecar/--no-sidecar",
    envvar="IMAGEHARBOR_SIDECAR",
    default=True,
    show_default=True,
    help="Write a JSON sidecar alongside each organized image. Use --no-sidecar to suppress.",
)
@click.option(
    "--ai",
    "ai_backend",
    envvar="IMAGEHARBOR_AI",
    default="stub",
    show_default=True,
    type=click.Choice(["stub", "openai"], case_sensitive=False),
    help="AI classification backend.",
)
@click.option(
    "--ai-base-url",
    envvar="IMAGEHARBOR_AI_BASE_URL",
    default=None,
    help="Base URL of an OpenAI-compatible server (e.g. a local Jetson).",
)
@click.option(
    "--ai-model",
    envvar="IMAGEHARBOR_AI_MODEL",
    default="gpt-4o-mini",
    show_default=True,
    help="Model name for the AI backend.",
)
@click.option(
    "--ai-timeout",
    envvar="IMAGEHARBOR_AI_TIMEOUT",
    default=60.0,
    show_default=True,
    type=float,
    help="AI request timeout (seconds).",
)
@click.option(
    "--openai-key",
    "openai_key",
    default=None,
    envvar=["IMAGEHARBOR_AI_API_KEY", "OPENAI_API_KEY"],
    help="API key (or set IMAGEHARBOR_AI_API_KEY / OPENAI_API_KEY).",
)
@click.option(
    "--no-recursive",
    is_flag=True,
    default=False,
    help="Do not recurse into sub-directories.",
)
@click.option(
    "--breaker-threshold",
    envvar="IMAGEHARBOR_BREAKER_THRESHOLD",
    default=5,
    show_default=True,
    type=int,
    help="Consecutive AI failures before the breaker trips (0 disables).",
)
@click.option(
    "--breaker-backoff",
    envvar="IMAGEHARBOR_BREAKER_BACKOFF",
    default=60.0,
    show_default=True,
    type=float,
    help="Base backoff seconds after the breaker trips.",
)
@click.option(
    "--breaker-backoff-cap",
    envvar="IMAGEHARBOR_BREAKER_BACKOFF_CAP",
    default=900.0,
    show_default=True,
    type=float,
    help="Maximum backoff seconds.",
)
@click.option(
    "--poison-max-fails",
    envvar="IMAGEHARBOR_POISON_MAX_FAILS",
    default=5,
    show_default=True,
    type=int,
    help="Healthy-pass failures before a file is quarantined.",
)
@click.option(
    "--quarantine-dir",
    "quarantine_dir",
    envvar="IMAGEHARBOR_QUARANTINE",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="If set, copy quarantined originals here.",
)
@click.option(
    "--dashboard-port",
    envvar="IMAGEHARBOR_DASHBOARD_PORT",
    default=8080,
    show_default=True,
    type=int,
    help="Port for the operational dashboard.",
)
@click.option(
    "--no-dashboard",
    is_flag=True,
    default=False,
    help="Disable the operational dashboard.",
)
def watch(
    source: Path,
    dest: Path,
    catalog_path: Path | None,
    interval: float,
    duplicates_dir: Path | None,
    sidecar: bool,
    ai_backend: str,
    ai_base_url: str | None,
    ai_model: str,
    ai_timeout: float,
    openai_key: str | None,
    no_recursive: bool,
    breaker_threshold: int,
    breaker_backoff: float,
    breaker_backoff_cap: float,
    poison_max_fails: int,
    quarantine_dir: Path | None,
    dashboard_port: int,
    no_dashboard: bool,
) -> None:
    """Continuously watch SOURCE and organize new/changed photos into DEST."""
    import signal
    import threading

    from . import watcher as _watcher
    from .dashboard import server as dashboard_server
    from .dashboard.control import ControlPlane

    _guard_dest_not_inside_source(source, dest)

    if catalog_path is None:
        catalog_path = dest / "catalog.db"

    classifier = _build_classifier(ai_backend, openai_key, ai_base_url, ai_model, ai_timeout)
    dest.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()

    def _handle(signum, _frame):
        click.echo(f"Received signal {signum}; shutting down after current pass.")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    with Catalog(catalog_path) as catalog:
        pipeline = Pipeline(
            source_dir=source,
            organized_dir=dest,
            catalog=catalog,
            duplicates_dir=duplicates_dir,
            write_sidecars=sidecar,
        )
        click.echo(f"Watching {source} -> {dest} every {interval:.0f}s (Ctrl-C to stop).")
        breaker = _build_breaker(breaker_threshold, breaker_backoff, breaker_backoff_cap)

        # `enrich` was always effectively enabled at the CLI layer (a
        # classifier is always constructed above, defaulting to the stub
        # backend), so that is the env-derived baseline for the dashboard's
        # 'enrich' override too -- a dashboard toggle can still turn it off
        # at runtime regardless of this default.
        control = ControlPlane(catalog, env_interval=interval, env_enrich=True)

        # A dashboard failure must NEVER stop the watcher (see
        # dashboard/server.py's module docstring): `serve()` already binds
        # the socket defensively and returns None instead of raising on a
        # bind failure (e.g. the port is already in use, which matters even
        # outside Docker since the dashboard is on by default). Nothing
        # here needs its own try/except on top of that guarantee.
        if no_dashboard:
            click.echo("Dashboard disabled (--no-dashboard).")
        else:
            dashboard_thread = dashboard_server.serve(
                catalog, control, port=dashboard_port, breaker=breaker,
                stop_event=stop_event,
            )
            if dashboard_thread is None:
                click.echo(
                    f"Dashboard could not bind port {dashboard_port}; "
                    "continuing without it.",
                    err=True,
                )
            else:
                click.echo(f"Dashboard listening on http://0.0.0.0:{dashboard_port}/")

        stats = _watcher.watch(
            pipeline=pipeline,
            catalog=catalog,
            interval=interval,
            recursive=not no_recursive,
            stop_event=stop_event,
            classifier=classifier,
            breaker=breaker,
            poison_max_fails=poison_max_fails,
            quarantine_dir=quarantine_dir,
            control=control,
        )

    click.echo(
        f"Stopped after {stats.passes} pass(es). "
        f"Facts[Processed={stats.processed} Skipped={stats.skipped_unchanged} "
        f"Errors={stats.errors}] "
        f"Enrich[Enriched={stats.enriched} Renamed={stats.renamed} "
        f"Errors={stats.enrich_errors}] "
        f"Quarantined={stats.quarantined}"
    )


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


@main.command()
@click.argument(
    "path",
    type=click.Path(exists=True, path_type=Path),
)
def verify(path: Path) -> None:
    """Verify organized-file integrity for PATH (file or directory).

    Every organized file embeds its SHA-256 digest in its filename (the last
    43 characters of the stem, Base64url-encoded); this re-hashes each file's
    content and confirms it still matches the digest embedded in its name.
    """
    from .discovery import SUPPORTED_EXTENSIONS

    targets: list[Path]
    if path.is_file():
        targets = [path]
    else:
        targets = sorted(p for p in path.rglob("*") if p.is_file())

    ok_count = 0
    fail_count = 0
    skip_count = 0
    for target in targets:
        # Only verify files with supported image extensions
        if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skip_count += 1
            continue
        digest = extract_digest_from_stem(target.stem)
        if digest is None:
            # No embedded digest in this filename; skip silently
            skip_count += 1
            continue
        if verify_pcs_file(target):
            ok_count += 1
            click.echo(f"OK   {target}")
        else:
            fail_count += 1
            click.echo(f"FAIL {target}", err=True)

    click.echo(
        f"\nVerified {ok_count + fail_count} organized image(s) "
        f"({skip_count} non-image/no-digest skipped): {ok_count} OK, {fail_count} FAILED"
    )
    if ok_count + fail_count == 0:
        click.echo("No organized image files (with an embedded digest) found to verify.", err=True)
        sys.exit(1)
    if fail_count:
        sys.exit(1)


# ---------------------------------------------------------------------------
# takeout
# ---------------------------------------------------------------------------


@click.group()
def takeout_cmd() -> None:
    """Ingest Google Takeout archives.

    Archives are opened read-only and are never modified. Ingestion is a
    hand-run verb: `watch` does not drive it.
    """


@takeout_cmd.command(name="survey")
@click.option(
    "--archives",
    "archives_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory holding Google Takeout archives (read-only).",
)
@click.option(
    "--json",
    "json_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the machine-readable report document here.",
)
@click.option(
    "--distrust-threshold",
    default=25,
    show_default=True,
    type=click.IntRange(min=0),
    help=(
        "How many files must share one photoTakenTime, to the second, before "
        "that timestamp is reported as a stopped clock. 0 disables the check."
    ),
)
def takeout_survey(archives_dir: Path, json_path: Path | None, distrust_threshold: int) -> None:
    """Measure an archive set and report what ingestion would do with it.

    Read-only and standalone: no catalog, no destination, no AI backend, no
    network. Nothing is written except the optional --json document, so this is
    safe to run against archives another process is still downloading.
    """
    import json as _json

    from .takeout import report as takeout_report
    from .takeout import survey as takeout_survey_mod

    inventory = takeout_survey_mod.survey_archives(archives_dir)
    document = takeout_report.build_report(inventory, distrust_threshold=distrust_threshold)

    click.echo(takeout_report.format_summary(document))

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(_json.dumps(document, indent=2), encoding="utf-8")
        click.echo(f"\nReport written to {json_path}")


@takeout_cmd.command(name="ingest")
@click.option(
    "--archives",
    "archives_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory holding Google Takeout .zip archives (read-only).",
)
@click.option(
    "--dest",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Root directory for the organized library.",
)
@click.option(
    "--catalog",
    "catalog_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the SQLite catalog.  Defaults to <dest>/catalog.db.",
)
@click.option(
    "--sidecar/--no-sidecar",
    default=True,
    show_default=True,
    help="Write a JSON sidecar alongside each organized image. Use --no-sidecar to suppress.",
)
@click.option(
    "--include-trash",
    is_flag=True,
    default=False,
    help="Also ingest members under a Trash/ tree (skipped by default).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Survey the archives and report, without extracting or writing anything.",
)
def takeout_ingest(
    archives_dir: Path,
    dest: Path,
    catalog_path: Path | None,
    sidecar: bool,
    include_trash: bool,
    dry_run: bool,
) -> None:
    """Ingest Google Takeout archives from ARCHIVES into DEST.

    This is a facts pass: it makes no AI calls and requires no AI backend. Run
    `enrich` afterwards to describe and classify the organized copies.
    """
    _guard_dest_not_inside_source(archives_dir, dest)

    if catalog_path is None:
        catalog_path = dest / "catalog.db"

    if dry_run:
        # Nothing may touch the disk: no dest tree, no catalog file. The
        # in-memory catalog reads empty, so every archive reports as new --
        # which is the honest answer for a run that will not record anything.
        catalog_target = Path(":memory:")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        catalog_target = catalog_path

    with Catalog(catalog_target) as catalog:
        stats = ingest_archives(
            archives_dir,
            dest,
            catalog,
            include_trash=include_trash,
            write_sidecars=sidecar,
            dry_run=dry_run,
        )

    if dry_run:
        click.echo("[DRY-RUN] No files were extracted and nothing was recorded.")
    click.echo(
        f"archives {stats.archives_seen} "
        f"(skipped {stats.archives_skipped}, reopened {stats.archives_reopened}, "
        f"corrupt {stats.archives_corrupt})"
    )
    click.echo(
        f"ingested {stats.ingested} / duplicates {stats.duplicates} / "
        f"deferred {stats.deferred} / trash {stats.skipped_trash} / "
        f"failed {stats.failed}"
    )
    if stats.missing_metadata:
        # Deliberately says "organized", not "ingested". The line above treats
        # `ingested` and `duplicates` as separate categories, but this counter
        # spans both -- and on a re-run of a multi-part export most members
        # arrive as duplicates, so reusing "ingested" here would make an
        # operator badly over-attribute the metadata gap to freshly-copied
        # files.
        click.echo(f"{stats.missing_metadata} organized without Google metadata")

    if stats.failed or stats.archives_corrupt:
        sys.exit(1)


@takeout_cmd.command(name="status")
@click.option(
    "--catalog",
    "catalog_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the SQLite catalog.",
)
def takeout_status(catalog_path: Path) -> None:
    """Report Takeout ingestion progress."""
    with Catalog(catalog_path) as cat:
        counts = cat.takeout_status_counts()

    archives = counts["archives"]
    total = sum(archives.values())
    detail = ", ".join(f"{n} {status}" for status, n in sorted(archives.items()))
    click.echo(f"{total} archive{'s' if total != 1 else ''}: {detail or 'none'}")

    members = counts["members"]
    member_detail = ", ".join(f"{n} {status}" for status, n in sorted(members.items()))
    click.echo(f"members: {member_detail or 'none'}")

    if counts["missing_metadata"]:
        click.echo(f"{counts['missing_metadata']} members missing Google metadata")


# Alias so `imageharbor takeout ingest` works
main.add_command(takeout_cmd, name="takeout")


# ---------------------------------------------------------------------------
# faces
# ---------------------------------------------------------------------------


def _faces_catalog_path(dest: Path, catalog_path: Path | None) -> Path:
    return catalog_path if catalog_path is not None else dest / "catalog.db"


def _faces_model_dir(model_dir: Path | None, dest: Path | None) -> Path:
    """Resolve a model directory: explicit flag, then env var, then <dest>/.faces-models.

    Model weights are not library data -- $IMAGEHARBOR_FACE_MODEL_DIR exists
    precisely so one download can be shared across libraries and containers
    (see the docker-compose model volume) -- but falling back to
    <dest>/.faces-models means a first run needs nothing beyond --dest.
    """
    if model_dir is not None:
        return model_dir
    env = os.environ.get("IMAGEHARBOR_FACE_MODEL_DIR")
    if env:
        return Path(env)
    if dest is not None:
        return dest / ".faces-models"
    raise click.ClickException(
        "no model directory given: pass --model-dir, set "
        "IMAGEHARBOR_FACE_MODEL_DIR, or pass --dest"
    )


def _require_onnx() -> None:
    """Gate a subcommand on the optional 'faces' extra.

    Reads `faces_pkg.HAS_ONNX` at call time (not imported as a bare name at
    module scope) so a test's `monkeypatch.setattr(faces_pkg, "HAS_ONNX",
    False)` is actually visible here.

    `scan` needs this because it constructs `Detector`/`Embedder`, which
    import onnxruntime directly. `cluster`, `calibrate`, and `status` never
    touch onnxruntime themselves, but they all open a `FaceStore`, and
    `store.py` imports numpy unconditionally -- numpy ships only inside the
    `faces` extra in pyproject.toml, not as a core dependency, so those three
    cannot actually run without the extra either. `HAS_ONNX` is the only
    importability signal the package exposes, and `uv sync --extra faces`
    installs numpy and onnxruntime together, so gating on it here turns a
    raw `ModuleNotFoundError` into the same clear, actionable message
    instead. Only `models download` (pure hashlib/urllib) is genuinely
    extra-free and is exempt.
    """
    from . import faces as faces_pkg

    if not faces_pkg.HAS_ONNX:
        raise click.ClickException(
            "face models need the optional 'faces' extra: "
            "uv sync --extra faces"
        )


@click.group()
def faces_cmd() -> None:
    """Detect faces, group them, and propose names.

    Runs entirely in-process with no AI backend and no network beyond a
    one-time model download. Faces never rename or move a file, and no name is
    written to a photo until a human confirms that cluster on the dashboard.
    """


@faces_cmd.command(name="scan")
@click.option(
    "--dest",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Root of the organized library.",
)
@click.option(
    "--catalog",
    "catalog_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the SQLite catalog. Defaults to <dest>/catalog.db.",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Stop after scanning this many photos.",
)
@click.option(
    "--model-dir",
    "model_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory holding face model weights. Defaults to "
         "$IMAGEHARBOR_FACE_MODEL_DIR or <dest>/.faces-models.",
)
@click.option(
    "--min-score",
    default=0.6,
    show_default=True,
    type=float,
    help="Reject a detection below this confidence.",
)
@click.option(
    "--min-box",
    default=32,
    show_default=True,
    type=int,
    help="Reject a detection smaller than this many pixels on its short side.",
)
def faces_scan(
    dest: Path,
    catalog_path: Path | None,
    limit: int | None,
    model_dir: Path | None,
    min_score: float,
    min_box: int,
) -> None:
    """Detect and embed faces in every organized photo not yet scanned.

    Runs the default detector and embedder (see `imageharbor.faces.models`).
    Never renames or moves a photo, and writes no name -- only face geometry
    and embeddings. Resumable at one-photo granularity: a re-run is a no-op
    for anything already scanned by the same detector.
    """
    _require_onnx()

    from .faces.detect import Detector
    from .faces.embed import Embedder
    from .faces.runner import QualityGate, scan
    from .faces.store import FaceStore

    catalog_path = _faces_catalog_path(dest, catalog_path)
    resolved_model_dir = _faces_model_dir(model_dir, dest)
    # <catalog_dir>/face-crops, per the design spec's "Crop cache" section --
    # a derived, deletable cache that belongs on the catalog volume, not
    # necessarily the (possibly NAS-mounted) --dest tree.
    crop_dir = catalog_path.parent / "face-crops"

    detector = Detector(resolved_model_dir)
    embedder = Embedder(resolved_model_dir)
    gate = QualityGate(min_score=min_score, min_box=min_box)

    with Catalog(catalog_path) as catalog, FaceStore(catalog_path) as store:
        result = scan(
            catalog, store, detector, embedder, crop_dir, gate=gate, limit=limit
        )

    click.echo(
        f"scanned={result.scanned} faces={result.faces} "
        f"rejected={result.rejected} errors={result.errors}"
    )


@faces_cmd.command(name="cluster")
@click.option(
    "--dest",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Root of the organized library.",
)
@click.option(
    "--catalog",
    "catalog_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the SQLite catalog. Defaults to <dest>/catalog.db.",
)
@click.option(
    "--threshold",
    required=True,
    type=float,
    help="Cosine-similarity threshold at or above which two faces join a "
         "cluster. Measure this first with `faces calibrate` -- never guess it.",
)
@click.option(
    "--min-score",
    default=0.6,
    show_default=True,
    type=float,
    help="Minimum score for a name proposal to be recorded.",
)
@click.option(
    "--min-support",
    default=2,
    show_default=True,
    type=int,
    help="Minimum supporting photos for a name proposal.",
)
@click.option(
    "--recluster",
    is_flag=True,
    default=False,
    help="Rebuild clustering that already exists for this library. Required "
         "once a prior `cluster` run has produced clusters, so a re-run is "
         "never silently destructive (any already-confirmed person is kept).",
)
def faces_cluster(
    dest: Path,
    catalog_path: Path | None,
    threshold: float,
    min_score: float,
    min_support: int,
    recluster: bool,
) -> None:
    """Group scanned faces into clusters and propose names from Google tags.

    Never assigns a person to a cluster -- a proposal is only written to the
    `proposals` table for a human to confirm on the dashboard.
    """
    _require_onnx()

    from .faces import models as face_models
    from .faces.runner import build_clusters, google_names
    from .faces.store import FaceStore

    catalog_path = _faces_catalog_path(dest, catalog_path)
    embed_model = face_models.DEFAULT_EMBEDDER

    with FaceStore(catalog_path) as store:
        existing = store.cluster_ids()
        if existing and not recluster:
            raise click.ClickException(
                f"{len(existing)} cluster(s) already exist; pass --recluster "
                "to rebuild them from scratch (any confirmed person is kept)."
            )

        photo_names = google_names(dest)
        made = build_clusters(
            store,
            photo_names,
            embed_model=embed_model,
            threshold=threshold,
            min_score=min_score,
            min_support=min_support,
        )
        proposals = sum(len(store.proposals_for(cid)) for cid in store.cluster_ids())

    click.echo(f"clusters={made} proposals={proposals}")


@faces_cmd.command(name="calibrate")
@click.option(
    "--dest",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Root of the organized library.",
)
@click.option(
    "--catalog",
    "catalog_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the SQLite catalog. Defaults to <dest>/catalog.db.",
)
@click.option(
    "--target-precision",
    default=0.99,
    show_default=True,
    type=float,
    help="Lowest acceptable fraction of same-name pairs at the chosen threshold.",
)
def faces_calibrate(dest: Path, catalog_path: Path | None, target_precision: float) -> None:
    """Measure the clustering threshold from the library's own Google-tagged photos.

    Precision here is over anchor pairs (photos with exactly one detected
    face and exactly one Google-tagged name). Run this after `scan` and
    before `cluster` -- the threshold must be measured, never guessed.
    """
    _require_onnx()

    from .faces import models as face_models
    from .faces.runner import google_names, measure_threshold
    from .faces.store import FaceStore

    catalog_path = _faces_catalog_path(dest, catalog_path)
    photo_names = google_names(dest)

    with FaceStore(catalog_path) as store:
        result = measure_threshold(
            store,
            photo_names,
            embed_model=face_models.DEFAULT_EMBEDDER,
            target_precision=target_precision,
        )

    click.echo(
        f"threshold={result.threshold:.4f} precision={result.precision:.4f} "
        f"recall={result.recall:.4f}"
    )
    click.echo(
        f"Next: imageharbor faces cluster --dest {dest} "
        f"--threshold {result.threshold:.4f}"
    )


@faces_cmd.command(name="status")
@click.option(
    "--dest",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Root of the organized library.",
)
@click.option(
    "--catalog",
    "catalog_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the SQLite catalog. Defaults to <dest>/catalog.db.",
)
def faces_status(dest: Path, catalog_path: Path | None) -> None:
    """Report face-scanning and review progress."""
    _require_onnx()

    from .faces.store import FaceStore

    catalog_path = _faces_catalog_path(dest, catalog_path)
    with FaceStore(catalog_path) as store:
        stats = store.stats()

    for key in ("faces", "scanned", "clusters", "people", "unreviewed", "singletons"):
        click.echo(f"{key:<12} {stats[key]}")


@faces_cmd.group(name="models")
def faces_models_cmd() -> None:
    """Manage local face model weights."""


@faces_models_cmd.command(name="download")
@click.option(
    "--dest",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Root of an organized library. Only used to derive a default "
         "--model-dir (<dest>/.faces-models) when neither --model-dir nor "
         "$IMAGEHARBOR_FACE_MODEL_DIR is set -- model weights are not "
         "library-specific, so this is optional here (unlike the other "
         "faces subcommands, where it is required).",
)
@click.option(
    "--model-dir",
    "model_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to store weights in. Defaults to "
         "$IMAGEHARBOR_FACE_MODEL_DIR or <dest>/.faces-models.",
)
def faces_models_download(dest: Path | None, model_dir: Path | None) -> None:
    """Download and verify the default detector and embedder weights.

    Does not require the 'faces' extra: this only fetches and checksums
    files (hashlib/urllib), it never imports onnxruntime or numpy.
    """
    from .faces import models as face_models
    from .faces.download import ensure

    resolved_model_dir = _faces_model_dir(model_dir, dest)
    detector_path = ensure(face_models.DETECTORS[face_models.DEFAULT_DETECTOR], resolved_model_dir)
    embedder_path = ensure(face_models.EMBEDDERS[face_models.DEFAULT_EMBEDDER], resolved_model_dir)

    click.echo(f"detector: {detector_path}")
    click.echo(f"embedder: {embedder_path}")


# Alias so `imageharbor faces scan` works
main.add_command(faces_cmd, name="faces")


# ---------------------------------------------------------------------------
# catalog query
# ---------------------------------------------------------------------------


@click.group()
def catalog_cmd() -> None:
    """Query the photo catalog."""


@catalog_cmd.command(name="list")
@click.option(
    "--catalog",
    "catalog_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the SQLite catalog.",
)
@click.option("--limit", default=50, show_default=True, help="Maximum rows to display.")
def catalog_list(catalog_path: Path, limit: int) -> None:
    """List photos in the catalog."""
    rows = []
    with Catalog(catalog_path) as cat:
        for i, row in enumerate(cat.iter_all()):
            if i >= limit:
                break
            rows.append(row)
    if not rows:
        click.echo("(empty catalog)")
        return
    for row in rows:
        # An unenriched row was never classified -- pcs_primary is NULL, and
        # showing it as a bare "None"/"900" would assert a classification
        # that was never made. Render "—" for exactly that case.
        pcs_display = "—" if row["enriched_at"] is None else str(row["pcs_primary"])
        click.echo(f"{row['sha256_b64url'][:12]}…  {pcs_display:>5}  {row['organized_path']}")


@catalog_cmd.command(name="get")
@click.option(
    "--catalog",
    "catalog_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the SQLite catalog.",
)
@click.argument("sha256")
def catalog_get(catalog_path: Path, sha256: str) -> None:
    """Look up a photo by its SHA-256 Base64url digest."""
    import json as _json

    from .catalog import _from_json

    with Catalog(catalog_path) as cat:
        row = cat.get_by_sha256(sha256)
    if row is None:
        click.echo(f"Not found: {sha256}", err=True)
        sys.exit(1)
    data = dict(row)
    # These columns are stored as JSON TEXT; decode them so the output shows
    # structured values instead of escaped strings.
    for col in ("secondary_tags", "objects", "exif", "processing_history"):
        if col in data and isinstance(data[col], str):
            data[col] = _from_json(data[col])
    click.echo(_json.dumps(data, indent=2, ensure_ascii=False))


# Alias so `imageharbor catalog list` works
main.add_command(catalog_cmd, name="catalog")


# ---------------------------------------------------------------------------
# sidecar
# ---------------------------------------------------------------------------


@click.group()
def sidecar_cmd() -> None:
    """Operate on sidecar files for an already-organized library."""


@sidecar_cmd.command(name="backfill")
@click.option(
    "--dest",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Root of the organized library.",
)
@click.option(
    "--catalog",
    "catalog_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the SQLite catalog. Defaults to <dest>/catalog.db.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Report what would be written without writing anything.",
)
def sidecar_backfill(dest: Path, catalog_path: Path | None, dry_run: bool) -> None:
    """Rebuild/merge sidecars for an already-organized library from DEST's catalog.

    For libraries organized before sidecars were the default (see `process
    --no-sidecar`), this writes a sidecar built from what the catalog holds:
    identity, sources, date and descriptor with their tiers, and a fresh EXIF
    read from each organized copy. A file that already has a sidecar is
    merged into, not skipped -- the merge is a no-op if the sidecar was
    already complete.

    Google Takeout metadata is NOT recoverable this way: `provenance` stays
    empty for backfilled files, because the original archive documents were
    never stored for them. Recovering that means re-ingesting the archives.
    """
    from .backfill import backfill_sidecars

    if catalog_path is None:
        catalog_path = dest / "catalog.db"

    with Catalog(catalog_path) as catalog:
        stats = backfill_sidecars(dest, catalog, dry_run=dry_run)

    if dry_run:
        click.echo("[DRY-RUN] No files were written.")
    click.echo(
        f"Cataloged={stats.cataloged}  Written={stats.written}  "
        f"Unchanged={stats.unchanged}  Failed={stats.failed}"
    )

    if stats.failed:
        sys.exit(1)


# Alias so `imageharbor sidecar backfill` works
main.add_command(sidecar_cmd, name="sidecar")
