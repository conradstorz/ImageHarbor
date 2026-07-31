"""Command-line interface for ImageHarbor."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from . import __version__
from .catalog import Catalog
from .hashing import extract_digest_from_stem, verify_pcs_file
from .pipeline import Pipeline


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
    default=False,
    show_default=True,
    help="Write a JSON sidecar alongside each organized image.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Plan operations without writing any files.",
)
@click.option(
    "--ai",
    "ai_backend",
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
def process(
    source: Path,
    dest: Path,
    catalog_path: Path | None,
    duplicates_dir: Path | None,
    sidecar: bool,
    dry_run: bool,
    ai_backend: str,
    ai_base_url: str | None,
    ai_model: str,
    ai_timeout: float,
    openai_key: str | None,
    no_recursive: bool,
) -> None:
    """Discover, classify, copy and catalog photos from SOURCE to DEST."""
    if catalog_path is None:
        catalog_path = dest / "catalog.db"

    classifier = _build_classifier(ai_backend, openai_key, ai_base_url, ai_model, ai_timeout)

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
            classifier=classifier,
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
    default=False,
    show_default=True,
    help="Write a JSON sidecar alongside each organized image.",
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
) -> None:
    """Continuously watch SOURCE and organize new/changed photos into DEST."""
    import signal
    import threading

    from . import watcher as _watcher

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
            classifier=classifier,
            duplicates_dir=duplicates_dir,
            write_sidecars=sidecar,
        )
        click.echo(f"Watching {source} -> {dest} every {interval:.0f}s (Ctrl-C to stop).")
        stats = _watcher.watch(
            pipeline=pipeline,
            catalog=catalog,
            source=source,
            interval=interval,
            recursive=not no_recursive,
            stop_event=stop_event,
        )

    click.echo(
        f"Stopped after {stats.passes} pass(es). "
        f"Processed={stats.processed} Skipped={stats.skipped_unchanged} Errors={stats.errors}"
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
    """Verify PCS filename integrity for PATH (file or directory)."""
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
            # Not a PCS-named file; skip silently
            skip_count += 1
            continue
        if verify_pcs_file(target):
            ok_count += 1
            click.echo(f"OK   {target}")
        else:
            fail_count += 1
            click.echo(f"FAIL {target}", err=True)

    click.echo(
        f"\nVerified {ok_count + fail_count} PCS image(s) "
        f"({skip_count} non-image/non-PCS skipped): {ok_count} OK, {fail_count} FAILED"
    )
    if ok_count + fail_count == 0:
        click.echo("No PCS-format image files found to verify.", err=True)
        sys.exit(1)
    if fail_count:
        sys.exit(1)


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
        click.echo(f"{row['sha256_b64url'][:12]}…  {row['pcs_primary']:3d}  {row['organized_path']}")


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
