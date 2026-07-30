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
    "--openai-key",
    default=None,
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (or set OPENAI_API_KEY env var).",
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
    openai_key: str | None,
    no_recursive: bool,
) -> None:
    """Discover, classify, copy and catalog photos from SOURCE to DEST."""
    if catalog_path is None:
        catalog_path = dest / "catalog.db"

    # Build classifier
    if ai_backend == "openai":
        from .ai_classifier import OpenAIClassifier

        classifier = OpenAIClassifier(api_key=openai_key)
    else:
        from .ai_classifier import StubClassifier

        classifier = StubClassifier()

    dest.mkdir(parents=True, exist_ok=True)

    with Catalog(catalog_path) as catalog:
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
    with Catalog(catalog_path) as cat:
        rows = list(cat.iter_all())
    rows = rows[:limit]
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

    with Catalog(catalog_path) as cat:
        row = cat.get_by_sha256(sha256)
    if row is None:
        click.echo(f"Not found: {sha256}", err=True)
        sys.exit(1)
    data = dict(row)
    click.echo(_json.dumps(data, indent=2, ensure_ascii=False))


# Alias so `imageharbor catalog list` works
main.add_command(catalog_cmd, name="catalog")
