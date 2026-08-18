"""Rebuild sidecars for an already-organized library from the catalog.

What this can write is bounded by what the catalog holds: identity, sources,
date and descriptor with their tiers, plus a fresh EXIF read from the
organized copy. **Google Takeout metadata is not recoverable this way** --
`provenance[]` stays empty for backfilled files, because the original archive
documents were never stored. Recovering those means re-ingesting the
archives.

For a row that already has a sidecar, backfill *merges* into it rather than
skipping it -- a sidecar written before this verb existed is otherwise
permanently thin, and `sidecar.merge_sidecar`'s append-only rule makes
merging an already-complete sidecar safe (it is a no-op by construction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .catalog import Catalog
from .date_resolver import date_from_row
from .exif_reader import read_exif
from .pipeline import _source_entry
from .relocate import resolve_organized_path
from .sidecar import merge_sidecar, read_sidecar, sidecar_path_for
from .sidecar_schema import merge as merge_documents

logger = logging.getLogger(__name__)


@dataclass
class BackfillStats:
    """Outcome of a backfill run.

    There is deliberately no separate ``missing`` counter: a row whose
    organized file cannot be resolved and a row that fails to write both have
    the same remedy for the operator (investigate the file), so both count
    toward ``failed``.
    """

    cataloged: int = 0
    written: int = 0
    unchanged: int = 0
    failed: int = 0


def _already_recorded_at_or_above(existing: dict, key: str, catalog_tier: int) -> bool:
    """True when *existing*'s sidecar already carries *key* at >= *catalog_tier*.

    A block a first-hand read (the facts pass, or a duplicate-upgrade re-merge)
    already wrote must never be re-asserted by backfill, which only ever
    reconstructs from the catalog's lossy columns -- see `_build_updates`.
    """
    block = existing.get(key)
    if not isinstance(block, dict):
        return False
    return (block.get("tier") or 0) >= catalog_tier


def _build_updates(organized_path: Path, row, existing: dict) -> dict:
    """The same update dict the facts pass builds for a fresh sidecar,
    rebuilt here from what the catalog and the organized copy still hold.

    *existing* is the sidecar already on disk (possibly ``{}``). A `date` or
    `descriptor` block it already carries at >= the catalog's tier is a
    first-hand observation backfill must not re-assert -- see the comment on
    the `date` block below for why re-asserting the date specifically would
    corrupt the record.
    """
    date = date_from_row(row)
    size = organized_path.stat().st_size
    extension = organized_path.suffix.lstrip(".").lower()
    exif_data = read_exif(organized_path)
    updates: dict = {
        "identity": {
            "sha256_b64url": row["sha256_b64url"],
            "size": size,
            "ext": extension,
        },
        "sources": [],  # populated by the caller, which has the catalog handle
        "exif": exif_data,
    }

    if not _already_recorded_at_or_above(existing, "date", date.tier):
        # The catalog stores a date STRING, so `date_from_row` reconstructs
        # midnight for the time-of-day it never held. Writing that as an
        # observation would assert precision nothing ever measured -- the
        # same error the date ladder refuses when it declines to read mtime
        # -- and history is never pruned, so it would be permanent. Backfill
        # therefore writes the date at the precision the catalog holds, and
        # defers entirely to any block a first-hand read already wrote.
        updates["date"] = {
            "value": date.date_str,
            "tier": date.tier,
            "source": date.source,
        }

    descriptor_tier = row["descriptor_tier"] or 0
    if not _already_recorded_at_or_above(existing, "descriptor", descriptor_tier):
        updates["descriptor"] = {
            "value": row["descriptor_value"] or "",
            "tier": descriptor_tier,
            "source": row["descriptor_source"] or "none",
        }

    return updates


def backfill_sidecars(
    organized_dir: Path, catalog: Catalog, *, dry_run: bool = False
) -> BackfillStats:
    """Rebuild/merge a sidecar for every cataloged photo under *organized_dir*.

    Iterates ``catalog.iter_all()``. For each row the organized file is
    located with `relocate.resolve_organized_path` (so a file moved since it
    was cataloged is found by digest rather than reported missing), a fresh
    update dict is built from the catalog row and the organized copy's own
    EXIF, and merged into the file's sidecar with `sidecar.merge_sidecar`.

    A row counts as ``written`` when the sidecar's bytes changed and
    ``unchanged`` when the merge changed nothing -- determined by comparing
    bytes rather than by asking the merge policy to predict its own output.
    In ``dry_run`` mode nothing is written; the same comparison is made
    against the merge's in-memory result instead, so the reported counts are
    still accurate.

    A row whose organized file cannot be resolved, or that fails to write,
    is counted in ``failed`` and does not stop the run.
    """
    stats = BackfillStats()

    for row in catalog.iter_all():
        stats.cataloged += 1
        try:
            _backfill_row(organized_dir, catalog, row, dry_run=dry_run, stats=stats)
        except Exception:
            logger.exception(
                "Backfill failed for %s", row["sha256_b64url"]
            )
            stats.failed += 1

    return stats


def _backfill_row(
    organized_dir: Path,
    catalog: Catalog,
    row,
    *,
    dry_run: bool,
    stats: BackfillStats,
) -> None:
    recorded = Path(row["organized_path"]) if row["organized_path"] else None
    if recorded is None:
        logger.warning("No organized_path on record for %s", row["sha256_b64url"])
        stats.failed += 1
        return

    actual = resolve_organized_path(organized_dir, recorded, row["sha256_b64url"])
    if actual is None:
        logger.warning(
            "Cannot backfill %s: organized file missing", row["sha256_b64url"]
        )
        stats.failed += 1
        return

    # `quarantine=False` in dry-run mode: a dry run inspects the existing
    # sidecar to decide what it WOULD write, and must not itself perform a
    # write (a corrupt-file quarantine rename) as a side effect of looking.
    existing = read_sidecar(actual, quarantine=not dry_run)
    updates = _build_updates(actual, row, existing)
    updates["sources"] = [
        _source_entry(r) for r in catalog.sources_for(row["sha256_b64url"])
    ]

    if dry_run:
        merged = merge_documents(existing, updates, observed_at="dry-run")
        if merged == existing:
            stats.unchanged += 1
        else:
            stats.written += 1
        return

    sidecar_path = sidecar_path_for(actual)
    before = sidecar_path.read_bytes() if sidecar_path.exists() else None
    merge_sidecar(actual, updates)
    after = sidecar_path.read_bytes()

    if before == after:
        stats.unchanged += 1
    else:
        stats.written += 1
