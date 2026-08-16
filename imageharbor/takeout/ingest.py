"""Two-phase Google Takeout ingestion.

The only module in this package with side effects. Its shape mirrors
``enrich.enrich_library``: iterate a work queue held in the catalog, do one
unit of work, commit the outcome, continue.

Phase 1 surveys every archive by reading central directories only, and builds
ONE pairing index across the whole batch. Phase 2 ingests member by member,
committing after each, so a ``kill -9`` costs one member's work.

This pass makes no AI calls, exactly like the facts pass, so there is nothing
for a circuit breaker to observe and it must not touch one.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ..catalog import Catalog
from ..pipeline import ExternalEvidence, Pipeline
from ..sidecar import merge_sidecar
from . import archive, metadata, pairing

logger = logging.getLogger(__name__)

STAGING_DIR_NAME = ".takeout-staging"

# Member statuses.
_PENDING = "pending"
_INGESTED = "ingested"
_DUPLICATE = "duplicate"
_DEFERRED = "deferred"
_PARSED = "parsed"
_IGNORED = "ignored"
_SKIPPED_TRASH = "skipped_trash"
_FAILED = "failed"


@dataclass
class IngestStats:
    """Aggregated outcome of one ingest run."""

    archives_seen: int = 0
    archives_skipped: int = 0     # already complete, nothing new to do
    archives_reopened: int = 0    # were complete; new work appeared (see _survey)
    archives_corrupt: int = 0
    ingested: int = 0
    duplicates: int = 0
    deferred: int = 0             # videos: enumerated and dated, never copied
    skipped_trash: int = 0
    failed: int = 0
    # Counted for duplicates as well as fresh ingests: the fact worth
    # reporting is "this member had no Google metadata", not how its bytes
    # happened to reach the library.
    missing_metadata: int = 0     # organized, but no sidecar could be paired
    per_archive: list[dict] = field(default_factory=list)


def _initial_status(member: archive.MemberInfo, include_trash: bool) -> str:
    if archive.is_trash(member.path) and not include_trash:
        return _SKIPPED_TRASH
    if member.kind in (archive.KIND_IMAGE, archive.KIND_VIDEO):
        return _PENDING
    if member.kind in (archive.KIND_METADATA, archive.KIND_ALBUM):
        return _PARSED    # terminal: read on demand, never a work item
    return _IGNORED


def ingest_archives(
    archives_dir: Path,
    organized_dir: Path,
    catalog: Catalog,
    *,
    include_trash: bool = False,
    write_sidecars: bool = False,
    dry_run: bool = False,
) -> IngestStats:
    """Ingest every ``*.zip`` under *archives_dir* into *organized_dir*."""
    return _Ingestor(
        archives_dir=archives_dir,
        organized_dir=organized_dir,
        catalog=catalog,
        include_trash=include_trash,
        write_sidecars=write_sidecars,
        dry_run=dry_run,
    ).run()


class _Ingestor:
    """Holds the state the two phases share."""

    def __init__(
        self,
        *,
        archives_dir: Path,
        organized_dir: Path,
        catalog: Catalog,
        include_trash: bool,
        write_sidecars: bool,
        dry_run: bool,
    ) -> None:
        self.archives_dir = archives_dir
        self.organized_dir = organized_dir
        self.catalog = catalog
        self.include_trash = include_trash
        self.write_sidecars = write_sidecars
        self.dry_run = dry_run
        self.staging_dir = organized_dir / STAGING_DIR_NAME
        self.stats = IngestStats()
        # Which archive holds each member, so a sidecar in another part can be
        # read without re-surveying.
        self.owner: dict[str, Path] = {}
        self.pairing_index = pairing.PairingIndex()
        self.pipeline = Pipeline(
            source_dir=self.staging_dir,
            organized_dir=organized_dir,
            catalog=catalog,
            write_sidecars=write_sidecars,
            consume_source=True,
        )

    # -- phase 1 ------------------------------------------------------------

    def _survey(self) -> list[tuple[archive.ArchiveIdentity, list[archive.MemberInfo]]]:
        """Enumerate every archive; return the ones with work left to do.

        The member lists come back with the identities so `--dry-run` can report
        from them directly: a dry run writes no `takeout_members` rows, so the
        catalog work queue would read empty and under-report.
        """
        todo: list[tuple[archive.ArchiveIdentity, list[archive.MemberInfo]]] = []
        all_members: list[str] = []
        # Archives already complete, held back until the index exists so the
        # second pass below can tell whether new metadata reached them.
        completed: list[tuple[archive.ArchiveIdentity, list]] = []

        for path in sorted(p for p in self.archives_dir.glob("*.zip") if p.is_file()):
            self.stats.archives_seen += 1
            identity = archive.identify(path, self.catalog)
            row = self.catalog.takeout_archive_get(identity.archive_id)

            if row is not None and row["status"] == "complete":
                reopened = False
                if self.include_trash and not self.dry_run:
                    # A user who changes their mind must not be blocked by the
                    # terminal status an earlier run recorded.
                    moved = self.catalog.takeout_members_unskip_trash(identity.archive_id)
                    if moved:
                        self.catalog.takeout_archive_set_status(
                            identity.archive_id, "partial"
                        )
                        self.stats.archives_reopened += 1
                        reopened = True
                if not reopened:
                    # A complete archive still contributes its member paths to
                    # the pairing index, and is re-examined in the second pass
                    # below once the index exists. Member paths come from the
                    # catalog, so no zip is opened and nothing is decompressed.
                    rows = self.catalog.takeout_members_all(identity.archive_id)
                    for member in rows:
                        all_members.append(member["member_path"])
                        self.owner[member["member_path"]] = path
                    completed.append((identity, rows))
                    continue

            try:
                with zipfile.ZipFile(path, "r") as zf:
                    members = list(archive.iter_members(zf))
            except (zipfile.BadZipFile, OSError) as exc:
                logger.error("Archive %s is unreadable: %s", path.name, exc)
                self.stats.archives_corrupt += 1
                if not self.dry_run:
                    self.catalog.takeout_archive_upsert(
                        archive_id=identity.archive_id,
                        last_path=str(path),
                        size=identity.size,
                        mtime_ns=identity.mtime_ns,
                        status="corrupt",
                        last_error=str(exc),
                    )
                continue

            if not self.dry_run:
                self.catalog.takeout_archive_upsert(
                    archive_id=identity.archive_id,
                    last_path=str(path),
                    size=identity.size,
                    mtime_ns=identity.mtime_ns,
                    member_count=len(members),
                    status="partial",
                )
                for member in members:
                    self.catalog.takeout_member_add(
                        archive_id=identity.archive_id,
                        member_path=member.path,
                        kind=member.kind,
                        size=member.size,
                        crc32=member.crc32,
                        status=_initial_status(member, self.include_trash),
                    )

            for member in members:
                all_members.append(member.path)
                self.owner[member.path] = path
                if _initial_status(member, self.include_trash) == _SKIPPED_TRASH:
                    self.stats.skipped_trash += 1

            todo.append((identity, members))

        # ONE index across every archive in the batch. Google's multi-part
        # zips split by size across the file list, so a photo and its sidecar
        # routinely land in different parts; a per-archive index would silently
        # lose metadata at every part boundary.
        self.pairing_index = pairing.build_index(all_members)

        # SECOND PASS -- the late-sidecar case, and the reason the survey has
        # two passes at all.
        #
        # Contributing a complete archive's member paths to the index (above)
        # only solves ONE ordering: sidecars ingested first, the photos they
        # describe arriving later. The opposite order is the common one -- you
        # download part 1, ingest it, and its photos land in Undated/ because
        # their sidecars are in part 2 which you have not downloaded yet. When
        # part 2 arrives, part 1 is `complete`, so without this pass nothing
        # would ever revisit those photos and they would stay Undated/ forever.
        #
        # A member is reopened only when it demonstrably gained something: it
        # is an image or a video, it has no sidecar on record, and the index
        # NOW resolves one for it. Re-ingesting it makes its bytes hash as a
        # duplicate, which routes through `_maybe_upgrade_from_duplicate` and
        # relocates the file -- no new placement code, exactly as designed.
        # The check is pure string work against an in-memory index, so an
        # archive that never gains metadata costs nothing to re-examine.
        for identity, rows in completed:
            stale = [
                row for row in rows
                if row["kind"] in (archive.KIND_IMAGE, archive.KIND_VIDEO)
                # Whitelist, not a blacklist. Only a member whose media work
                # actually COMPLETED may be reopened. `skipped_trash` is the
                # case that makes this load-bearing: real exports give trash
                # items sidecars, so a trash member is pairable, and without
                # this clause the second run of the very same command would
                # pour the deleted-photos tree into the library -- bypassing
                # --include-trash entirely and reporting nothing. `parsed`,
                # `ignored`, `pending`, and `failed` are excluded for the same
                # reason: none of them means "organized, but missing metadata".
                and row["status"] in (_INGESTED, _DUPLICATE, _DEFERRED)
                and not row["sidecar_path"]
                and pairing.sidecar_for(row["member_path"], self.pairing_index) is not None
            ]
            if not stale:
                self.stats.archives_skipped += 1
                continue

            self.stats.archives_reopened += 1
            members = [
                archive.MemberInfo(
                    path=row["member_path"], size=row["size"],
                    crc32=row["crc32"], kind=row["kind"],
                )
                for row in stale
            ]
            if self.dry_run:
                # Report the work without performing or recording any of it.
                todo.append((identity, members))
                continue

            for row in stale:
                # Every field this row already holds must be passed back:
                # `takeout_member_set` is a blind full-row UPDATE and would
                # otherwise null what it does not receive. `sidecar_path` is
                # deliberately left None -- re-ingesting is what establishes it.
                self.catalog.takeout_member_set(
                    identity.archive_id,
                    row["member_path"],
                    status=_PENDING,
                    sha256_b64url=row["sha256_b64url"],
                    taken_at=row["taken_at"],
                    sidecar_path=None,
                    last_error=row["last_error"],
                )
            self.catalog.takeout_archive_set_status(identity.archive_id, "partial")
            todo.append((identity, members))

        return todo

    # -- phase 2 ------------------------------------------------------------

    def _read_sidecar(self, sidecar_member: str) -> metadata.TakeoutMetadata:
        owner = self.owner.get(sidecar_member)
        if owner is None:
            return metadata.EMPTY
        try:
            with zipfile.ZipFile(owner, "r") as zf:
                return metadata.parse_photo_metadata(archive.read_member(zf, sidecar_member))
        except (zipfile.BadZipFile, KeyError, OSError) as exc:
            logger.warning("Unreadable sidecar %s (%s); ingesting without it",
                           sidecar_member, exc)
            return metadata.EMPTY

    def _label(self, archive_path: Path, member_path: str) -> str:
        return f"{archive_path}!{member_path}"

    def _mark_failed(self, identity: archive.ArchiveIdentity, row, error: str) -> None:
        """Record a member as failed WITHOUT discarding what it already knew.

        `takeout_member_set` is a blind full-row UPDATE, so a failure branch
        that passes only `status` and `last_error` nulls `sha256_b64url`,
        `taken_at`, and `sidecar_path`. That matters most for a member the
        second survey pass just reopened: the reopen deliberately carried
        those values forward, and a failed retry would silently throw them
        away. The values self-heal on the next successful attempt, but a
        `failed` row should still describe what is actually known.

        A failure of THIS write is deliberately left to propagate. The catalog
        is the work queue, so a catalog that cannot be written means every
        subsequent member would do work that can never be recorded or resumed
        -- aborting loudly is the honest outcome, and swallowing it would let a
        run appear to succeed while recording nothing. It is stated here
        because the abort would otherwise look accidental.
        """
        self.catalog.takeout_member_set(
            identity.archive_id,
            row["member_path"],
            status=_FAILED,
            sha256_b64url=row["sha256_b64url"],
            taken_at=row["taken_at"],
            sidecar_path=row["sidecar_path"],
            last_error=error,
        )

    def _ingest_image(
        self,
        zf: zipfile.ZipFile,
        identity: archive.ArchiveIdentity,
        row,
    ) -> None:
        member_path = row["member_path"]
        member = archive.MemberInfo(
            path=member_path, size=row["size"], crc32=row["crc32"], kind=row["kind"]
        )
        sidecar_member = pairing.sidecar_for(member_path, self.pairing_index)
        meta = self._read_sidecar(sidecar_member) if sidecar_member else metadata.EMPTY

        staged = None
        try:
            staged = archive.extract_to(zf, member, self.staging_dir)
            result = self.pipeline.process_file(
                staged,
                source_label=self._label(identity.path, member_path),
                evidence=ExternalEvidence(
                    date=meta.photo_taken_at, original_name=meta.title
                ),
            )
        except Exception as exc:  # a member failure never fails the archive
            logger.warning("Failed to ingest %s: %s", member_path, exc, exc_info=True)
            self.stats.failed += 1
            self._mark_failed(identity, row, str(exc))
            return
        finally:
            if staged is not None:
                archive.discard_staged(staged)

        if result.status == "error":
            self.stats.failed += 1
            self._mark_failed(identity, row, result.error)
            return

        status = _DUPLICATE if result.status == "duplicate" else _INGESTED
        if status == _DUPLICATE:
            self.stats.duplicates += 1
        else:
            self.stats.ingested += 1
        if sidecar_member is None:
            self.stats.missing_metadata += 1

        # The member is marked terminal only AFTER process_file returned a
        # non-error result, so the copy -> verify -> catalog ordering remains
        # the sole arbiter of truth and takeout_members can only lag it.
        # `last_error` is deliberately omitted: this member just succeeded,
        # so clearing any error from a previous attempt is correct.
        self.catalog.takeout_member_set(
            identity.archive_id,
            member_path,
            status=status,
            sha256_b64url=result.sha256_b64url,
            taken_at=meta.photo_taken_at.isoformat() if meta.photo_taken_at else None,
            sidecar_path=sidecar_member,
        )

        if self.write_sidecars:
            # A duplicate leaves `result.organized_path` None -- the file it
            # matched already lives somewhere else, and the pipeline does not
            # repeat that path back on the duplicate branch. The late-sidecar
            # reopen (this module's headline case) IS a duplicate branch: the
            # member re-ingests, hashes as a duplicate of itself, and
            # `_maybe_upgrade_from_duplicate` relocates it -- so without this
            # fallback the photo gets relocated but its Google provenance is
            # silently never recorded. The catalog is the source of truth for
            # where a digest actually lives, duplicate or not.
            organized_path = result.organized_path
            if organized_path is None:
                row = self.catalog.get_by_sha256(result.sha256_b64url)
                if row is not None and row["organized_path"] is not None:
                    organized_path = Path(row["organized_path"])
            if organized_path is not None:
                self._merge_takeout_sidecar(organized_path, identity, member_path,
                                            sidecar_member, meta)

    def _merge_takeout_sidecar(
        self, organized_path: Path, identity, member_path: str,
        sidecar_member: str | None, meta: metadata.TakeoutMetadata,
    ) -> None:
        """Record Google's metadata as provenance. None of it is load-bearing.

        A sidecar failure must never fail an image that is already copied,
        verified, and catalogued.
        """
        try:
            merge_sidecar(
                organized_path,
                {
                    "takeout": {
                        "archive": identity.path.name,
                        "archive_id": identity.archive_id,
                        "member": member_path,
                        "sidecar": sidecar_member,
                        # Album membership, recorded not materialized: the
                        # containing directory IS the album in every Takeout
                        # layout. Placement stays date-derived, so this is a
                        # record and never a path.
                        "album": member_path.rpartition("/")[0].rpartition("/")[2] or None,
                        "title": meta.title,
                        "description": meta.description,
                        "photo_taken_time": (
                            meta.photo_taken_at.isoformat() if meta.photo_taken_at else None
                        ),
                        # Provenance only: creationTime is upload time, and is
                        # never allowed to place a file.
                        "creation_time": (
                            meta.creation_at.isoformat() if meta.creation_at else None
                        ),
                        "latitude": meta.latitude,
                        "longitude": meta.longitude,
                        "people": list(meta.people),
                        "favorited": meta.favorited,
                        "google_exif": meta.google_exif,
                    }
                },
            )
        except Exception:
            logger.warning(
                "Failed to write Takeout sidecar block for %s; image is organized "
                "and catalogued", organized_path.name, exc_info=True,
            )

    def _defer_video(self, identity: archive.ArchiveIdentity, row) -> None:
        """Record a video with its capture date. No bytes are copied.

        Wrapped in the same isolation the image path gets. The work here is
        local -- an index lookup, a sidecar parse that cannot raise, and one
        catalog write -- so a failure is unlikely, but "unlikely" is not the
        contract: a member exception must fail that member and let the loop
        continue, never abort the batch.
        """
        member_path = row["member_path"]
        try:
            sidecar_member = pairing.sidecar_for(member_path, self.pairing_index)
            meta = self._read_sidecar(sidecar_member) if sidecar_member else metadata.EMPTY
            # `last_error` is deliberately omitted: this member just succeeded,
            # so clearing any error from a previous attempt is correct.
            self.catalog.takeout_member_set(
                identity.archive_id,
                member_path,
                status=_DEFERRED,
                # Always None in practice -- nothing hashes or copies a video. Passed
                # for call-site symmetry with `_mark_failed`, so every write to this
                # table visibly accounts for every field it could null.
                sha256_b64url=row["sha256_b64url"],
                taken_at=meta.photo_taken_at.isoformat() if meta.photo_taken_at else None,
                sidecar_path=sidecar_member,
            )
        except Exception as exc:
            logger.warning("Failed to defer %s: %s", member_path, exc, exc_info=True)
            self.stats.failed += 1
            self._mark_failed(identity, row, str(exc))
            return
        self.stats.deferred += 1

    def _ingest_archive(self, identity: archive.ArchiveIdentity) -> None:
        pending = self.catalog.takeout_members_pending(identity.archive_id)
        if not pending:
            self.catalog.takeout_archive_set_status(identity.archive_id, "complete")
            return

        images = [r for r in pending if r["kind"] == archive.KIND_IMAGE]
        videos = [r for r in pending if r["kind"] == archive.KIND_VIDEO]

        try:
            with zipfile.ZipFile(identity.path, "r") as zf:
                for row in images:
                    self._ingest_image(zf, identity, row)
        except (zipfile.BadZipFile, OSError) as exc:
            logger.error("Archive %s failed mid-ingest: %s", identity.path.name, exc)
            self.stats.archives_corrupt += 1
            self.catalog.takeout_archive_set_status(
                identity.archive_id, "corrupt", str(exc)
            )
            return

        for row in videos:
            self._defer_video(identity, row)

        if not self.catalog.takeout_members_pending(identity.archive_id):
            self.catalog.takeout_archive_set_status(identity.archive_id, "complete")

        self.stats.per_archive.append(
            {"archive": identity.path.name, "members": len(images) + len(videos)}
        )

    # -- driver -------------------------------------------------------------

    def run(self) -> IngestStats:
        todo = self._survey()

        if self.dry_run:
            # Report what phase 2 WOULD do, without extracting a byte or
            # writing a row. Counted from the surveyed member lists, NOT from
            # the catalog work queue -- a dry run wrote no rows, so the queue
            # would read empty.
            for _identity, members in todo:
                for member in members:
                    if _initial_status(member, self.include_trash) != _PENDING:
                        continue
                    if member.kind == archive.KIND_IMAGE:
                        self.stats.ingested += 1
                    elif member.kind == archive.KIND_VIDEO:
                        self.stats.deferred += 1
            return self.stats

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        for identity, _members in todo:
            self._ingest_archive(identity)
        return self.stats
