"""End-to-end guarantees: re-runs converge and never degrade a file."""

from pathlib import Path

import pytest

from imageharbor.ai_classifier import AIClassifier, ContentDescription, StubClassifier
from imageharbor.catalog import Catalog
from imageharbor.enrich import enrich_library
from imageharbor.pipeline import Pipeline
from imageharbor.tiers import (
    DATE_FILENAME_PATTERN,
    DATE_NONE,
    DESC_AI_SUBJECT,
    DESC_HUMAN_FILENAME,
    DESC_NONE,
)


# ---------------------------------------------------------------------------
# Fixtures (copied from tests/test_pipeline.py -- not previously defined here)
# ---------------------------------------------------------------------------


@pytest.fixture()
def organized_dir(tmp_path: Path) -> Path:
    d = tmp_path / "organized"
    d.mkdir()
    return d


@pytest.fixture()
def catalog(tmp_path: Path) -> Catalog:
    cat = Catalog(tmp_path / "catalog.db")
    yield cat
    cat.close()


class Fixed(StubClassifier):
    def __init__(self, subject):
        self._subject = subject

    def describe(self, image_path, exif_data=None):
        return ContentDescription(
            primary_subject=self._subject, scene="s", objects=[], caption="c",
            tags=[], ocr_text="", model_version="fixed-1",
        )


class Broken(AIClassifier):
    def describe(self, image_path, exif_data=None):
        raise RuntimeError("down")


def _snapshot(dest: Path) -> set[str]:
    return {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}


def _sidecar_bytes(dest: Path) -> dict[str, bytes]:
    """Raw sidecar contents, keyed by relative path."""
    return {
        str(p.relative_to(dest)): p.read_bytes()
        for p in dest.rglob("*.json")
        if p.is_file()
    }


def _library(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG_20190704_123456.jpg").write_bytes(b"one")
    (src / "Emma's graduation.jpg").write_bytes(b"two")
    (src / "IMG_1234.jpg").write_bytes(b"three")
    return src, tmp_path / "dest"


def test_facts_then_enrich_reaches_a_fixed_point(tmp_path):
    src, dest = _library(tmp_path)
    with Catalog(tmp_path / "c.db") as cat:
        Pipeline(src, dest, cat, write_sidecars=True).run()
        enrich_library(cat, dest, Fixed("beach"), write_sidecars=True)
        after_first_cycle = _snapshot(dest)

        # Capture the actual IMAGE bytes at each path (not just the count of
        # names) so this test would fail if a later cycle silently swapped
        # one file's content for another's while preserving the set of
        # relative paths.
        image_bytes_after_first_cycle = {
            p: p.read_bytes()
            for p in dest.rglob("*")
            if p.is_file() and p.suffix != ".json"
        }

        sidecars: dict[str, bytes] = {}
        for i in range(3):
            Pipeline(src, dest, cat, write_sidecars=True).run()
            # reclassify=True forces every row back through Fixed("mountain")
            # even though all three are already enriched -- otherwise
            # iter_unenriched finds nothing to do, Fixed("mountain") is never
            # called, and this loop would prove only that the facts pass is
            # idempotent, not that the enrich half converges under a
            # changing AI answer (the guarantee this test claims to pin).
            again = enrich_library(
                cat, dest, Fixed("mountain"), write_sidecars=True, reclassify=True
            )
            assert again.total == 3    # the pass genuinely ran on every file
            assert again.renamed == 0  # and changed nothing: equal tier is a no-op

            # Sidecars must be stable ACROSS these repeated re-runs (i=0,1,2):
            # the classifier answer ("mountain") is identical every time in
            # this loop, so nothing here should cause per-pass churn (a stray
            # timestamp, non-deterministic key order, etc). This is narrower
            # than "sidecars never change at all" -- the one legitimate
            # change, "beach" -> "mountain" on the FIRST call in this loop,
            # already happened before this snapshot is taken, so it is not
            # what this assertion is checking.
            if i == 0:
                sidecars = _sidecar_bytes(dest)
            else:
                assert _sidecar_bytes(dest) == sidecars

        assert _snapshot(dest) == after_first_cycle
        # The classifier changed subject ("beach" -> "mountain") between
        # cycles, but the AI-subject descriptor tier can never outrank or
        # replace a tier already recorded, and a repeated run at an equal
        # tier is defined to be a no-op -- so file bytes at each path must be
        # byte-for-byte identical to the first cycle, not merely present.
        for path, content in image_bytes_after_first_cycle.items():
            assert path.read_bytes() == content, f"{path} changed after re-runs"


def test_an_outage_between_good_runs_loses_nothing(tmp_path):
    src, dest = _library(tmp_path)
    with Catalog(tmp_path / "c.db") as cat:
        Pipeline(src, dest, cat, write_sidecars=True).run()
        enrich_library(cat, dest, Fixed("beach"), write_sidecars=True)
        healthy = _snapshot(dest)
        healthy_rows = {
            row["sha256_b64url"]: (
                row["organized_path"], row["date_value"], row["date_tier"],
                row["descriptor_value"], row["descriptor_tier"],
                row["pcs_primary"], row["pcs_name"],
            )
            for row in cat.iter_all()
        }

        # reclassify=True forces these rows back through Broken() even though
        # they are already enriched -- otherwise iter_unenriched would find
        # nothing to do and Broken() would never actually be called, which
        # would let this test pass without exercising the outage at all.
        enrich_stats = enrich_library(
            cat, dest, Broken(), write_sidecars=True, reclassify=True
        )
        Pipeline(src, dest, cat, write_sidecars=True).run()
        enrich_library(cat, dest, Broken(), write_sidecars=True, reclassify=True)

        assert _snapshot(dest) == healthy
        # The outage must show up as failures, not as silent skipping --
        # otherwise this test would pass even if Broken() were never actually
        # invoked.
        assert enrich_stats.errors > 0
        assert enrich_stats.enriched == 0
        after_rows = {
            row["sha256_b64url"]: (
                row["organized_path"], row["date_value"], row["date_tier"],
                row["descriptor_value"], row["descriptor_tier"],
                row["pcs_primary"], row["pcs_name"],
            )
            for row in cat.iter_all()
        }
        assert after_rows == healthy_rows


def test_a_library_organized_with_no_ai_at_all_is_complete(tmp_path):
    """The facts pass alone must produce a fully organized, verified library."""
    from imageharbor.hashing import verify_pcs_file

    src, dest = _library(tmp_path)
    with Catalog(tmp_path / "c.db") as cat:
        stats = Pipeline(src, dest, cat, write_sidecars=True).run()

    assert stats.copied == 3
    images = [p for p in dest.rglob("*.jpg")]
    assert len(images) == 3
    assert all(verify_pcs_file(p) for p in images)
    # Every organized file must be reachable by content: its embedded digest
    # must round-trip to a real catalog row and back to the same path.
    with Catalog(tmp_path / "c.db") as cat:
        for path in images:
            from imageharbor.hashing import extract_digest_from_stem

            digest = extract_digest_from_stem(path.stem)
            assert digest is not None
            row = cat.get_by_sha256(digest)
            assert row is not None
            assert Path(row["organized_path"]) == path
            assert row["enriched_at"] is None


def test_a_better_named_duplicate_upgrades_the_descriptor(tmp_path):
    """Dedup does real organizing work, not just copy-skipping."""
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir(parents=True)
    (src / "a" / "IMG_1234.jpg").write_bytes(b"same")
    dest = tmp_path / "dest"

    with Catalog(tmp_path / "c.db") as cat:
        first = Pipeline(src, dest, cat).run()
        digest = first.results[0].sha256_b64url
        assert first.results[0].organized_path.stem == digest
        first_row = cat.get_by_sha256(digest)
        assert first_row["descriptor_tier"] == DESC_NONE
        original_path = Path(first_row["organized_path"])
        assert original_path.exists()

        (src / "b" / "Emma's graduation.jpg").write_bytes(b"same")
        Pipeline(src, dest, cat).run()

        assert len(cat.sources_for(digest)) == 2

        # The identity check that actually pins the fix: the record's
        # descriptor tier must have moved from "no human name" to "human
        # filename", the organized path must be the NEW, better-named file
        # (not just any file), the old bare-digest path must be gone (it was
        # relocated, not copied alongside), and the surviving file's name
        # must actually contain the human descriptor -- not merely be a
        # different path.
        upgraded_row = cat.get_by_sha256(digest)
        assert upgraded_row["descriptor_tier"] == DESC_HUMAN_FILENAME
        assert upgraded_row["descriptor_value"] == "emmas-graduation"
        upgraded_path = Path(upgraded_row["organized_path"])
        assert upgraded_path != original_path
        assert not original_path.exists()
        assert upgraded_path.exists()
        assert upgraded_path.name == f"emmas-graduation_{digest}.jpg"


def test_a_duplicate_upgrade_re_merges_the_sidecar(tmp_path):
    """A duplicate upgrade must not leave the sidecar's date/descriptor blocks
    holding pre-upgrade values.

    `_maybe_upgrade_from_duplicate` carries the sidecar FILE to the new path
    (a plain rename), but a rename alone doesn't touch its JSON content --
    unlike enrich.py, which explicitly re-merges after a tier-gated
    relocation. Regression test for the equivalent fix on the duplicate-
    upgrade path.
    """
    from imageharbor.sidecar import read_sidecar

    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir(parents=True)
    (src / "a" / "IMG_1234.jpg").write_bytes(b"same")
    dest = tmp_path / "dest"

    with Catalog(tmp_path / "c.db") as cat:
        first = Pipeline(src, dest, cat, write_sidecars=True).run()
        digest = first.results[0].sha256_b64url
        original_path = Path(cat.get_by_sha256(digest)["organized_path"])
        original_sidecar = read_sidecar(original_path)
        assert original_sidecar["descriptor"]["tier"] == DESC_NONE
        assert original_sidecar["descriptor"]["value"] == ""

        (src / "b" / "Emma's graduation.jpg").write_bytes(b"same")
        Pipeline(src, dest, cat, write_sidecars=True).run()

        upgraded_row = cat.get_by_sha256(digest)
        upgraded_path = Path(upgraded_row["organized_path"])
        assert upgraded_path != original_path

        upgraded_sidecar = read_sidecar(upgraded_path)
        # The sidecar at the NEW path must reflect the NEW (upgraded) tier
        # and value, matching what the catalog now records -- not stale
        # pre-upgrade data merely carried along by the file rename.
        assert upgraded_sidecar["descriptor"]["tier"] == DESC_HUMAN_FILENAME
        assert upgraded_sidecar["descriptor"]["value"] == "emmas-graduation"
        assert upgraded_sidecar["descriptor"]["tier"] == upgraded_row["descriptor_tier"]
        assert upgraded_sidecar["descriptor"]["value"] == upgraded_row["descriptor_value"]
        assert upgraded_sidecar["date"]["tier"] == upgraded_row["date_tier"]
        # Facts recorded by the FIRST pass (identity) must survive the merge.
        assert upgraded_sidecar["identity"]["sha256_b64url"] == digest
        # Both source paths must be reflected, not just the triggering one.
        assert {s["path"] for s in upgraded_sidecar["sources"]} == {
            str(src / "a" / "IMG_1234.jpg"),
            str(src / "b" / "Emma's graduation.jpg"),
        }


def test_a_tied_descriptor_survives_while_only_the_date_upgrades(tmp_path):
    """Tie-break regression (Fix Round 1, Critical 2).

    `tiers.is_upgrade` fires when EITHER dimension improves; the OTHER
    dimension may merely tie. A duplicate that supplies a better date but a
    descriptor at the SAME tier as the one already on record must adopt the
    new date while leaving the original descriptor untouched -- a tie is not
    an upgrade for that dimension. The reviewer reproduced a live bug where a
    `>=` tie-break let the later, unrelated human filename ("bobs-party-2019")
    silently replace the correct, already-chosen one ("emmas-graduation").
    """
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir(parents=True)
    # First seen: a human-authored name, no date anywhere in it.
    (src / "a" / "Emma's graduation.jpg").write_bytes(b"tie-break")
    dest = tmp_path / "dest"

    with Catalog(tmp_path / "c.db") as cat:
        first = Pipeline(src, dest, cat).run()
        digest = first.results[0].sha256_b64url
        first_row = cat.get_by_sha256(digest)
        assert first_row["descriptor_tier"] == DESC_HUMAN_FILENAME
        assert first_row["descriptor_value"] == "emmas-graduation"
        assert first_row["date_tier"] == DATE_NONE

        # Same bytes, found later at a path that is ALSO human-authored
        # (same descriptor tier -- a tie) but additionally carries a date
        # (a strict improvement on the date dimension only).
        (src / "b" / "Bobs party 2019-07-04.jpg").write_bytes(b"tie-break")
        Pipeline(src, dest, cat).run()

        upgraded_row = cat.get_by_sha256(digest)
        # The date dimension strictly improved, so it must be adopted...
        assert upgraded_row["date_tier"] == DATE_FILENAME_PATTERN
        assert upgraded_row["date_value"] == "2019-07-04"
        # ...but the descriptor dimension only tied (30 == 30), so the
        # ORIGINAL descriptor must survive untouched -- not be overwritten
        # by the unrelated name "bobs-party-2019" that happened to arrive
        # alongside the better date.
        assert upgraded_row["descriptor_tier"] == DESC_HUMAN_FILENAME
        assert upgraded_row["descriptor_value"] == "emmas-graduation"

        upgraded_path = Path(upgraded_row["organized_path"])
        assert upgraded_path.exists()
        assert upgraded_path.name == f"2019-07-04-emmas-graduation_{digest}.jpg"


def test_a_human_named_duplicate_displaces_an_ai_name(tmp_path):
    """The one cross-pass transition nothing covered: enrichment names a file,
    then the facts pass finds a better-named duplicate and displaces it.

    Enrichment can only ever reach DESC_AI_SUBJECT (20); a human-authored
    filename is 30. So a duplicate arriving later under a human name must win --
    and the classification enrichment recorded must survive the rename.
    """
    from imageharbor.sidecar import read_sidecar

    src = tmp_path / "src"
    # discover_images yields in sorted posix-path order, so "a" (camera-named,
    # no-op duplicate on the second pass) must sort before "b" (human-named,
    # the one that triggers the upgrade).
    (src / "a").mkdir(parents=True)
    (src / "b").mkdir(parents=True)
    (src / "a" / "IMG_1234.jpg").write_bytes(b"same-bytes")
    dest = tmp_path / "dest"

    with Catalog(tmp_path / "c.db") as cat:
        # 1. Facts pass on the camera-named file alone: no human name, no AI --
        # descriptor tier must be DESC_NONE.
        first = Pipeline(src, dest, cat, write_sidecars=True).run()
        digest = first.results[0].sha256_b64url
        first_row = cat.get_by_sha256(digest)
        assert first_row["descriptor_tier"] == DESC_NONE

        # 2. Enrich: the AI names it "beach" -- DESC_AI_SUBJECT, a strict
        # upgrade over DESC_NONE, so a rename happens.
        enrich_library(cat, dest, Fixed("beach"), write_sidecars=True)
        enriched_row = cat.get_by_sha256(digest)
        assert enriched_row["descriptor_tier"] == DESC_AI_SUBJECT
        enriched_path = Path(enriched_row["organized_path"])
        assert enriched_path.name == f"beach_{digest}.jpg"
        enriched_sidecar = read_sidecar(enriched_path)
        assert enriched_sidecar["classification"]["primary_subject"] == "beach"

        # 3. Add a duplicate under a human-authored name and re-run the facts
        # pass. Human filename (30) beats AI subject (20): this must displace
        # the AI's name, not merely tie or lose to it.
        (src / "b" / "Emma's graduation.jpg").write_bytes(b"same-bytes")
        Pipeline(src, dest, cat, write_sidecars=True).run()

        # 4. The upgrade happened: new human-named path, DESC_HUMAN_FILENAME,
        # old path and old sidecar are both gone.
        upgraded_row = cat.get_by_sha256(digest)
        assert upgraded_row["descriptor_tier"] == DESC_HUMAN_FILENAME
        upgraded_path = Path(upgraded_row["organized_path"])
        assert upgraded_path.name == f"emmas-graduation_{digest}.jpg"
        assert not enriched_path.exists()
        assert upgraded_path.exists()
        assert not read_sidecar(enriched_path)
        old_sidecar_path = enriched_path.with_name(f"{enriched_path.stem}.json")
        assert not old_sidecar_path.exists()

        # 5. Enrichment's classification work survived the rename -- it was
        # carried and re-merged, not lost.
        upgraded_sidecar = read_sidecar(upgraded_path)
        assert upgraded_sidecar["classification"]["primary_subject"] == "beach"
        assert upgraded_sidecar["descriptor"]["value"] == "emmas-graduation"
        assert upgraded_sidecar["identity"]["sha256_b64url"] == digest

        # 6. It cannot be clawed back: a different AI answer under
        # reclassify=True must not displace the now-human-named file. 20 can
        # never outrank 30.
        clawback_stats = enrich_library(
            cat, dest, Fixed("mountain"), write_sidecars=True, reclassify=True
        )
        assert clawback_stats.renamed == 0
        final_row = cat.get_by_sha256(digest)
        assert Path(final_row["organized_path"]) == upgraded_path
        assert upgraded_path.exists()


def test_late_evidence_upgrades_a_duplicate_out_of_undated(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    """The late-arriving-sidecar case, at the pipeline level.

    Part 1 ingests with no Google date -> Undated/. Part 2 arrives carrying the
    sidecar, the bytes hash as a duplicate, and the EXISTING monotonic upgrade
    machinery relocates the file. No new code path is needed for this --
    only that `_maybe_upgrade_from_duplicate` is given the evidence.
    """
    from datetime import datetime

    from imageharbor.pipeline import ExternalEvidence, Pipeline

    staged = tmp_path / "IMG_1234.jpg"
    staged.write_bytes(b"\xff\xd8\xff\xe0" + b"\x07" * 16 + b"\xff\xd9")

    pipeline = Pipeline(tmp_path, organized_dir, catalog)
    first = pipeline.process_file(staged)
    assert first.organized_path.parent == organized_dir / "Undated"

    second = pipeline.process_file(
        staged, evidence=ExternalEvidence(date=datetime(2015, 3, 9, 12, 56, 32))
    )
    assert second.status == "duplicate"

    row = catalog.get_by_sha256(first.sha256_b64url)
    assert Path(row["organized_path"]).parent == organized_dir / "2015" / "2015-03"
    assert Path(row["organized_path"]).exists()
    assert not first.organized_path.exists()


def test_re_ingesting_the_same_evidence_is_a_rename_no_op(
    tmp_path: Path, organized_dir: Path, catalog: Catalog
) -> None:
    from datetime import datetime

    from imageharbor.pipeline import ExternalEvidence, Pipeline

    staged = tmp_path / "IMG_1234.jpg"
    staged.write_bytes(b"\xff\xd8\xff\xe0" + b"\x08" * 16 + b"\xff\xd9")
    evidence = ExternalEvidence(date=datetime(2015, 3, 9))

    pipeline = Pipeline(tmp_path, organized_dir, catalog)
    first = pipeline.process_file(staged, evidence=evidence)
    before = catalog.get_by_sha256(first.sha256_b64url)["organized_path"]

    pipeline.process_file(staged, evidence=evidence)
    after = catalog.get_by_sha256(first.sha256_b64url)["organized_path"]

    assert before == after
    assert Path(after).exists()
