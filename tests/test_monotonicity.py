"""End-to-end guarantees: re-runs converge and never degrade a file."""

from pathlib import Path

from imageharbor.ai_classifier import AIClassifier, ContentDescription, StubClassifier
from imageharbor.catalog import Catalog
from imageharbor.enrich import enrich_library
from imageharbor.pipeline import Pipeline
from imageharbor.tiers import DESC_HUMAN_FILENAME, DESC_NONE


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

        # Capture the actual file identities (not just the count of names) so
        # this test would fail if a later cycle silently swapped one file's
        # content for another's while preserving the set of relative paths.
        digests_after_first_cycle = {
            p: p.read_bytes() for p in dest.rglob("*") if p.is_file()
        }

        for _ in range(3):
            Pipeline(src, dest, cat, write_sidecars=True).run()
            enrich_library(cat, dest, Fixed("mountain"), write_sidecars=True)

        assert _snapshot(dest) == after_first_cycle
        # The classifier changed subject ("beach" -> "mountain") between
        # cycles, but the AI-subject descriptor tier can never outrank or
        # replace a tier already recorded, and a repeated run at an equal
        # tier is defined to be a no-op -- so file bytes at each path must be
        # byte-for-byte identical to the first cycle, not merely present.
        for path, content in digests_after_first_cycle.items():
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
