"""Opt-in corpus test: does the pairing index agree with ImageHarbor's
built-in ladder on a slice of the real Google Takeout export on this
machine.

Follows the sibling `Takeout_Inventory` project's own `corpus` convention
(see its `tests/test_corpus.py` and `pyproject.toml`): a `corpus`-marked
test is opt-in, excluded from the default suite here too (`addopts = '-m
"not corpus"'` in this repo's `pyproject.toml`), and run explicitly with
`uv run pytest -m corpus -v`. SKIPPED is a normal outcome on a machine with
no export at `EXPORT_DIR` -- a FAILURE is not.

Scoped to a HANDFUL of the real archives, never the whole export: the full
export behind `EXPORT_DIR` is 175 archives / ~388 GB, and a full
`scan_takeout` over all of it takes several minutes. A representative slice
is enough to run two independent, real pairing engines (ImageHarbor's own
`pairing.py` and Takeout_Inventory's real `scan_takeout` + pairing engine,
loaded exactly the way `tests/test_takeout_index_equivalence.py` loads it)
against real Google-authored file names -- it does not need every archive
to prove that, and a multi-minute run does not belong in the suite even
behind an opt-in marker when a several-archive one proves the same
properties in seconds (confirmed interactively: 4 real archives, including
a ~2 GB one, scan in well under 15 seconds).

The chosen archives are HARD-LINKED, never copied, into an isolated
directory next to this source tree: a hard link shares the original bytes
on disk instead of duplicating multi-GB files, but only works within one
NTFS volume. If `EXPORT_DIR` and this repository are not on the same drive,
`os.link` raises `OSError` and the test skips rather than silently copying
gigabytes.

Invariants only. The corpus is not a fixture: this file must never assert
a count -- a hard number breaks the moment the export changes, and teaches
whoever hits the break to edit the number instead of investigate.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from imageharbor.takeout import index_reader, pairing
from tests.test_takeout_index_equivalence import _SIBLING, _SIBLING_PATH

# Overridable so a different machine can point at its own export, the same
# idea as Takeout_Inventory's own TAKEOUT_DIR env var.
EXPORT_DIR = Path(os.environ.get(
    "TAKEOUT_DIR",
    r"D:\Users\Conrad\Documents\programming\Google_Takeout_Downloader\takeout",
))

# Where the hard-linked subset is staged. Deliberately NOT inside EXPORT_DIR
# (several projects around this one treat their export directory as
# read-only) and deliberately NOT pytest's `tmp_path` (that lives under the
# system temp directory, which on this machine is a different drive from
# both EXPORT_DIR and this repo -- a hard link across drives is impossible).
# Same volume as EXPORT_DIR is required for the hard-link subset to work at
# all; a repo checked out next to the export, as this one is, satisfies
# that without hard-coding a drive letter.
_LINK_ROOT = Path(__file__).resolve().parents[1] / ".corpus-tmp"

# First, two interior, and last -- picked by position, not name, so this
# keeps working as the export grows or shrinks at either end.
_SUBSET_SIZE = 4

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(
        not EXPORT_DIR.is_dir(), reason=f"no Takeout export at {EXPORT_DIR}"
    ),
    pytest.mark.skipif(
        _SIBLING is None,
        reason=f"Takeout_Inventory sibling not importable from {_SIBLING_PATH}",
    ),
]


def _pick_archives(archives: list[Path]) -> list[Path]:
    if len(archives) <= _SUBSET_SIZE:
        return archives
    picks = [
        archives[0],
        archives[len(archives) // 3],
        archives[2 * len(archives) // 3],
        archives[-1],
    ]
    # De-duplicate while keeping order, in case a small export makes two
    # positions land on the same file.
    seen: set[Path] = set()
    out: list[Path] = []
    for a in picks:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _hardlink_subset(dest: Path) -> Path:
    """Hard-link `_SUBSET_SIZE` real archives into `dest` so the scan below
    sees a slice of the real export, never all 175 archives. Raises OSError
    (turned into a skip by the caller) if `dest` is not on the same volume
    as EXPORT_DIR.
    """
    archives = sorted(EXPORT_DIR.glob("*.zip"))
    if not archives:
        pytest.skip(f"no .zip archives under {EXPORT_DIR}")
    subset_dir = dest / "export-subset"
    subset_dir.mkdir()
    for src in _pick_archives(archives):
        os.link(src, subset_dir / src.name)
    return subset_dir


def test_indexed_and_builtin_agree_on_the_real_export(tmp_path):
    """Invariants only. The corpus is not a fixture: never assert a count."""
    _LINK_ROOT.mkdir(exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=_LINK_ROOT) as link_dir:
            export_subset = _hardlink_subset(Path(link_dir))
            archive_names = sorted(p.name for p in export_subset.glob("*.zip"))
            assert archive_names, "the hard-link subset should contain archives"

            cache_dir = tmp_path / "ti-cache"
            inv = _SIBLING.scan_takeout(
                export_subset, cache_dir, workers=4, on_progress=lambda *a, **k: None
            )
            idx_path = tmp_path / "corpus-index.sqlite"
            _SIBLING.write_index_sqlite(inv, idx_path)

            archive_stats = {p.name: p.stat() for p in export_subset.glob("*.zip")}
            index = index_reader.IndexPairings.open(idx_path, archive_stats)

            # An index built moments ago from exactly these archives must
            # cover every one of them.
            covered = sum(1 for name in archive_names if index.covers(name))
            assert covered == len(archive_names), (
                "the index should cover a matching export"
            )
            assert index.pairings, "the sampled archives should yield pairings"

            all_members = [m.path for m in inv.members]
            media_members = [m for m in all_members if not m.lower().endswith(".json")]
            assert media_members, "the sampled archives should contain real media"

            builtin_index = pairing.build_index(all_members)

            compared = 0
            for member in media_members:
                builtin = pairing.sidecar_for(member, builtin_index)
                indexed = index.sidecar_for(member)
                if builtin.sidecar and indexed and indexed.sidecar:
                    compared += 1
                    # The two engines never name a different sidecar for the
                    # same member -- if they did, whichever ran would decide
                    # a photo's date.
                    assert builtin.sidecar == indexed.sidecar, (
                        f"{member}: built-in ladder says {builtin.sidecar}, "
                        f"index says {indexed.sidecar}"
                    )
                    # m6: the branch's headline property is `confidence`,
                    # not merely which sidecar gets named.
                    assert builtin.confidence == indexed.confidence, (
                        f"{member}: built-in confidence {builtin.confidence}, "
                        f"index confidence {indexed.confidence}"
                    )
            assert compared > 0, (
                "no member had both a built-in and an indexed answer to "
                "compare -- this run proves nothing"
            )

            # Every pairing the index hands back carries a valid confidence.
            assert all(
                p.confidence in (pairing.OWN, pairing.RELATED, pairing.NO_MATCH)
                for p in index.pairings.values()
            ), "every indexed pairing must carry one of the three known confidences"
    except OSError as exc:
        pytest.skip(
            f"could not hard-link the real export into {_LINK_ROOT} "
            f"(different volume from {EXPORT_DIR}?): {exc}"
        )
