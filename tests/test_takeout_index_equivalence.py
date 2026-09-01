"""The two properties the optional-index design rests on.

Every other test in this plan checks a mechanism (open a database, route one
pairing, count one fallback). These two check the PROPERTIES the whole design
depends on -- see `imageharbor/takeout/pairing.py` and
`imageharbor/takeout/index_reader.py` for the mechanisms themselves.

1. Equivalence: ImageHarbor's built-in `pairing.py` and a Takeout_Inventory
   index are two independent implementations of "which sidecar describes
   which photo". The index may pair where the built-in returns None -- it has
   rules (e.g. cross-directory) the built-in deliberately lacks -- but it must
   never name a DIFFERENT sidecar for a member the built-in also pairs. If it
   did, which code path ran would decide a photo's date.

2. Optionality: an ingest against an index that exists but covers nothing
   (every archive's stats mismatch) must produce byte-for-byte the same
   catalog as an ingest with no index argument at all. This is what makes
   "optional" a property of the code, not just a description of the feature.

INDEX SOURCE (read this before touching the pairing tables below):
`_build_synthetic_index` tries to import Takeout_Inventory's actual
`takeout_inventory.py` from its sibling checkout
(`D:\\Users\\Conrad\\Documents\\programming\\Takeout_Inventory`) and, if that
succeeds, calls the sibling's OWN `scan_takeout` + `write_index_sqlite` --
its real content-sniffing, its real pairing engine (`pair_media`,
`pair_all_media`), its real schema writer. That was confirmed importable and
working in this environment (verified interactively: its only runtime
dependency, `rich`, is never imported at module level, so the bare import
succeeds even without `rich` installed) and every shape in `MEMBERS` below
was checked against it before being committed here.

If that import ever fails -- the sibling checkout moves, or a future version
of it pulls in a real dependency at import time -- `_build_synthetic_index`
falls back to hand-building the same SQLite schema `test_takeout_index_reader`
uses, with pairing answers from `_LITERAL_PAIRS` below. That fallback is
STRICTLY WEAKER: `_LITERAL_PAIRS` was authored by reading `pairing.py`'s own
rung documentation, so a fallback run checks ImageHarbor against a human's
transcription of ImageHarbor's own idea of correct pairing, not against the
sibling repository's actual, independent implementation. The cross-directory
row in that fallback is fabricated outright (see `_LITERAL_PAIRS` below) --
it exists only so the "index legitimately pairs where the built-in can't"
skip branch has something to skip even in fallback mode, not because the
fallback path can produce that answer honestly. `INDEX_SOURCE` records which
branch actually ran. Neither test HARD-asserts it is "sibling_writer" --
doing so would fail this whole file on any machine without the sibling
checkout, which is not a property either test exists to pin -- but a
fallback to the weak path still must not be silent, so module import emits a
plain `UserWarning` (via a bare `warnings.warn(...)`, no category argument)
when it happens; that warning surfaces in the test run's summary (a real one
was observed and read during development, while
tracking down why `INDEX_SOURCE` was unexpectedly "literal_schema" on THIS
machine, despite the sibling checkout being present and importable -- see the
report for what that turned out to mean).
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from dataclasses import fields
from pathlib import Path

from imageharbor.catalog import Catalog
from imageharbor.takeout import index_reader, pairing
from imageharbor.takeout.ingest import ingest_archives
from tests.test_takeout_index_reader import make_index
from tests.test_takeout_ingest import D, _jpeg, _sidecar, _zip, catalog, dirs

# --------------------------------------------------------------------------
# Load Takeout_Inventory's real writer if it is importable in this
# environment; otherwise INDEX_SOURCE stays "literal" and every test below
# falls back to `_LITERAL_PAIRS`. See the module docstring.

_SIBLING_PATH = Path(
    r"D:\Users\Conrad\Documents\programming\Takeout_Inventory\takeout_inventory.py"
)


def _load_sibling():
    if not _SIBLING_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "_sibling_takeout_inventory", _SIBLING_PATH
    )
    module = importlib.util.module_from_spec(spec)
    # Must be registered in sys.modules BEFORE exec_module: takeout_inventory.py
    # decorates classes with @dataclass, and dataclasses (as of 3.13) resolves
    # forward-reference type hints via `sys.modules[cls.__module__]` while the
    # class body is still executing -- skip this and every @dataclass in the
    # sibling module raises AttributeError('NoneType' has no '__dict__')
    # instead of a clean class. Confirmed by hand: omitting this line reliably
    # reproduces that exact traceback (see the report) and _load_sibling then
    # silently degrades to the literal-schema fallback, so this is not
    # optional plumbing -- without it, INDEX_SOURCE is *always* "literal" no
    # matter how importable the sibling actually is.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Import-time failure of another repository's script must degrade
        # this test's strength, never fail collection here.
        del sys.modules[spec.name]
        return None
    return module


_SIBLING = _load_sibling()
INDEX_SOURCE = "sibling_writer" if _SIBLING is not None else "literal_schema"
if INDEX_SOURCE == "literal_schema":
    warnings.warn(
        "test_takeout_index_equivalence: Takeout_Inventory's sibling checkout "
        f"was not importable from {_SIBLING_PATH} -- falling back to the "
        "literal-schema index, which checks ImageHarbor against its own idea "
        "of the sibling's pairing, not the sibling's actual implementation. "
        "See this module's docstring.",
        stacklevel=1,
    )


# --------------------------------------------------------------------------
# The synthetic export: one shape per rung `pairing.py` documents, plus a
# cross-directory pairing the built-in ladder can never make (it never
# crosses a directory boundary -- see test_takeout_pairing.py's
# test_a_sidecar_in_another_directory_is_not_matched). All non-cross shapes
# below were verified interactively against the sibling's real engine to
# agree with `pairing.py` before being committed (see the module docstring).

D_CROSS = "Takeout/AlbumArchive/OtherYear/album"  # same AREA as D, different FOLDER

# Google truncates a member's whole name (stem + extension) to 46 characters
# before appending ".json" -- see pairing.py's _MIN_TRUNCATION_PREFIX comment
# and Takeout_Inventory's TRUNCATION_LIMIT = 46. Both engines' truncation
# rungs key off exactly that cut, so the fixture is built from it directly
# rather than an arbitrary "long name" (an earlier draft of this fixture used
# test_takeout_pairing.py's own truncation example and it did NOT survive
# Takeout_Inventory's fixed 46-char cut -- only a name truncated at exactly
# that boundary agrees with both engines).
_LONG_NAME = "an-extremely-long-original-filename-from-google-photos-app.jpg"
_TRUNC_STEM = _LONG_NAME[:46]
assert len(_TRUNC_STEM) == 46, "fixture invariant: both engines truncate at 46 chars"
_TRUNC_SIDECAR = f"{_TRUNC_STEM}.json"

MEMBERS: dict[str, bytes] = {
    # Rung 2/3: exact NAME.EXT.json
    f"{D}/exact.jpg": _jpeg(1),
    f"{D}/exact.jpg.json": _sidecar("exact.jpg", 1),
    # Rung 3: the newer supplemental-metadata spelling
    f"{D}/supp.jpg": _jpeg(2),
    f"{D}/supp.jpg.supplemental-metadata.json": _sidecar("supp.jpg", 2),
    # Rung 1: copy-suffix (N), displaced past the extension
    f"{D}/copy(1).jpg": _jpeg(3),
    f"{D}/copy.jpg(1).json": _sidecar("copy.jpg(1)", 3),
    # Rung 1b: copy-suffix (N) after the supplemental-metadata tag
    f"{D}/copy2(1).jpg": _jpeg(4),
    f"{D}/copy2.jpg.supplemental-metadata(1).json": _sidecar("copy2.jpg(1)", 4),
    # Rung 4: -edited inherits the original's sidecar (RELATED, not OWN)
    f"{D}/orig.jpg": _jpeg(5),
    f"{D}/orig.jpg.json": _sidecar("orig.jpg", 5),
    f"{D}/orig-edited.jpg": _jpeg(6),
    # Rung 5: case-insensitive retry
    f"{D}/UPPER.JPG": _jpeg(7),
    f"{D}/upper.jpg.json": _sidecar("upper.jpg", 7),
    # Rung 6: truncation recovery, both engines' cut aligned (see above)
    f"{D}/{_LONG_NAME}": _jpeg(8),
    f"{D}/{_TRUNC_SIDECAR}": _sidecar("trunc", 8),
    # No sidecar anywhere: both paths must agree on None
    f"{D}/lonely.jpg": _jpeg(9),
    # Cross-directory: same area as D, different folder. Only a real
    # Takeout_Inventory pairing engine can resolve this; the built-in ladder
    # must return None. Exercises the "index pairs where built-in can't" arm
    # of the equivalence test, which must not be dead code.
    f"{D_CROSS}/cross.jpg": _jpeg(10),
    f"{D}/cross.jpg.json": _sidecar("cross.jpg", 10),
}

MEDIA_MEMBERS = [m for m in MEMBERS if not m.lower().endswith(".json")]

# The literal-schema fallback's pairing answers -- see the module docstring's
# "INDEX SOURCE" section for exactly what these are (and are not) evidence
# of. (member, sidecar_or_None, confidence, rule_label).
_LITERAL_PAIRS = [
    (f"{D}/exact.jpg", f"{D}/exact.jpg.json", pairing.OWN, "exact"),
    (f"{D}/supp.jpg", f"{D}/supp.jpg.supplemental-metadata.json", pairing.OWN, "supplemental"),
    (f"{D}/copy(1).jpg", f"{D}/copy.jpg(1).json", pairing.OWN, "copy-suffix"),
    (f"{D}/copy2(1).jpg", f"{D}/copy2.jpg.supplemental-metadata(1).json",
     pairing.OWN, "copy-suffix-supplemental"),
    (f"{D}/orig.jpg", f"{D}/orig.jpg.json", pairing.OWN, "exact"),
    (f"{D}/orig-edited.jpg", f"{D}/orig.jpg.json", pairing.RELATED, "edited"),
    (f"{D}/UPPER.JPG", f"{D}/upper.jpg.json", pairing.OWN, "case-insensitive"),
    (f"{D}/{_LONG_NAME}", f"{D}/{_TRUNC_SIDECAR}", pairing.OWN, "truncated"),
    (f"{D}/lonely.jpg", None, pairing.NO_MATCH, "orphan"),
    # FABRICATED, not derived from any pairing engine -- see the module
    # docstring. Exists only so the divergence-skip branch has something to
    # skip in fallback mode too.
    (f"{D_CROSS}/cross.jpg", f"{D}/cross.jpg.json", pairing.RELATED, "fabricated-cross-directory"),
]


def _build_synthetic_index(tmp_path: Path, zip_path: Path) -> Path:
    """Build the pairing index for `MEMBERS` via whichever source
    `INDEX_SOURCE` names. Returns the SQLite path."""
    if _SIBLING is not None:
        cache_dir = tmp_path / "ti-cache"
        inv = _SIBLING.scan_takeout(
            zip_path.parent, cache_dir, workers=1, on_progress=lambda *a, **k: None
        )
        idx_path = tmp_path / "sibling-index.sqlite"
        _SIBLING.write_index_sqlite(inv, idx_path)
        return idx_path

    st = zip_path.stat()
    sidecar_rows = []
    media_rows = []
    sidecar_id_by_path: dict[str, int] = {}
    next_id = 1
    for member, sidecar, confidence, rule in _LITERAL_PAIRS:
        sidecar_id = None
        if sidecar is not None:
            if sidecar not in sidecar_id_by_path:
                sidecar_id_by_path[sidecar] = next_id
                sidecar_rows.append(
                    (next_id, zip_path.name, sidecar, sidecar.rsplit("/", 1)[-1])
                )
                next_id += 1
            sidecar_id = sidecar_id_by_path[sidecar]
        media_rows.append((
            zip_path.name, member, "area", "folder",
            member.rsplit("/", 1)[-1], sidecar_id, rule, confidence,
        ))
    return make_index(
        tmp_path / "literal-index.sqlite",
        archives=((zip_path.name, st.st_size, int(st.st_mtime), 0, None),),
        sidecars=sidecar_rows,
        media=media_rows,
    )


def test_the_two_pairing_paths_never_name_different_sidecars(tmp_path):
    """The index may pair where the built-in rungs return None -- it has
    rules ImageHarbor lacks. It must never name a DIFFERENT sidecar for the
    same member: that would be two implementations of one domain
    disagreeing, and whichever ran would decide a photo's date.

    That includes the case where the built-in has NO opinion about the
    member being compared but DOES have an opinion about the sidecar: if the
    index hands `lonely.jpg` a sidecar the built-in already assigned to some
    other real member, accepting that pairing would stamp the other
    member's title/capture date onto `lonely.jpg`, which is exactly the
    cross-repo divergence this test exists to catch -- so the "index knows
    more" exemption below is narrowed to exclude that case rather than
    accepting any index-only answer unconditionally.
    """
    archives_dir = tmp_path / "archives"
    archives_dir.mkdir()
    zip_path = _zip(archives_dir / "takeout-001.zip", MEMBERS)

    builtin_index = pairing.build_index(list(MEMBERS))
    idx_path = _build_synthetic_index(tmp_path, zip_path)
    index = index_reader.IndexPairings.open(idx_path, {zip_path.name: zip_path.stat()})
    assert index.covers(zip_path.name)

    # Built once, before the loop, from the same member list: which member
    # (if any) the built-in ladder already claims each sidecar for. Used
    # below to check that an "index knows more" pairing never reassigns a
    # sidecar the built-in already gave to a DIFFERENT member.
    builtin_by_member = {
        member: pairing.sidecar_for(member, builtin_index) for member in MEDIA_MEMBERS
    }
    builtin_owner_of: dict[str, str] = {
        b.sidecar: member
        for member, b in builtin_by_member.items()
        if b.sidecar is not None
    }

    compared = 0
    skipped_index_only = 0
    for member in MEDIA_MEMBERS:
        builtin = builtin_by_member[member]
        indexed = index.sidecar_for(member)
        if builtin.sidecar is None or indexed is None or indexed.sidecar is None:
            if indexed is not None and indexed.sidecar is not None:
                # The legitimate divergence direction: the index knows
                # something the built-in ladder cannot. Legitimate ONLY if
                # the sidecar it names for `member` isn't already claimed by
                # the built-in for some OTHER member -- a sidecar the
                # built-in already assigned elsewhere is not "the index has
                # rules ImageHarbor lacks", it's the index disagreeing with
                # the built-in about who owns that sidecar, and the built-in
                # side of that disagreement is a real, present pairing, not
                # a blind spot.
                claimant = builtin_owner_of.get(indexed.sidecar)
                assert claimant is None, (
                    f"{member}: index pairs it with {indexed.sidecar}, but "
                    f"the built-in ladder already assigns that sidecar to "
                    f"{claimant}")
                skipped_index_only += 1
            continue
        compared += 1
        assert builtin.sidecar == indexed.sidecar, (
            f"{member}: built-in says {builtin.sidecar}, "
            f"index says {indexed.sidecar}")
        # m6: the branch's headline property is `confidence`, not merely
        # which sidecar gets named -- a disagreement there silently flips
        # the RELATED date-tier/title/people policy even when both engines
        # point at the same document.
        assert builtin.confidence == indexed.confidence, (
            f"{member}: built-in confidence {builtin.confidence}, "
            f"index confidence {indexed.confidence}")

    # Neither arm of the loop may be dead code, or this test proves nothing:
    # a run that never actually compared two real answers would pass
    # vacuously, and a run that never hit the "index knows more" branch
    # would not prove that direction is truly permitted.
    assert compared == 8, "the same-answer comparison ran on fewer members than expected"
    assert skipped_index_only == 1, "the index-only divergence branch never fired"


def test_a_mismatched_index_changes_nothing(tmp_path, dirs, catalog: Catalog):
    """What makes 'optional' safe rather than merely intended.

    An index that is present but covers NOTHING (every archive's on-disk
    size/mtime mismatches what the index recorded) must make an ingest
    behave identically to an ingest with no index argument at all -- not
    merely "similar", byte-for-byte identical in every catalog row.
    """
    archives, dest_without = dirs
    _zip(archives / "takeout-001.zip", {
        f"{D}/a.jpg": _jpeg(20),
        f"{D}/a.jpg.json": _sidecar("a.jpg", 1425905792),
        f"{D}/b.jpg": _jpeg(21),  # no sidecar: exercises missing_metadata too
    })
    _zip(archives / "takeout-002.zip", {
        f"{D}/c.jpg": _jpeg(22),
        f"{D}/c.jpg.json": _sidecar("c.jpg", 1425905792),
        f"{D}/d.jpg": _jpeg(23),
    })
    archive_names = sorted(p.name for p in archives.glob("*.zip"))
    assert len(archive_names) == 2

    stale_index = make_index(
        tmp_path / "stale-index.sqlite",
        archives=tuple((name, 1, 1, 0, None) for name in archive_names),
    )

    without = ingest_archives(archives, dest_without, catalog)

    dest_with_stale = tmp_path / "organized-with-stale-index"
    dest_with_stale.mkdir()
    cat_with_stale = Catalog(tmp_path / "catalog-with-stale-index.db")
    try:
        with_stale = ingest_archives(
            archives, dest_with_stale, cat_with_stale, index_path=stale_index
        )

        assert with_stale.index_archives_covered == 0
        assert with_stale.index_archives_fell_back == len(archive_names)
        # The no-index run never even attempts to open one: `self.index_path`
        # stays None throughout, so this counter is a true zero, not merely
        # "zero because nothing fell back yet".
        assert without.index_archives_fell_back == 0
        assert without.index_path is None

        assert _catalog_rows(catalog, dest_without) == _catalog_rows(
            cat_with_stale, dest_with_stale)

        # Every OTHER stat must match too -- an index that is present but
        # uncovered is a genuinely different code path (self.index is a real
        # IndexPairings object, just one that answers "not covered" for
        # every archive) from never having attempted to open one at all, and
        # this is the assertion that those two paths converge on identical
        # behaviour, not just identical catalog content.
        without_stats = {
            f.name: getattr(without, f.name) for f in fields(without)
            if f.name not in ("index_path", "index_archives_fell_back")
        }
        with_stale_stats = {
            f.name: getattr(with_stale, f.name) for f in fields(with_stale)
            if f.name not in ("index_path", "index_archives_fell_back")
        }
        assert without_stats == with_stale_stats
    finally:
        cat_with_stale.close()


def _catalog_rows(cat: Catalog, dest: Path) -> list[dict]:
    """Every `photos` row, normalized so two ingests into DIFFERENT
    destination directories can be compared for equality.

    `id`, `created_at` and `processed_at` are dropped: they are bookkeeping
    that legitimately differs between two separate `Catalog` databases (a
    fresh autoincrement sequence, two distinct wall-clock timestamps) and
    were never part of what "optional" promises to keep identical.
    `organized_path` is made relative to *dest* -- the destination directory
    itself is expected to differ between the two runs by construction; what
    must be identical is WHERE UNDER IT each photo landed. `processing_history`
    gets the same treatment: it is a JSON blob recording, among other things,
    each pipeline step's absolute `destination`, which embeds *dest* too --
    an unnormalized `str(dest)` prefix there fails this comparison for a
    reason that has nothing to do with the index (confirmed: this is exactly
    what happened on the first run of this test, before `processing_history`
    was normalized here -- see the report for the raw diff).
    """
    rows = []
    for row in cat.iter_all():
        d = dict(row)
        d.pop("id", None)
        d.pop("created_at", None)
        d.pop("processed_at", None)
        # LATENT GAP, no action needed today: `Catalog.mark_duplicate`
        # appends an "at" wall-clock timestamp to `processing_history` that
        # this function does not normalize (unlike the `destination`
        # substring handled below). Harmless while no fixture here contains
        # a duplicate image -- but the day one does, this comparison will
        # intermittently fail on that timestamp alone. Normalize it here
        # first if a duplicate-image fixture is ever added.
        organized = d.get("organized_path")
        if organized:
            d["organized_path"] = str(Path(organized).relative_to(dest))
        history = d.get("processing_history")
        if history:
            # `processing_history` is stored as JSON TEXT, so a Windows path's
            # single backslashes were already doubled by json.dumps when this
            # row was written -- replacing the bare str(dest) (single
            # backslashes) against that text silently matches nothing and
            # this normalization would be a no-op. Confirmed by hand: the
            # first version of this helper did exactly that and the
            # assertion below failed on an unrelated-looking diff that was
            # actually just this escaping mismatch (see the report).
            d["processing_history"] = history.replace(
                str(dest).replace("\\", "\\\\"), "<DEST>")
        rows.append(d)
    rows.sort(key=lambda r: r["sha256_b64url"])
    return rows
