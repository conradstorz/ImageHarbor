"""Match a Takeout media member to its Google JSON sidecar.

Pure: string work over member paths, no filesystem, no zip. Zip member paths
always use ``/`` regardless of the host OS, so this module splits on ``/``
directly and never touches ``pathlib``.

Google mutates sidecar names in several ways, and the rules below were verified
at 86/86 = 100% against the real export. The single most important rule is the
one that produces no answer: **if no rung yields exactly one match, return
None.** The member is then ingested from EXIF and its filename alone, which is
a fully correct outcome -- a photo without Google metadata is organized,
verified, and cataloged like any other. A WRONG pairing, by contrast, writes
another photo's capture date into this photo's name and folder, which is
exactly the quiet corruption the project's SHA-256 discipline exists to
prevent.

Rung order:

1. ``NAME(N).EXT`` -> ``NAME.EXT(N).json`` OR ``NAME.EXT.supplemental-metadata(N).json``
   (the copy suffix moves AFTER the extension, either directly before
   ``.json`` or after the ``supplemental-metadata`` tag -- both are exact,
   newer exports use the latter spelling)
2. ``NAME.EXT`` -> ``NAME.EXT.json``
3. ``NAME.EXT`` -> ``NAME.EXT.supplemental-metadata.json``  (newer exports)
4. ``-edited`` derivatives: strip the suffix and retry 1-3. Google emits no
   sidecar for an edited copy; it inherits the original's.
5. Case-insensitive retry of 1-4.
6. Truncation recovery: a unique prefix match among UNCLAIMED sidecars in the
   same directory. Google Photos truncates member stems at roughly 47
   characters, and truncates the media name and its sidecar name to different
   lengths.

Rungs 5 and 6 are swapped relative to the design document, deliberately: an
exact match that differs only in extension case is strictly stronger evidence
than a unique-prefix match, so running the fuzzier rung first could shadow it.
Rungs 1-4 keep the specified order.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)

_JSON_SUFFIX = ".json"
_SUPPLEMENTAL = ".supplemental-metadata.json"
_EDITED = "-edited"

# Google's copy suffix, e.g. "2015-03-09(1)".
_PAREN_RE = re.compile(r"^(?P<base>.*)\((?P<n>\d+)\)$")

# A truncation-recovery prefix shorter than this is not evidence, it is a
# coincidence: "abc.json" is a prefix of every "abc*" in the directory.
_MIN_TRUNCATION_PREFIX = 12


@dataclass
class PairingIndex:
    """Precomputed lookups over every member path in the batch.

    Holds only name->name strings: a 60 GB export can carry 100k+ members, and
    keeping the index at strings rather than parsed metadata keeps it near
    10 MB instead of hundreds. Sidecar CONTENT is read lazily, on demand, by
    the caller.
    """

    sidecars: frozenset[str] = frozenset()
    # Lowercased path -> the one real path, or None when two members differ
    # only in case (in which case a case-insensitive match would be a guess).
    sidecars_ci: Mapping[str, str | None] = field(default_factory=dict)
    # Directory -> sidecar paths inside it, for truncation recovery.
    by_dir: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Sidecars already matched exactly (rungs 1-5) by some media member.
    # Truncation recovery must never steal one of these.
    claimed: frozenset[str] = frozenset()
    # Media paths that occur more than once across the batch. The index is
    # keyed on bare member paths with no archive dimension, so two archives
    # sharing a media path are indistinguishable here -- `sidecar_for` refuses
    # to pair any of them rather than risk dating one archive's bytes with
    # another archive's sidecar.
    ambiguous_media: frozenset[str] = frozenset()
    # Sidecar paths that occur more than once across the batch. Same reasoning
    # as `ambiguous_media`, mirrored onto the other side of the pairing: the
    # index has no archive dimension, and the ingest layer's `self.owner` map
    # (member path -> owning archive) is last-writer-wins, so a sidecar path
    # duplicated across archives can silently hand a photo another archive's
    # sidecar bytes. Declining costs those members only their Google metadata
    # -- they still organize from EXIF and filename, and `missing_metadata`
    # makes the gap visible. A wrong date is permanent and silent, which is
    # strictly worse.
    ambiguous_sidecars: frozenset[str] = frozenset()


def _split(member_path: str) -> tuple[str, str]:
    """Return ``(directory, name)``; directory is ``""`` at the archive root."""
    directory, _, name = member_path.rpartition("/")
    return directory, name


def _is_sidecar(member_path: str) -> bool:
    return member_path.lower().endswith(_JSON_SUFFIX)


def _name_variants(name: str) -> list[str]:
    """The member name, then its pre-``-edited`` original (rung 4)."""
    variants = [name]
    base, dot, ext = name.rpartition(".")
    if dot and base.lower().endswith(_EDITED):
        variants.append(f"{base[: -len(_EDITED)]}.{ext}")
    return variants


def _candidates(media_path: str) -> list[str]:
    """Every exact sidecar path *media_path* could legitimately pair with."""
    directory, name = _split(media_path)
    prefix = f"{directory}/" if directory else ""
    out: list[str] = []
    for variant in _name_variants(name):
        base, dot, ext = variant.rpartition(".")
        if not dot:  # no extension at all
            base, ext = variant, ""
        # Rung 1 first: the (N) form is unambiguous and must not be shadowed
        # by the generic rule, which some exports ALSO satisfy.
        match = _PAREN_RE.match(base)
        if match and ext:
            out.append(f"{prefix}{match.group('base')}.{ext}({match.group('n')}){_JSON_SUFFIX}")
            # Newer exports write the copy marker AFTER the supplemental-
            # metadata tag instead: NAME.EXT.supplemental-metadata(N).json.
            out.append(
                f"{prefix}{match.group('base')}.{ext}"
                f".supplemental-metadata({match.group('n')}){_JSON_SUFFIX}"
            )
        out.append(f"{prefix}{variant}{_JSON_SUFFIX}")
        out.append(f"{prefix}{variant}{_SUPPLEMENTAL}")
    return out


def _media_part(sidecar_name: str) -> str:
    """The media-name portion of a sidecar's basename."""
    lower = sidecar_name.lower()
    if lower.endswith(_SUPPLEMENTAL):
        return sidecar_name[: -len(_SUPPLEMENTAL)]
    return sidecar_name[: -len(_JSON_SUFFIX)]


def _exact_match(media_path: str, index: PairingIndex) -> str | None:
    """Rungs 1-4, then rung 5 (the same candidates, case-insensitively)."""
    candidates = _candidates(media_path)
    for candidate in candidates:
        if candidate in index.sidecars:
            return candidate
    for candidate in candidates:
        hit = index.sidecars_ci.get(candidate.lower())
        if hit is not None:
            return hit
    return None


def _truncation_match(media_path: str, index: PairingIndex) -> str | None:
    """Rung 6: a unique prefix match among unclaimed sidecars in this directory."""
    directory, name = _split(media_path)
    matches = [
        sidecar
        for sidecar in index.by_dir.get(directory, ())
        if sidecar not in index.claimed
        and len(_media_part(_split(sidecar)[1])) >= _MIN_TRUNCATION_PREFIX
        and name.startswith(_media_part(_split(sidecar)[1]))
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        logger.debug(
            "Ambiguous truncation recovery for %s (%d candidates); no pairing",
            media_path, len(matches),
        )
    return None


def build_index(members: Iterable[str]) -> PairingIndex:
    """Build a pairing index over *members*.

    *members* is every member path in the batch -- media and sidecars, across
    every archive. Google's multi-part zips split by size across the file list,
    so ``IMG_1234.jpg`` can land in part 1 while ``IMG_1234.jpg.json`` lands in
    part 2; a per-archive index would silently lose metadata at every part
    boundary.
    """
    all_members = list(members)
    sidecars_seen = [m for m in all_members if _is_sidecar(m)]
    media = [m for m in all_members if not _is_sidecar(m)]

    # A sidecar path occurring more than once across the batch is unusable,
    # for the same reason a duplicated media path is unusable (see
    # `ambiguous_media` below): the index is keyed on bare member paths with
    # no archive dimension, and the ingest layer's owner map is
    # last-writer-wins, so a duplicated sidecar path can supply another
    # archive's bytes to a photo it was never shipped with. Partitioned out
    # BEFORE `sidecars`/`sidecars_ci`/`by_dir` are built, so no rung --
    # exact, case-insensitive, or truncation -- can ever match through it.
    sidecar_counts = Counter(sidecars_seen)
    ambiguous_sidecars = frozenset(
        path for path, count in sidecar_counts.items() if count > 1
    )
    sidecars = [m for m in sidecars_seen if m not in ambiguous_sidecars]

    sidecar_set = frozenset(sidecars)

    ci: dict[str, str | None] = {}
    for sidecar in sidecars:
        key = sidecar.lower()
        # A second member differing only in case makes a case-insensitive hit
        # a guess rather than a match, so poison the key instead.
        ci[key] = None if key in ci else sidecar

    by_dir: dict[str, list[str]] = {}
    for sidecar in sidecars:
        by_dir.setdefault(_split(sidecar)[0], []).append(sidecar)

    partial = PairingIndex(
        sidecars=sidecar_set,
        sidecars_ci=ci,
        by_dir={k: tuple(sorted(v)) for k, v in by_dir.items()},
    )

    # `_exact_match` reads only `partial.sidecars`/`sidecars_ci`, which are
    # already built from the FILTERED list above -- an ambiguous sidecar was
    # never added to either, so it cannot be matched here and cannot appear
    # in `claimed`. That is deliberate: `claimed` exists solely to stop
    # truncation recovery (rung 6) from stealing a sidecar some exact rung
    # already used, and a sidecar that can never be returned by `sidecar_for`
    # in the first place has nothing to protect.
    claimed = {
        match
        for media_path in media
        if (match := _exact_match(media_path, partial)) is not None
    }
    partial.claimed = frozenset(claimed)

    # A media path occurring in more than one archive cannot be paired: the
    # index is keyed on bare member paths, so two exports sharing a path are
    # indistinguishable here, and pairing one archive's sidecar to another
    # archive's bytes is precisely the silent misfiling this module exists to
    # refuse. Declining costs those members their Google metadata and nothing
    # else -- they still organize from EXIF and filename, and `missing_metadata`
    # makes the gap visible.
    seen: set[str] = set()
    ambiguous: set[str] = set()
    for m in media:
        if m in seen:
            ambiguous.add(m)
        else:
            seen.add(m)
    partial.ambiguous_media = frozenset(ambiguous)
    partial.ambiguous_sidecars = ambiguous_sidecars

    return partial


def sidecar_for(media_path: str, index: PairingIndex) -> str | None:
    """Return *media_path*'s sidecar member path, or None if none is certain."""
    if _is_sidecar(media_path):
        return None
    if media_path in index.ambiguous_media:
        return None
    exact = _exact_match(media_path, index)
    if exact is not None:
        return exact
    return _truncation_match(media_path, index)
