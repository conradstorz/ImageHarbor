"""Quality tiers for date and descriptor provenance.

Two independent integer ladders decide what a re-run is allowed to change.
Ranks are spaced by 10 so a future source slots in without renumbering -- the
same append-only discipline the PCS taxonomy uses.

This module is pure: no I/O, and no imports from the rest of the package.
"""

from __future__ import annotations

# --- Date tier: decides placement -----------------------------------------
DATE_EXIF_ORIGINAL = 40      # EXIF DateTimeOriginal
DATE_EXTERNAL_SIDECAR = 30   # Google Takeout photoTakenTime, via ExternalEvidence.date
DATE_RELATED_SIDECAR = 25    # photoTakenTime from a RELATED file's sidecar -
                             # usually this file's unedited original, so the
                             # same photograph's capture instant. Above
                             # EXIF_OTHER, which records when a file was
                             # written rather than when a photo was taken.
                             # Deliberately breaks the 10-step spacing: this sits
                             # between two existing rungs, not at the end.
DATE_EXIF_OTHER = 20         # DateTimeDigitized, DateTime
DATE_FILENAME_PATTERN = 10   # date parsed out of the original filename
DATE_NONE = 0                # no trustworthy date -> Undated/

# File mtime is deliberately absent from this ladder. It is evidence of when a
# file was copied, not of when a photo was taken.

DATE_SOURCE_NAMES: dict[int, str] = {
    DATE_EXIF_ORIGINAL: "exif_original",
    DATE_EXTERNAL_SIDECAR: "external_sidecar",
    DATE_RELATED_SIDECAR: "related_sidecar",
    DATE_EXIF_OTHER: "exif_other",
    DATE_FILENAME_PATTERN: "filename_pattern",
    DATE_NONE: "none",
}

# --- Descriptor tier: decides the name ------------------------------------
DESC_HUMAN_FILENAME = 30     # original stem that no camera pattern matched
DESC_AI_SUBJECT = 20         # classifier primary_subject
DESC_NONE = 0                # nothing available

DESC_SOURCE_NAMES: dict[int, str] = {
    DESC_HUMAN_FILENAME: "human_filename",
    DESC_AI_SUBJECT: "ai_subject",
    DESC_NONE: "none",
}


def is_upgrade(old: tuple[int, int], new: tuple[int, int]) -> bool:
    """Return True if *new* is strictly better than *old*.

    Both arguments are ``(date_tier, descriptor_tier)``.  An upgrade requires
    a strict improvement in at least one dimension and no regression in either.
    Equality in both dimensions is NOT an upgrade, which is what makes a
    repeated run a no-op.
    """
    old_date, old_desc = old
    new_date, new_desc = new
    if new_date < old_date or new_desc < old_desc:
        return False
    return new_date > old_date or new_desc > old_desc
