"""Read a preserved Picasa face-tag roster as autocomplete vocabulary.

No roster file was present in the export this module was written and tested
against (`.superpowers/sdd/progress.md`, "FINDING: there is no Picasa roster
in this export" -- 175 archives scanned; the complete non-media inventory was
31 files, none resembling `contacts.xml` or any Picasa face-tag document).
This module targets Picasa's documented `contacts.xml` shape
(`<contact name="..." id="..."/>`) from documentation, not from an observed
file. A future reader adding a real roster export must verify the on-disk
shape against that export before trusting this parser -- do not assume the
format below matches without checking.

The roster names people across many entries and **carries no photo reference
at all**, so it can never be evidence about any image. Names enter `people`
with `source='picasa_roster'` purely so the review UI can offer them as
autocomplete vocabulary, and are never attached to a cluster or a photo.

Parsing never raises. A corrupt or absent supplementary document degrades to
"no names", the same discipline `exif_reader.read_exif` and
`takeout.metadata` follow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from xml.etree import ElementTree

from .names import normalize

logger = logging.getLogger(__name__)

PROVENANCE_DIR = ".takeout-provenance"
# Picasa's documented export name. Unverified against a real file -- see the
# module docstring.
ROSTER_NAMES = ("contacts.xml",)


def find_roster_files(dest: Path) -> list[Path]:
    """Locate every preserved roster under the organized root.

    Returns an empty list -- cleanly, no error -- when nothing matches. That
    is the expected case for this export, and for any export without a
    preserved Picasa roster.
    """
    root = Path(dest) / PROVENANCE_DIR
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.name in ROSTER_NAMES
    )


def parse_names(data: bytes) -> list[str]:
    """Extract normalized, de-duplicated names from a roster document.

    Never raises: malformed XML (or anything else that isn't a valid
    document) degrades to an empty list, matching the rest of this
    codebase's discipline for corrupt supplementary documents.
    """
    try:
        root = ElementTree.fromstring(data)
    except (ElementTree.ParseError, ValueError):
        return []

    seen: dict[str, None] = {}
    for element in root.iter():
        raw = element.get("name") or element.get("display_name") or ""
        cleaned = normalize(raw)
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen)


def import_names(store, dest: Path) -> int:
    """Add every roster name under *dest* to `people`. Returns how many were new.

    `FaceStore.add_person` always returns the person's id -- whether the row
    was just inserted or already existed (`INSERT OR IGNORE` followed by a
    `SELECT`) -- so it cannot itself signal idempotency. Idempotency is
    instead tracked here, against a snapshot of names already known before
    this call, updated as each new name is added.
    """
    existing = set(store.known_names())
    added = 0
    for path in find_roster_files(dest):
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.warning("could not read roster %s: %s", path.name, exc)
            continue
        for name in parse_names(data):
            if name in existing:
                continue
            store.add_person(name, "picasa_roster")
            existing.add(name)
            added += 1
    return added
