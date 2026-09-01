"""Person-name normalization. Pure: no I/O, no imports from the package.

Two defects are present in this library's real name vocabulary, and they are
handled asymmetrically on purpose.

Whitespace is noise. `Gladys Blankenbeker ` carries a trailing space in all 461
of its occurrences, and the sidecar's `people` list is keyed on the name, so an
unnormalized key silently splits one person into two entries. Stripping it
cannot merge two different people, so it is applied automatically.

Case might not be noise. `pete storz` (1,539) and `claire Storz` (442) look like
drift, but this same vocabulary contains `Conrad Storz` (3,309) and `Conrad
Storz III` (980) -- a father and a son distinguished only by a suffix. A
vocabulary that proves suffixes are load-bearing is not one to apply automatic
identity judgements to. Case variants are therefore *reported* by
`case_variants` for a human to confirm, never folded.
"""

from __future__ import annotations

from collections.abc import Iterable


def normalize(name: str) -> str:
    """Strip surrounding whitespace and collapse internal runs. Case is kept."""
    return " ".join(name.split())


def _case_key(name: str) -> tuple[int, str]:
    """Key that matches two strings only when they differ *purely* by case.

    ``str.casefold()`` is Unicode-normalizing, not case-folding: it merges
    strings of different length, e.g. ``'Weiß'`` and ``'Weiss'``.
    Per-character ``str.lower()`` doesn't expand or contract characters the
    way casefold does, so pairing it with the original length catches that
    length-changing case. It does *not* catch same-length compatibility
    collisions -- the Kelvin sign (U+212A) still keys the same as ``'K'``,
    because Unicode's simple case mapping sends both to ``'k'``. No
    per-character scheme can separate them without giving up case-insensitive
    comparison. That's acceptable here: ``case_variants`` only ever suggests
    a merge to a human, it never performs one.
    """
    return (len(name), "".join(ch.lower() for ch in name))


def case_variants(names: Iterable[str]) -> dict[str, list[str]]:
    """Group normalized names that differ only by case.

    Returns ``{lowercased_key: [variant, ...]}`` for keys with more than one
    spelling, variants sorted for determinism. These are *suggestions* for the
    review UI; nothing here merges anything.
    """
    groups: dict[tuple[int, str], set[str]] = {}
    for raw in names:
        cleaned = normalize(raw)
        if not cleaned:
            continue
        groups.setdefault(_case_key(cleaned), set()).add(cleaned)
    return {
        lower: sorted(variants)
        for (_, lower), variants in sorted(groups.items())
        if len(variants) > 1
    }
