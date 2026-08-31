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


def case_variants(names: Iterable[str]) -> dict[str, list[str]]:
    """Group normalized names that differ only by case.

    Returns ``{casefolded_key: [variant, ...]}`` for keys with more than one
    spelling, variants sorted for determinism. These are *suggestions* for the
    review UI; nothing here merges anything.
    """
    groups: dict[str, set[str]] = {}
    for raw in names:
        cleaned = normalize(raw)
        if not cleaned:
            continue
        groups.setdefault(cleaned.casefold(), set()).add(cleaned)
    return {k: sorted(v) for k, v in sorted(groups.items()) if len(v) > 1}
