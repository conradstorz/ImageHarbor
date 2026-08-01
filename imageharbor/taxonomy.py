"""Extensible PCS taxonomy backed by the catalog `taxonomy` table.

Codes are strings matching ``^\\d+(~\\d+)*$``: plain integers for the common
three-level case (parent 500 -> sub 540 -> leaf 541), and a ``~N`` suffix for
overflow / depth. The 9 top-level classes are fixed; growth happens beneath and
is append-only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import Catalog
from .pcs import PCS_CATEGORIES

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_TOP_RE = re.compile(r"^[1-9]00$")
_SUB_RE = re.compile(r"^[1-9][1-9]0$")


def slug(label: str) -> str:
    """Filesystem-safe folder slug: lowercase, ascii-alnum, hyphen-joined."""
    return _SLUG_RE.sub("-", label.lower()).strip("-") or "unnamed"


@dataclass
class TaxonomyNode:
    code: str
    parent_code: str | None
    label: str
    folder_name: str
    aliases: list[str] = field(default_factory=list)
    alias_of: str | None = None
    active: bool = True


def _node(row) -> TaxonomyNode:
    import json
    return TaxonomyNode(
        code=row["code"],
        parent_code=row["parent_code"],
        label=row["label"],
        folder_name=row["folder_name"],
        aliases=json.loads(row["aliases"]),
        alias_of=row["alias_of"],
        active=bool(row["active"]),
    )


class Taxonomy:
    """Catalog-backed taxonomy registry."""

    def __init__(self, catalog: Catalog) -> None:
        self._cat = catalog

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def ensure_seeded(self) -> None:
        """Seed from the legacy PCS_CATEGORIES on first use."""
        if not self._cat.taxonomy_is_empty():
            return
        # Insert parents first (parent_code None), then children.
        for code, cat in sorted(PCS_CATEGORIES.items()):
            parent = None if cat.parent is None else str(cat.parent)
            self._cat.taxonomy_insert(
                str(code), parent, cat.name, f"{code}-{slug(cat.name)}"
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, code: str) -> TaxonomyNode | None:
        row = self._cat.taxonomy_get(code)
        return _node(row) if row else None

    def children(self, parent_code: str | None) -> list[TaxonomyNode]:
        return [_node(r) for r in self._cat.taxonomy_children(parent_code)]

    def resolve_alias(self, code: str) -> str:
        """Follow the alias_of chain to the canonical active code."""
        seen: set[str] = set()
        cur = code
        while cur and cur not in seen:
            seen.add(cur)
            node = self.get(cur)
            if node is None or node.alias_of is None:
                return cur
            cur = node.alias_of
        return cur

    def folder_path(self, code: str) -> str:
        """Slash-joined folder path from the top-level ancestor to `code`."""
        parts: list[str] = []
        cur: str | None = code
        while cur:
            node = self.get(cur)
            if node is None:
                break
            parts.append(node.folder_name)
            cur = node.parent_code
        return "/".join(reversed(parts))

    # ------------------------------------------------------------------
    # Numbering (append-only)
    # ------------------------------------------------------------------

    def _integer_child_slots(self, parent_code: str) -> list[str]:
        if _TOP_RE.match(parent_code):
            base = int(parent_code)
            return [str(base + 10 * k) for k in range(1, 10)]
        if _SUB_RE.match(parent_code):
            base = int(parent_code)
            return [str(base + k) for k in range(1, 10)]
        return []  # leaf or ~-extended node: decimal children only

    def mint_child(self, parent_code: str, label: str) -> str:
        existing = {n.code for n in self.children(parent_code)}
        for candidate in self._integer_child_slots(parent_code):
            if candidate not in existing:
                return self._create(candidate, parent_code, label)
        # Overflow / depth: next ~N under parent_code
        n = 0
        prefix = parent_code + "~"
        for c in existing:
            if c.startswith(prefix) and c[len(prefix):].isdigit():
                n = max(n, int(c[len(prefix):]))
        return self._create(f"{parent_code}~{n + 1}", parent_code, label)

    def _create(self, code: str, parent_code: str, label: str) -> str:
        self._cat.taxonomy_insert(code, parent_code, label, f"{code}-{slug(label)}")
        return code
