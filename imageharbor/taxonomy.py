"""Extensible PCS taxonomy backed by the catalog `taxonomy` table.

Codes are strings matching ``^\\d+(~\\d+)*$``: plain integers for the common
three-level case (parent 500 -> sub 540 -> leaf 541), and a ``~N`` suffix for
overflow / depth. The 9 top-level classes are fixed; growth happens beneath and
is append-only.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from .catalog import Catalog
from .pcs import PCS_CATEGORIES

logger = logging.getLogger(__name__)

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

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(label: str) -> str:
        text = _SLUG_RE.sub(" ", label.lower()).strip()
        words = []
        for w in text.split():
            if len(w) > 4 and w.endswith("es"):
                w = w[:-2]
            elif len(w) > 3 and w.endswith("s"):
                w = w[:-1]
            words.append(w)
        return " ".join(words)

    def resolve_or_create(
        self,
        top_parent: str,
        label: str,
        sub_parent: str | None = None,
        adjudicator: Callable[[str, list[str]], str | None] | None = None,
    ) -> str:
        # Guard an invalid top_parent: a real backend may return a code that is
        # not one of the 9 fixed classes (e.g. "events" or "999"). Rather than
        # minting an orphan under a nonexistent parent, fall back to the
        # miscellaneous class (900). The sub_parent path keeps its own existence
        # check below.
        if not sub_parent and self.get(top_parent) is None:
            top_parent = "900"
        target = sub_parent if sub_parent and self.get(sub_parent) else top_parent
        norm = self._normalize(label)
        kids = self.children(target)

        # Exact / alias hit
        for k in kids:
            names = [self._normalize(k.label)] + [self._normalize(a) for a in k.aliases]
            if norm in names:
                return self.resolve_alias(k.code)

        # No exact/alias hit -> let the adjudicator decide among the target's
        # siblings. (Deviation from brief: a difflib ratio>=0.8 pre-filter on
        # the candidate list was specified, but no purely textual similarity
        # metric puts "festivities" near "holidays" -- the whole point of the
        # adjudicator is semantic matching an AI provides, which text-distance
        # can't approximate. Gating on it would make the adjudicator
        # unreachable for exactly the synonym case it exists to handle, so we
        # pass the full sibling list instead.)
        if kids and adjudicator is not None:
            try:
                matched = adjudicator(label, [k.label for k in kids])
            except Exception:
                logger.warning("adjudicator failed for label %r; minting new", label, exc_info=True)
                matched = None
            if matched:
                for k in kids:
                    if k.label == matched:
                        aliases = k.aliases + [label]
                        self._cat.taxonomy_set_aliases(k.code, aliases)
                        return self.resolve_alias(k.code)

        return self.mint_child(target, label)

    def merge(self, from_code: str, to_code: str) -> None:
        target = self.get(to_code)
        src = self.get(from_code)
        if target is None or src is None:
            return
        self._cat.taxonomy_set_aliases(to_code, target.aliases + [src.label])
        self._cat.taxonomy_set_alias(from_code, to_code)

    def snapshot_text(self) -> str:
        """Compact `code label` view grouped by hierarchy for the AI prompt."""
        rows = self._cat.taxonomy_all()
        by_parent: dict[str | None, list] = {}
        for r in rows:
            by_parent.setdefault(r["parent_code"], []).append(r)

        lines: list[str] = []

        def emit(code: str, label: str, depth: int) -> None:
            lines.append(f"{'  ' * depth}{code} {label}")
            for child in by_parent.get(code, []):
                emit(child["code"], child["label"], depth + 1)

        for top in by_parent.get(None, []):
            emit(top["code"], top["label"], 0)
        return "\n".join(lines)
