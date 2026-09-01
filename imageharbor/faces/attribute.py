"""Propose person names for clusters from Google's tags. Pure: no I/O.

This module only ever *proposes*. Nothing here writes an identity; that happens
only when a human confirms a cluster on the dashboard.

Every qualifying name is returned, not just the best one. Two people
photographed together always -- a couple, a pair of siblings -- score
identically, and picking the alphabetically-first would be an arbitrary
assertion dressed up as an answer. Offering both is the honest output.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .names import normalize


@dataclass(frozen=True)
class Proposal:
    cluster_id: int
    name: str
    support: int          # photos in this cluster tagged with this name
    total_tagged: int     # photos in this cluster tagged with anything
    score: float          # support / total_tagged
    untagged_photos: int  # what confirming would newly name -- the payoff


def propose(
    cluster_photos: Mapping[int, Sequence[str]],
    photo_names: Mapping[str, Sequence[str]],
    *,
    min_score: float,
    min_support: int,
) -> list[Proposal]:
    """Rank name proposals per cluster, best first."""
    out: list[Proposal] = []

    for cluster_id in sorted(cluster_photos):
        photos = list(dict.fromkeys(cluster_photos[cluster_id]))

        counts: Counter[str] = Counter()
        tagged = 0
        for digest in photos:
            # A name repeated on one photo is one photo's worth of evidence.
            found = {
                normalize(n) for n in photo_names.get(digest, ()) if normalize(n)
            }
            if found:
                tagged += 1
                counts.update(found)

        if tagged == 0:
            continue

        for name, support in counts.items():
            score = support / tagged
            if score >= min_score and support >= min_support:
                out.append(
                    Proposal(
                        cluster_id=cluster_id,
                        name=name,
                        support=support,
                        total_tagged=tagged,
                        score=score,
                        untagged_photos=len(photos) - tagged,
                    )
                )

    out.sort(key=lambda p: (p.cluster_id, -p.score, p.name))
    return out
