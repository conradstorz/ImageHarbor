"""Google Takeout archive ingestion.

A container walk rather than a filesystem walk. Three of the four modules here
are pure (`metadata`, `pairing`, and `archive.classify`) so the logic most
likely to be wrong can be tested exhaustively without a zip on disk;
`ingest` is the only module with side effects.

Archives are opened read-only and are never modified, moved, or written
alongside. Ingestion makes no AI calls, so -- exactly like the facts pass --
it never consults or feeds the circuit breaker.
"""

from __future__ import annotations
