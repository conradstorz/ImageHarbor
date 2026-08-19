"""Operational dashboard: reporting, projection, and a small control gateway.

Served in-process by `watch` on a daemon thread. The split here mirrors the
rest of the project: the module most likely to be wrong (`projections`) has no
I/O and is table-tested, while `server` owns the sockets.

Nothing in this package may stop the watcher. A dashboard that cannot start,
cannot render, or cannot query still leaves photos being organized -- the
observability layer is subordinate to the work.
"""
