"""Runtime control: the pause flag and the settings that override config.

Precedence is deliberately simple and deliberately visible. An env var supplies
the value at first start; a dashboard change writes a row that wins from then
on. The failure mode that creates is obvious and nasty -- edit
docker-compose.yml, restart, and nothing changes because a stored override from
months ago is silently winning -- so `overrides()` reports both values and the
UI shows the conflict wherever it is in effect.

Reverting an override DELETES the row rather than writing the env value into
it. Writing it back would look identical today and shadow every future config
change.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from imageharbor.catalog import Catalog

logger = logging.getLogger(__name__)

# Keys stored in the `settings` table. 'paused' has no env counterpart -- it
# is pure dashboard state, not a config override -- so it is handled
# separately from 'interval'/'enrich'/'faces' throughout this module.
_PAUSED_KEY = "paused"
_INTERVAL_KEY = "interval"
_ENRICH_KEY = "enrich"
# 'faces' mirrors 'enrich' exactly: same '0'/'1' storage, same env-vs-stored
# precedence, same hostile-value handling -- the faces pass is a third
# on/off dial alongside enrichment, not a different kind of setting.
_FACES_KEY = "faces"


def _parse_interval(raw: Any) -> float | None:
    """Parse a stored interval string to a positive, finite float.

    Returns None for anything that cannot be trusted as a real interval:
    unparseable text, NaN, +/-Infinity, zero, or negative. Zero or negative
    is treated as unreadable rather than "instant" -- a 0-second poll
    interval would spin the watcher loop as fast as the CPU allows, which is
    exactly the kind of quiet corruption this project's discipline exists to
    prevent. Infinity is rejected for a sharper reason: it is not merely a
    silly interval, it is not representable at all where an interval is
    actually consumed -- ``threading.Event.wait(math.inf)`` raises
    ``OverflowError`` (a `float` timestamp/timeout must fit a platform
    `time_t`), so a stored `inf` must never reach a caller as if it were a
    real, if extreme, number. `math.isfinite` is the single check that
    correctly rejects NaN and both infinities together -- see
    `set_override` below, which enforces the same rule at write time. A
    huge but well-formed *finite* value (e.g. "1e9") is still accepted: it
    is a silly interval, not an unreadable one, and an operator who set it
    gets what they asked for.
    """
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if value <= 0:
        return None
    return value


def _parse_bool_flag(raw: Any) -> bool | None:
    """Parse a stored '0'/'1' flag. Anything else is unreadable, not a guess.

    Only the two canonical stored forms round-trip. A hand-edited value like
    "maybe", "", "yes", or "2" is not obviously true or false, so it must not
    be coerced toward either -- it is treated as absent/unreadable and the
    caller falls back to its own default (env value, or "not paused").
    """
    if raw is None:
        return None
    if raw == "1":
        return True
    if raw == "0":
        return False
    return None


class ControlPlane:
    """Holds the pause flag and the interval/enrich override state.

    One instance is shared, in-process, between the HTTP dashboard thread and
    the watch loop (see docs/superpowers/specs/2026-08-19-dashboard-design.md,
    "Architecture") -- that is what makes `paused` authoritative rather than
    eventually-consistent. `interval` and `enrich_enabled` are read fresh from
    the catalog on every access (never cached) so a dashboard change takes
    effect on the *next* sleep/pass, per the design's "Takes effect" column,
    rather than being frozen at construction time.
    """

    def __init__(
        self,
        catalog: Catalog,
        *,
        env_interval: float,
        env_enrich: bool,
        env_faces: bool = False,
    ) -> None:
        self._catalog = catalog
        self._env_interval = env_interval
        self._env_enrich = env_enrich
        # Defaults to False (unlike env_enrich, which every existing caller
        # passes explicitly): faces is a new, heavier, opt-in extra -- a
        # caller that hasn't been updated to know about it yet must not
        # silently start running face detection.
        self._env_faces = env_faces
        # Seed the in-memory pause flag from the durable row so a restart
        # honors a pause made before the process died (see
        # test_pause_survives_a_restart). After construction this flag is the
        # sole authority for pause_check() -- it is never re-read from the
        # database.
        raw_paused = catalog.setting_get(_PAUSED_KEY)
        stored = _parse_bool_flag(raw_paused)
        if raw_paused is not None and stored is None:
            logger.warning(
                "stored 'paused' setting %r is unreadable; starting unpaused",
                raw_paused,
            )
        self._paused = bool(stored) if stored is not None else False

    # ------------------------------------------------------------------
    # pause
    # ------------------------------------------------------------------

    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, value: bool) -> None:
        self._paused = bool(value)
        self._catalog.setting_set(_PAUSED_KEY, "1" if self._paused else "0")

    def pause_check(self) -> bool:
        """True when the caller (facts pass / enrichment pass) should stop.

        Reads the in-memory flag only. This is called between every photo, so
        a database query here would put SQLite in the inner loop of the
        facts pass -- the database copy exists purely for restart durability
        (see __init__); it is never consulted here.
        """
        return self._paused

    # ------------------------------------------------------------------
    # interval / enrich
    # ------------------------------------------------------------------

    def _resolve_interval(self) -> tuple[float, bool]:
        raw = self._catalog.setting_get(_INTERVAL_KEY)
        if raw is None:
            return self._env_interval, False
        parsed = _parse_interval(raw)
        if parsed is None:
            logger.warning(
                "stored 'interval' setting %r is unreadable; falling back "
                "to env value %r",
                raw, self._env_interval,
            )
            return self._env_interval, False
        return parsed, True

    def _resolve_enrich(self) -> tuple[bool, bool]:
        raw = self._catalog.setting_get(_ENRICH_KEY)
        if raw is None:
            return self._env_enrich, False
        parsed = _parse_bool_flag(raw)
        if parsed is None:
            logger.warning(
                "stored 'enrich' setting %r is unreadable; falling back "
                "to env value %r",
                raw, self._env_enrich,
            )
            return self._env_enrich, False
        return parsed, True

    def _resolve_faces(self) -> tuple[bool, bool]:
        raw = self._catalog.setting_get(_FACES_KEY)
        if raw is None:
            return self._env_faces, False
        parsed = _parse_bool_flag(raw)
        if parsed is None:
            logger.warning(
                "stored 'faces' setting %r is unreadable; falling back "
                "to env value %r",
                raw, self._env_faces,
            )
            return self._env_faces, False
        return parsed, True

    @property
    def interval(self) -> float:
        value, _ = self._resolve_interval()
        return value

    @property
    def enrich_enabled(self) -> bool:
        value, _ = self._resolve_enrich()
        return value

    @property
    def faces_enabled(self) -> bool:
        value, _ = self._resolve_faces()
        return value

    # ------------------------------------------------------------------
    # override precedence
    # ------------------------------------------------------------------

    def set_override(self, key: str, value: Any) -> None:
        if key == _INTERVAL_KEY:
            try:
                interval = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"interval must be numeric, got {value!r}") from exc
            # math.isfinite is the one check that rejects NaN AND both
            # +/-Infinity -- see _parse_interval's docstring for why a
            # stored Infinity is not just "silly" but actively unsafe: it
            # reaches Event.wait() downstream and raises OverflowError there.
            if not math.isfinite(interval) or interval <= 0:
                raise ValueError(
                    f"interval must be a positive, finite number, got {value!r}"
                )
            self._catalog.setting_set(_INTERVAL_KEY, str(interval))
        elif key == _ENRICH_KEY:
            self._catalog.setting_set(_ENRICH_KEY, "1" if value else "0")
        elif key == _FACES_KEY:
            self._catalog.setting_set(_FACES_KEY, "1" if value else "0")
        elif key == _PAUSED_KEY:
            self.set_paused(bool(value))
        else:
            raise ValueError(f"unknown setting key: {key!r}")

    def revert(self, key: str) -> None:
        """Delete the stored override so the env var (or default) applies.

        Writing the env value into the row instead would look identical
        today and silently shadow every future config change -- see the
        module docstring. `Catalog.setting_delete` is idempotent, so
        reverting a key with no stored override is a no-op.
        """
        self._catalog.setting_delete(key)
        if key == _PAUSED_KEY:
            # 'paused' has no env fallback to "restore" -- absence means
            # the default, unpaused, state.
            self._paused = False

    def overrides(self) -> dict[str, dict]:
        """Per-key `{"value", "env_value", "overridden"}` for the UI.

        Only keys with an env counterpart participate -- 'paused' has none,
        so it is not part of the "stored value silently beats a compose
        change" warning this method exists to surface.
        """
        interval_value, interval_overridden = self._resolve_interval()
        enrich_value, enrich_overridden = self._resolve_enrich()
        faces_value, faces_overridden = self._resolve_faces()
        return {
            _INTERVAL_KEY: {
                "value": interval_value,
                "env_value": self._env_interval,
                "overridden": interval_overridden,
            },
            _ENRICH_KEY: {
                "value": enrich_value,
                "env_value": self._env_enrich,
                "overridden": enrich_overridden,
            },
            _FACES_KEY: {
                "value": faces_value,
                "env_value": self._env_faces,
                "overridden": faces_overridden,
            },
        }
