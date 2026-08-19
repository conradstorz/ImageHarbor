"""Tests for imageharbor.dashboard.control.ControlPlane.

Two things make this module easy to get quietly wrong (see the module
docstring and the design doc): reverting an override must DELETE the settings
row rather than write the env value back into it, and `pause_check()` must
read the in-memory flag rather than the database, because it is called
between every photo. The tests below are written so that each one fails if
either of those shortcuts is taken.

The settings table is also something a human can hand-edit. A stored
`interval` of `"abc"`, `""`, `"-5"`, `"0"`, or `"1e9"` and a stored `paused`
of `"maybe"` must each resolve to something deliberate, never to a silent 0
or a crash -- see the "hostile values" section at the bottom.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from imageharbor.catalog import Catalog
from imageharbor.dashboard.control import ControlPlane


@pytest.fixture()
def catalog(tmp_path: Path) -> Catalog:
    db = tmp_path / "test_catalog.db"
    cat = Catalog(db)
    yield cat
    cat.close()


# ---------------------------------------------------------------------------
# override precedence
# ---------------------------------------------------------------------------


def test_no_override_follows_the_env_value(catalog: Catalog) -> None:
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    assert control.interval == 300
    assert control.enrich_enabled is True


def test_an_override_wins_over_the_env_value(catalog: Catalog) -> None:
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    control.set_override("interval", 120)
    assert control.interval == 120

    control.set_override("enrich", False)
    assert control.enrich_enabled is False


def test_reverting_restores_the_env_value(catalog: Catalog) -> None:
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    control.set_override("interval", 120)
    control.revert("interval")
    assert control.interval == 300


def test_reverting_deletes_the_row_so_a_later_config_change_is_seen(catalog: Catalog) -> None:
    """The trap this avoids: edit docker-compose, restart, nothing changes.

    If revert wrote the env value into the table instead of deleting the row,
    the stored copy would shadow every future config change silently.
    """
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    control.set_override("interval", 120)
    control.revert("interval")
    assert catalog.setting_get("interval") is None
    # a later "config change" is now visible
    assert ControlPlane(catalog, env_interval=600, env_enrich=True).interval == 600


def test_reverting_a_key_with_no_stored_override_is_a_noop(catalog: Catalog) -> None:
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    control.revert("interval")  # must not raise
    assert control.interval == 300


# ---------------------------------------------------------------------------
# pause
# ---------------------------------------------------------------------------


def test_pause_survives_a_restart(catalog: Catalog) -> None:
    """A container that comes back running after a deliberate pause is the
    kind of surprise that makes an operator stop trusting the button."""
    ControlPlane(catalog, env_interval=300, env_enrich=True).set_paused(True)
    assert ControlPlane(catalog, env_interval=300, env_enrich=True).paused is True


def test_unpaused_by_default(catalog: Catalog) -> None:
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    assert control.paused is False
    assert control.pause_check() is False


def test_pause_check_returns_true_when_paused(catalog: Catalog) -> None:
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    control.set_paused(True)
    assert control.pause_check() is True


def test_pause_check_reflects_an_in_memory_change_without_reconstruction(catalog: Catalog) -> None:
    """`pause_check()` must observe `set_paused` on the SAME instance
    immediately -- it is polled from the same in-memory object the HTTP
    handler and the run loop share, in the same process. There is no second
    `ControlPlane` instance and no DB round trip involved in this path."""
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    assert control.pause_check() is False
    control.set_paused(True)
    assert control.pause_check() is True
    control.set_paused(False)
    assert control.pause_check() is False


def test_pause_check_does_not_query_the_database(catalog: Catalog, monkeypatch) -> None:
    """`pause_check()` is called between every photo; a query there would put
    SQLite in the inner loop of the facts pass. Sabotage `setting_get` after
    construction and confirm `pause_check()` still works from memory."""
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    control.set_paused(True)

    def _boom(key: str) -> None:
        raise AssertionError("pause_check() must not read the database")

    monkeypatch.setattr(catalog, "setting_get", _boom)
    assert control.pause_check() is True


def test_reverting_paused_returns_to_unpaused(catalog: Catalog) -> None:
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    control.set_paused(True)
    control.revert("paused")
    assert control.paused is False
    assert catalog.setting_get("paused") is None


# ---------------------------------------------------------------------------
# overrides() for the UI
# ---------------------------------------------------------------------------


def test_overrides_report_both_values_for_the_ui(catalog: Catalog) -> None:
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    control.set_override("interval", 120)

    overrides = control.overrides()

    assert overrides["interval"] == {"value": 120, "env_value": 300, "overridden": True}
    assert overrides["enrich"] == {"value": True, "env_value": True, "overridden": False}


def test_overrides_reports_not_overridden_when_nothing_is_stored(catalog: Catalog) -> None:
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    overrides = control.overrides()
    assert overrides["interval"]["overridden"] is False
    assert overrides["interval"]["value"] == 300
    assert overrides["enrich"]["overridden"] is False
    assert overrides["enrich"]["value"] is True


def test_overrides_does_not_include_pause_as_a_config_override(catalog: Catalog) -> None:
    """Pause has no env counterpart to conflict with -- it is not part of the
    "edit compose, restart, nothing happens" trap this dict exists to
    surface, so it does not belong in the UI's override warning."""
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    control.set_paused(True)
    assert "paused" not in control.overrides()


# ---------------------------------------------------------------------------
# unreadable stored values: absent vs. present-but-unreadable vs. genuinely
# invalid, never silently coerced to 0/None/False
# ---------------------------------------------------------------------------


def test_an_unparseable_stored_value_falls_back_to_env(catalog: Catalog) -> None:
    """A hand-edited settings row must not break startup."""
    catalog.setting_set("interval", "not-a-number")
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    assert control.interval == 300


@pytest.mark.parametrize("hostile_interval", ["abc", "", "-5", "0", "nan"])
def test_hostile_stored_intervals_fall_back_to_env_not_zero(
    catalog: Catalog, hostile_interval: str, caplog
) -> None:
    """None of these must ever produce 0 -- a 0-second poll interval would
    spin the watcher loop as fast as the CPU allows."""
    catalog.setting_set("interval", hostile_interval)
    with caplog.at_level(logging.WARNING):
        control = ControlPlane(catalog, env_interval=300, env_enrich=True)
        assert control.interval == 300
    assert control.interval != 0
    assert "interval" in caplog.text.lower()


def test_a_huge_but_well_formed_stored_interval_is_accepted(catalog: Catalog) -> None:
    """1e9 is a silly poll interval, not an unreadable one -- an operator who
    set it gets what they asked for rather than a second-guessed fallback."""
    catalog.setting_set("interval", "1e9")
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    assert control.interval == pytest.approx(1e9)


@pytest.mark.parametrize("hostile_paused", ["maybe", "", "yes", "2"])
def test_hostile_stored_paused_values_fall_back_to_unpaused(
    catalog: Catalog, hostile_paused: str, caplog
) -> None:
    """A stored `paused` that is neither '0' nor '1' cannot be trusted as a
    deliberate pause, so a hand-edited row must not silently start the
    container paused (or silently un-pause a real one -- see the equivalent
    'enrich' case below for the other direction)."""
    catalog.setting_set("paused", hostile_paused)
    with caplog.at_level(logging.WARNING):
        control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    assert control.paused is False
    assert "paused" in caplog.text.lower()


@pytest.mark.parametrize("hostile_enrich", ["maybe", "", "yes", "2"])
def test_hostile_stored_enrich_values_fall_back_to_env(
    catalog: Catalog, hostile_enrich: str, caplog
) -> None:
    catalog.setting_set("enrich", hostile_enrich)
    with caplog.at_level(logging.WARNING):
        control = ControlPlane(catalog, env_interval=300, env_enrich=False)
        assert control.enrich_enabled is False
        control2 = ControlPlane(catalog, env_interval=300, env_enrich=True)
        assert control2.enrich_enabled is True


def test_set_override_rejects_a_non_positive_interval(catalog: Catalog) -> None:
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    with pytest.raises(ValueError):
        control.set_override("interval", 0)
    with pytest.raises(ValueError):
        control.set_override("interval", -5)
    # neither rejected write left a row behind
    assert catalog.setting_get("interval") is None
    assert control.interval == 300


def test_set_override_rejects_an_unknown_key(catalog: Catalog) -> None:
    control = ControlPlane(catalog, env_interval=300, env_enrich=True)
    with pytest.raises(ValueError):
        control.set_override("bogus", 1)
