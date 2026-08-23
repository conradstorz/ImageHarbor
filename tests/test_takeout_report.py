"""Report logic is pure: an inventory in, a document out. No filesystem."""

from collections import Counter

from imageharbor.takeout import report

# Fields that are Counters on SurveyInventory. The helper below wraps plain
# dicts so a test can pass {".screen": 3963} without losing .most_common().
_COUNTER_FIELDS = {
    "ext_counts", "ext_bytes", "kind_counts", "kind_bytes", "area_counts",
    "misnamed_counts", "misnamed_bytes", "timestamp_counts", "year_counts",
    "misnamed_kind_counts", "misnamed_kind_bytes",
}


def _inv(**overrides):
    inv = report.SurveyInventory()
    for key, value in overrides.items():
        if key in _COUNTER_FIELDS and not isinstance(value, Counter):
            value = Counter(value)
        setattr(inv, key, value)
    return inv


# --- distrusted timestamp clusters ---------------------------------------

def test_cluster_at_threshold_is_distrusted():
    counts = {"1968-01-12T10:35:03": 25}
    assert report.find_distrusted_timestamps(counts, 25) == frozenset({"1968-01-12T10:35:03"})


def test_cluster_below_threshold_is_not_distrusted():
    counts = {"1968-01-12T10:35:03": 24}
    assert report.find_distrusted_timestamps(counts, 25) == frozenset()


def test_a_burst_of_shots_sharing_a_second_is_not_a_cluster():
    """Real bursts share a second; they do not share it 200 times."""
    counts = {"2019-07-04T12:33:11": 4, "2019-07-04T12:33:12": 6}
    assert report.find_distrusted_timestamps(counts, 25) == frozenset()


def test_multiple_clusters_are_all_reported():
    counts = {"1968-01-12T10:35:03": 210, "2000-01-01T00:00:00": 40, "2019-01-01T09:00:00": 3}
    assert report.find_distrusted_timestamps(counts, 25) == frozenset(
        {"1968-01-12T10:35:03", "2000-01-01T00:00:00"}
    )


def test_threshold_of_zero_does_not_distrust_everything():
    """A zero threshold disables the rule rather than condemning every date."""
    counts = {"2019-01-01T09:00:00": 3}
    assert report.find_distrusted_timestamps(counts, 0) == frozenset()


# --- refuse to guess ------------------------------------------------------

def test_duplicates_are_reported_as_an_upper_bound_never_as_a_count():
    inv = _inv(basename_collisions=9850, basename_collision_members=22402)
    doc = report.build_report(inv, distrust_threshold=25)
    dupes = doc["projection"]["duplicates"]
    # 22,402 members across 9,850 colliding names: at most one of each name is
    # an original, so the bound is 12,552 -- not the whole colliding population.
    assert dupes["name_collision_upper_bound"] == 12552
    assert dupes["colliding_members"] == 22402
    assert dupes["exact"] is None
    assert "upper bound" in dupes["note"].lower()


def test_an_empty_archive_set_is_labelled_empty_not_reported_as_a_clean_result():
    """Zero members and "no archives found" must not read the same."""
    doc = report.build_report(_inv(), distrust_threshold=25)
    assert doc["archives"]["status"] == "empty"
    assert doc["projection"]["organized_today"] == 0
    assert doc["projection"]["organized_after_deferred_fixes"] == 0


# --- sequence gaps --------------------------------------------------------

def test_a_gap_filled_by_a_loose_part_is_not_reported_as_missing():
    inv = _inv(
        part_numbers={"001", "002", "004"},
        loose_parts=[report.LoosePart(name="VID-003.mp4", size=10, part="003", kind="video")],
    )
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["missing_parts"] == []
    assert doc["archives"]["loose_parts"] == 1


def test_a_genuinely_missing_part_is_reported():
    inv = _inv(part_numbers={"001", "002", "004"}, loose_parts=[])
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["missing_parts"] == ["003"]


def test_mixed_width_part_numbers_use_the_max_observed_width():
    """Width must not come from an arbitrary set element.

    {"1", "2", "004"} mixes an unpadded "1"/"2" with a zero-padded "004". The
    gap (3) must be reported at the max width actually observed (3, from
    "004"), which makes the result order-independent BY CONSTRUCTION -- width
    comes from max(), never from set iteration order.

    This is asserted once, not in a loop: PYTHONHASHSEED is fixed for the life
    of a process, so repeating the same call in one process re-uses the same
    set ordering and proves nothing about run-to-run variation. The property
    that matters is structural, and reading the implementation is how it is
    checked.
    """
    assert report._missing_parts({"1", "2", "004"}, []) == ["003"]


def test_mixed_width_part_numbers_via_build_report():
    inv = _inv(part_numbers={"1", "2", "004"}, loose_parts=[])
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["missing_parts"] == ["003"]


def test_a_non_numeric_part_entry_does_not_crash():
    inv = _inv(part_numbers={"001", "002", "004", "unknown"}, loose_parts=[])
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["missing_parts"] == ["003"]


def test_format_summary_does_not_raise_on_a_completely_empty_report():
    text = report.format_summary(report.build_report(report.SurveyInventory(), distrust_threshold=25))
    assert isinstance(text, str)


# --- misnamed media -------------------------------------------------------

def test_misnamed_media_is_surfaced_by_declared_extension():
    inv = _inv(misnamed_counts={".screen": 3963, ".tile": 26})
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["anomalies"]["misnamed_media"]["total"] == 3989
    assert doc["anomalies"]["misnamed_media"]["by_extension"][".screen"] == 3963


# --- descriptor distribution ---------------------------------------------

def test_descriptor_distribution_is_reported_without_judging_it():
    inv = _inv(descriptor_human=47460, descriptor_machine=31751)
    doc = report.build_report(inv, distrust_threshold=25)
    desc = doc["anomalies"]["descriptors"]
    assert desc["human_tier30"] == 47460
    assert desc["machine_tier0"] == 31751
    assert round(desc["human_share"], 2) == 0.6


def test_descriptor_share_is_zero_when_nothing_was_named():
    """Not a division by zero, and not a misleading 100%."""
    doc = report.build_report(_inv(), distrust_threshold=25)
    assert doc["anomalies"]["descriptors"]["human_share"] == 0.0


# --- summary --------------------------------------------------------------

def test_format_summary_is_text_and_mentions_the_headline_numbers():
    inv = _inv(kind_counts={"image": 5, "video": 2}, archives=[
        report.ArchiveFact(name="a.zip", size=10, members=7)
    ])
    text = report.format_summary(report.build_report(inv, distrust_threshold=25))
    assert isinstance(text, str)
    assert "image" in text


# --- C1: only a media loose part may cover a sequence gap -----------------

def test_a_non_media_loose_file_does_not_cover_a_missing_part():
    """transfer-log-002.txt must not erase genuinely-absent part 002.

    Google delivers an oversized *media* file as its own raw part. A text file
    is never a part, and folding it into the covered set turns a real gap into
    "missing parts none".
    """
    inv = _inv(
        part_numbers={"001", "003"},
        loose_parts=[
            report.LoosePart(name="transfer-log-002.txt", size=10, part="002", kind=None)
        ],
    )
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["missing_parts"] == ["002"]


def test_an_image_loose_part_covers_a_missing_part():
    inv = _inv(
        part_numbers={"001", "003"},
        loose_parts=[report.LoosePart(name="huge-002.jpg", size=10, part="002", kind="image")],
    )
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["missing_parts"] == []


# --- I1b: an unparsed numbering scheme is not "none missing" --------------

def test_no_parsable_part_number_is_reported_as_undetermined_not_as_none():
    """The refuse-to-guess rule: gap detection that never ran must say so."""
    inv = _inv(part_numbers=set())
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["missing_parts"] is None
    assert doc["archives"]["part_numbering"] == "unrecognized"
    text = report.format_summary(doc)
    assert "not determined" in text
    assert "missing parts none" not in text


def test_an_all_non_numeric_part_number_set_is_also_undetermined():
    inv = _inv(part_numbers={"unknown", "alpha"})
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["missing_parts"] is None
    assert doc["archives"]["part_numbering"] == "unrecognized"


def test_a_recognized_numbering_scheme_with_no_gap_says_so():
    inv = _inv(part_numbers={"001", "002", "003"})
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["missing_parts"] == []
    assert doc["archives"]["part_numbering"] == "recognized"
    assert "none" in report.format_summary(doc)


# --- I2: the projection names the pipeline it describes -------------------

def test_projection_separates_what_ingest_does_today_from_deferred_work():
    """Video is deferred and misnamed media go to provenance -- today.

    organized_today must count recognized images only; the deferred-fixes
    figure adds recognized video and sniffed-media on top.
    """
    inv = _inv(
        kind_counts={"image": 10, "video": 4, "other": 7, "metadata": 3},
        kind_bytes={"image": 100, "video": 400, "other": 70, "metadata": 3},
        misnamed_kind_counts={"image": 5, "video": 1},
        misnamed_kind_bytes={"image": 50, "video": 20},
    )
    proj = report.build_report(inv, distrust_threshold=25)["projection"]
    assert proj["organized_today"] == 10
    assert proj["organized_after_deferred_fixes"] == 10 + 4 + 5 + 1
    assert proj["bytes_today"] == 100
    assert proj["bytes_after_deferred_fixes"] == 100 + 400 + 50 + 20
    assert proj["archive_total_bytes"] == 573
    assert "defer" in proj["note"].lower()
    assert "provenance" in proj["note"].lower()


def test_projection_bytes_today_is_not_the_archive_total():
    """The 345-GiB-vs-100-GiB failure: those two must not be one number."""
    inv = _inv(
        kind_counts={"image": 1, "metadata": 1},
        kind_bytes={"image": 10, "metadata": 90},
    )
    proj = report.build_report(inv, distrust_threshold=25)["projection"]
    assert proj["bytes_today"] == 10
    assert proj["archive_total_bytes"] == 100


def test_format_summary_labels_both_projection_figures():
    inv = _inv(
        kind_counts={"image": 10, "video": 4},
        kind_bytes={"image": 100, "video": 400},
        misnamed_kind_counts={"image": 5},
        misnamed_kind_bytes={"image": 50},
    )
    text = report.format_summary(report.build_report(inv, distrust_threshold=25))
    assert "today" in text
    assert "organized estimate " not in text  # the old ambiguous label is gone


# --- M2: the duplicate upper bound is the tight one -----------------------

def test_duplicate_upper_bound_is_members_minus_distinct_names():
    """Three copies of one name are two duplicates, not three."""
    inv = _inv(basename_collisions=1, basename_collision_members=3)
    dupes = report.build_report(inv, distrust_threshold=25)["projection"]["duplicates"]
    assert dupes["name_collision_upper_bound"] == 2
    assert dupes["distinct_colliding_names"] == 1


# --- M3: an unreadable archive degrades the status ------------------------

def test_an_unreadable_archive_degrades_the_archive_status():
    inv = _inv(
        archives=[report.ArchiveFact(name="a.zip", size=1, members=0, error="boom")],
        unreadable_archives=1,
    )
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["status"] == "degraded"
    assert "degraded" in report.format_summary(doc)


def test_a_clean_archive_set_is_ok_and_the_status_is_printed():
    inv = _inv(archives=[report.ArchiveFact(name="a.zip", size=1, members=2)])
    doc = report.build_report(inv, distrust_threshold=25)
    assert doc["archives"]["status"] == "ok"
    assert "ok" in report.format_summary(doc)


# --- T6: the descriptor tally says what it left out -----------------------

def test_descriptor_block_records_the_media_it_could_not_tally():
    r"""The 3,963 .screen files carry 19-digit machine names and are excluded.

    is_camera_generated("5427880241588018962") is False (the bare-digits
    pattern is ^\d{9,13}$), so those are the files that would be MOST wrongly
    pinned at tier 30 -- the tally must not be silent about omitting them.
    """
    inv = _inv(
        descriptor_human=47460,
        descriptor_machine=31751,
        misnamed_counts={".screen": 3963, ".tile": 113},
    )
    desc = report.build_report(inv, distrust_threshold=25)["anomalies"]["descriptors"]
    assert desc["excluded_unrecognized_extension"] == 4076
    assert "exclud" in desc["note"].lower()
