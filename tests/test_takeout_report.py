"""Report logic is pure: an inventory in, a document out. No filesystem."""

from collections import Counter

from imageharbor.takeout import report

# Fields that are Counters on SurveyInventory. The helper below wraps plain
# dicts so a test can pass {".screen": 3963} without losing .most_common().
_COUNTER_FIELDS = {
    "ext_counts", "ext_bytes", "kind_counts", "kind_bytes", "area_counts",
    "misnamed_counts", "misnamed_bytes", "timestamp_counts", "year_counts",
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
    assert dupes["name_collision_upper_bound"] == 22402
    assert dupes["exact"] is None
    assert "upper bound" in dupes["note"].lower()


def test_an_empty_archive_set_is_labelled_empty_not_reported_as_a_clean_result():
    """Zero members and "no archives found" must not read the same."""
    doc = report.build_report(_inv(), distrust_threshold=25)
    assert doc["archives"]["status"] == "empty"
    assert doc["projection"]["organized_estimate"] == 0


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
