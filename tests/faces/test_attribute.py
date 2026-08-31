"""Proposal scoring. Pure, table-driven."""

import pytest

from imageharbor.faces import attribute


def test_unanimous_cluster_proposes_that_name():
    props = attribute.propose(
        {1: ["a", "b", "c"]},
        {"a": ["Emma"], "b": ["Emma"], "c": ["Emma"]},
        min_score=0.6,
        min_support=2,
    )
    assert len(props) == 1
    assert props[0].name == "Emma"
    assert props[0].support == 3
    assert props[0].total_tagged == 3
    assert props[0].score == pytest.approx(1.0)
    assert props[0].untagged_photos == 0


def test_untagged_photos_are_counted_as_the_payoff():
    props = attribute.propose(
        {1: ["a", "b"] + [f"u{i}" for i in range(340)]},
        {"a": ["Emma"], "b": ["Emma"]},
        min_score=0.6,
        min_support=2,
    )
    assert props[0].total_tagged == 2
    assert props[0].untagged_photos == 340


def test_a_cluster_with_no_tagged_photos_proposes_nothing():
    assert attribute.propose({1: ["a", "b"]}, {}, min_score=0.6, min_support=1) == []


def test_below_min_score_proposes_nothing():
    props = attribute.propose(
        {1: ["a", "b", "c"]},
        {"a": ["Emma"], "b": ["Judy"], "c": ["Pete"]},
        min_score=0.6,
        min_support=1,
    )
    assert props == []


def test_below_min_support_proposes_nothing():
    props = attribute.propose(
        {1: ["a"]}, {"a": ["Emma"]}, min_score=0.6, min_support=2
    )
    assert props == []


def test_two_always_co_occurring_names_both_qualify():
    # A couple photographed together always. Neither is more supported than the
    # other, so both are offered and the human picks. Never an arbitrary winner.
    props = attribute.propose(
        {1: ["a", "b"]},
        {"a": ["Judy", "Pete"], "b": ["Judy", "Pete"]},
        min_score=0.6,
        min_support=2,
    )
    assert sorted(p.name for p in props) == ["Judy", "Pete"]


def test_names_are_normalized_before_counting():
    props = attribute.propose(
        {1: ["a", "b"]},
        {"a": ["Emma "], "b": ["Emma"]},
        min_score=0.6,
        min_support=2,
    )
    assert len(props) == 1
    assert props[0].name == "Emma"
    assert props[0].support == 2


def test_case_variants_are_not_merged():
    # "pete storz" and "Pete Storz" stay separate; merging is a human call.
    props = attribute.propose(
        {1: ["a", "b"]},
        {"a": ["pete storz"], "b": ["Pete Storz"]},
        min_score=0.9,
        min_support=2,
    )
    assert props == []


def test_a_repeated_name_on_one_photo_counts_once():
    props = attribute.propose(
        {1: ["a", "b"]},
        {"a": ["Emma", "Emma"], "b": ["Emma"]},
        min_score=0.6,
        min_support=2,
    )
    assert props[0].support == 2


def test_output_is_sorted_by_score_then_name():
    props = attribute.propose(
        {1: ["a", "b", "c", "d"]},
        {"a": ["Emma"], "b": ["Emma"], "c": ["Emma"], "d": ["Judy"]},
        min_score=0.2,
        min_support=1,
    )
    assert [p.name for p in props] == ["Emma", "Judy"]
