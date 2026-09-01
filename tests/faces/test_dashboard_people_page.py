"""The People section must be present in the served page."""

from pathlib import Path

PAGE = Path("imageharbor/dashboard/index.html").read_text(encoding="utf-8")


def test_page_has_a_people_section():
    assert 'id="people"' in PAGE


def test_page_fetches_the_people_api():
    assert "/api/people" in PAGE


def test_page_renders_face_crops():
    assert "/api/face-crop/" in PAGE


def test_page_offers_every_review_action():
    for action in ("confirm", "reject", "merge", "split"):
        assert f"/api/people/{action}" in PAGE


def test_page_shows_the_payoff_number():
    # The count of untagged photos a confirmation would name is the whole point
    # of the feature and must be visible before the operator clicks.
    assert "untagged_photos" in PAGE


def test_page_reports_hidden_singletons():
    assert "singletons_hidden" in PAGE
