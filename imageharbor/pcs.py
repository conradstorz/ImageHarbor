"""Photo Classification Standard (PCS) taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PCSCategory:
    """A single node in the PCS taxonomy."""

    code: int
    name: str
    label: str
    parent: Optional[int] = None


# ---------------------------------------------------------------------------
# PCS taxonomy (version 1)
# ---------------------------------------------------------------------------
# Ranges:
#   100 – People
#   200 – Animals
#   300 – Places
#   400 – Transportation
#   500 – Events
#   600 – Nature
#   700 – Documents
#   800 – Architecture
#   900 – Miscellaneous

PCS_VERSION = "1"

PCS_CATEGORIES: dict[int, PCSCategory] = {
    # --- 100 People ---
    100: PCSCategory(100, "people", "People"),
    110: PCSCategory(110, "portraits", "Portraits", parent=100),
    120: PCSCategory(120, "groups", "Groups", parent=100),
    130: PCSCategory(130, "children", "Children", parent=100),
    140: PCSCategory(140, "celebrations", "Celebrations", parent=100),
    # --- 200 Animals ---
    200: PCSCategory(200, "animals", "Animals"),
    210: PCSCategory(210, "pets", "Pets", parent=200),
    220: PCSCategory(220, "wildlife", "Wildlife", parent=200),
    230: PCSCategory(230, "birds", "Birds", parent=200),
    # --- 300 Places ---
    300: PCSCategory(300, "places", "Places"),
    310: PCSCategory(310, "urban", "Urban", parent=300),
    320: PCSCategory(320, "rural", "Rural", parent=300),
    330: PCSCategory(330, "beach", "Beach", parent=300),
    340: PCSCategory(340, "mountains", "Mountains", parent=300),
    350: PCSCategory(350, "indoor", "Indoor", parent=300),
    # --- 400 Transportation ---
    400: PCSCategory(400, "transportation", "Transportation"),
    410: PCSCategory(410, "cars", "Cars", parent=400),
    420: PCSCategory(420, "aircraft", "Aircraft", parent=400),
    430: PCSCategory(430, "boats", "Boats", parent=400),
    440: PCSCategory(440, "trains", "Trains", parent=400),
    # --- 500 Events ---
    500: PCSCategory(500, "events", "Events"),
    510: PCSCategory(510, "sports", "Sports", parent=500),
    520: PCSCategory(520, "concerts", "Concerts", parent=500),
    530: PCSCategory(530, "ceremonies", "Ceremonies", parent=500),
    # --- 600 Nature ---
    600: PCSCategory(600, "nature", "Nature"),
    610: PCSCategory(610, "plants", "Plants & Flowers", parent=600),
    620: PCSCategory(620, "landscapes", "Landscapes", parent=600),
    630: PCSCategory(630, "weather", "Weather", parent=600),
    640: PCSCategory(640, "sky", "Sky", parent=600),
    # --- 700 Documents ---
    700: PCSCategory(700, "documents", "Documents"),
    710: PCSCategory(710, "text", "Text", parent=700),
    720: PCSCategory(720, "charts", "Charts & Graphs", parent=700),
    730: PCSCategory(730, "receipts", "Receipts", parent=700),
    # --- 800 Architecture ---
    800: PCSCategory(800, "architecture", "Architecture"),
    810: PCSCategory(810, "residential", "Residential", parent=800),
    820: PCSCategory(820, "commercial", "Commercial", parent=800),
    830: PCSCategory(830, "historic", "Historic", parent=800),
    # --- 900 Miscellaneous ---
    900: PCSCategory(900, "miscellaneous", "Miscellaneous"),
    910: PCSCategory(910, "food", "Food", parent=900),
    920: PCSCategory(920, "objects", "Objects", parent=900),
    930: PCSCategory(930, "abstract", "Abstract", parent=900),
}

# Ordered list of valid codes for AI prompt construction
VALID_CODES: list[int] = sorted(PCS_CATEGORIES.keys())


def get_category(code: int) -> Optional[PCSCategory]:
    """Return the PCSCategory for *code*, or None if unknown."""
    return PCS_CATEGORIES.get(code)


def resolve_code(code: int) -> int:
    """Return *code* if it is a known leaf/branch; fall back to 900."""
    return code if code in PCS_CATEGORIES else 900
