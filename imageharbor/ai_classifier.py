"""AI classification interface and built-in implementations.

Public surface
--------------
PhotoClassification   – dataclass holding all AI-derived metadata
AIClassifier          – abstract base class
StubClassifier        – deterministic stub (no network calls; good for tests)
OpenAIClassifier      – real classifier backed by OpenAI Vision API (optional)
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .pcs import PCS_CATEGORIES, VALID_CODES, PCS_VERSION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PhotoClassification:
    """All AI-derived metadata for a single image."""

    pcs_code: int
    descriptor: str              # 1–3 words, pre-normalisation
    caption: str = ""
    objects: list[str] = field(default_factory=list)
    secondary_tags: list[str] = field(default_factory=list)
    ocr_text: str = ""
    model_version: str = "stub-1.0"
    pcs_version: str = PCS_VERSION


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AIClassifier(ABC):
    """Abstract photo classifier."""

    @abstractmethod
    def classify(self, image_path: Path, exif_data: dict[str, Any]) -> PhotoClassification:
        """Classify *image_path* and return a PhotoClassification."""


# ---------------------------------------------------------------------------
# Stub (deterministic, no network)
# ---------------------------------------------------------------------------


class StubClassifier(AIClassifier):
    """Deterministic stub that returns predictable output without AI calls.

    Useful for testing and pipeline dry-runs.  The PCS code and descriptor
    are derived from the filename so results are repeatable.
    """

    MODEL_VERSION = "stub-1.0"

    def classify(self, image_path: Path, exif_data: dict[str, Any]) -> PhotoClassification:
        stem = image_path.stem.lower()
        pcs_code = 900  # default: miscellaneous

        # Very simple keyword scan so tests can predict the code
        keyword_map: list[tuple[str, int]] = [
            ("portrait|person|people|face|selfie", 110),
            ("group|crowd|team|family", 120),
            ("child|kid|baby", 130),
            ("wedding|birthday|party|celebration", 140),
            ("pet|dog|cat|puppy|kitten", 210),
            ("animal|wildlife|lion|tiger|bear|fox", 220),
            ("bird|eagle|hawk|robin|sparrow", 230),
            ("street|city|downtown|urban", 310),
            ("farm|countryside|village|rural", 320),
            ("beach|ocean|sea|coast|shore|dunes", 330),
            ("mountain|hill|peak|cliff|ridge", 340),
            ("room|interior|inside|indoor", 350),
            ("car|vehicle|auto|truck|suv", 410),
            ("plane|airplane|aircraft|jet|helicopter", 420),
            ("boat|ship|yacht|vessel|sail", 430),
            ("train|rail|subway|metro", 440),
            ("sport|game|match|race|field|stadium", 510),
            ("concert|music|band|stage|performer", 520),
            ("ceremony|graduation|award|formal", 530),
            ("flower|plant|garden|tree|forest|leaf", 610),
            ("landscape|valley|plains|meadow", 620),
            ("storm|rain|snow|fog|cloud|lightning", 630),
            ("sky|sunset|sunrise|stars|moon", 640),
            ("document|letter|page|text|receipt|scan", 710),
            ("chart|graph|diagram|table", 720),
            ("receipt|invoice|bill", 730),
            ("building|house|apartment|mansion", 810),
            ("office|store|mall|shop", 820),
            ("castle|cathedral|ruin|monument|historic", 830),
            ("food|meal|dish|pizza|burger|sushi", 910),
            ("object|thing|item|product", 920),
            ("abstract|pattern|texture|art", 930),
        ]

        for pattern, code in keyword_map:
            if re.search(pattern, stem):
                pcs_code = code
                break

        # Build a simple descriptor from the first two non-trivial words of the stem
        words = re.sub(r"[^a-z0-9]+", " ", stem).split()
        descriptor_words = [w for w in words if len(w) > 1][:2]
        descriptor = " ".join(descriptor_words) if descriptor_words else "photo"

        return PhotoClassification(
            pcs_code=pcs_code,
            descriptor=descriptor,
            caption=f"Stub classification for {image_path.name}",
            objects=[],
            secondary_tags=[],
            ocr_text="",
            model_version=self.MODEL_VERSION,
        )


# ---------------------------------------------------------------------------
# OpenAI Vision classifier (optional; requires `pip install openai`)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a photo archivist that classifies images using the Photo Classification Standard (PCS).

Respond ONLY with a JSON object containing exactly these keys:
- pcs_code   (integer): one of the valid PCS codes listed below
- descriptor (string):  1-3 words, lowercase, hyphens, describing the image content
- caption    (string):  one sentence describing the image (max 120 chars)
- objects    (array of strings): notable objects detected in the image
- secondary_tags (array of strings): additional descriptive tags
- ocr_text   (string): any text visible in the image, or empty string

Valid PCS codes and their meanings:
{pcs_list}

Rules:
- pcs_code MUST be one of the listed codes.
- descriptor uses only a-z, 0-9, and hyphens.
- Keep the JSON compact and valid.
"""

_USER_PROMPT = "Classify this image according to the PCS taxonomy."


def _build_pcs_list() -> str:
    lines = []
    for code in VALID_CODES:
        cat = PCS_CATEGORIES[code]
        indent = "  " if cat.parent is not None else ""
        lines.append(f"{indent}{code}: {cat.label}")
    return "\n".join(lines)


class OpenAIClassifier(AIClassifier):
    """Classifier backed by the OpenAI Vision API.

    Requires the ``openai`` package: ``pip install imageharbor[openai]``
    """

    MODEL_VERSION = "gpt-4o-mini"

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        try:
            import openai as _openai  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for OpenAIClassifier. "
                "Install it with: pip install imageharbor[openai]"
            ) from exc

        self._openai = _openai
        self._client = _openai.OpenAI(api_key=api_key)
        self._model = model
        self.MODEL_VERSION = model

    def classify(self, image_path: Path, exif_data: dict[str, Any]) -> PhotoClassification:
        import base64

        with open(image_path, "rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("ascii")

        suffix = image_path.suffix.lower().lstrip(".")
        media_type = f"image/{suffix}" if suffix in ("jpg", "jpeg", "png", "gif", "webp") else "image/jpeg"

        system_msg = _SYSTEM_PROMPT.format(pcs_list=_build_pcs_list())

        response = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
                                "detail": "low",
                            },
                        },
                        {"type": "text", "text": _USER_PROMPT},
                    ],
                },
            ],
            max_tokens=512,
        )

        raw = response.choices[0].message.content or "{}"
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("OpenAI returned invalid JSON: %s", raw)
            data = {}

        pcs_code = int(data.get("pcs_code", 900))
        if pcs_code not in PCS_CATEGORIES:
            logger.warning("OpenAI returned unknown PCS code %s; using 900", pcs_code)
            pcs_code = 900

        return PhotoClassification(
            pcs_code=pcs_code,
            descriptor=str(data.get("descriptor", "photo")),
            caption=str(data.get("caption", "")),
            objects=list(data.get("objects", [])),
            secondary_tags=list(data.get("secondary_tags", [])),
            ocr_text=str(data.get("ocr_text", "")),
            model_version=self.MODEL_VERSION,
        )
