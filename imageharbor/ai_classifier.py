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

    top_parent: str              # one of the 9 fixed classes, e.g. "500"
    label: str                   # category label, e.g. "holidays" (reuse or new)
    sub_parent: str | None = None  # existing sub code to place a new leaf under
    descriptor: str = "photo"    # 1-3 words for the filename
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
    def classify(
        self, image_path: Path, exif_data: dict[str, Any], taxonomy_snapshot: str
    ) -> PhotoClassification:
        """Classify *image_path* and return a PhotoClassification.

        *taxonomy_snapshot* is a compact text view of the current taxonomy so
        the classifier can reuse an existing category label when one fits.
        """

    def adjudicate(self, label: str, candidates: list[str]) -> str | None:
        """Return the candidate the label is a synonym of, or None.

        Default: no match. Subclasses backed by a real model may override this
        to let the AI decide whether a proposed *label* is the same category as
        one of the existing *candidates*.
        """
        return None


# ---------------------------------------------------------------------------
# Stub (deterministic, no network)
# ---------------------------------------------------------------------------

# Very simple keyword scan so tests can predict the code. Module-level so it is
# built once and shared across all StubClassifier instances/calls.
_KEYWORD_MAP: list[tuple[str, int]] = [
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
    ("document|letter|page|text|scan", 710),
    ("chart|graph|diagram|table", 720),
    ("receipt|invoice|bill", 730),
    ("building|house|apartment|mansion", 810),
    ("office|store|mall|shop", 820),
    ("castle|cathedral|ruin|monument|historic", 830),
    ("food|meal|dish|pizza|burger|sushi", 910),
    ("object|thing|item|product", 920),
    ("abstract|pattern|texture|art", 930),
]


class StubClassifier(AIClassifier):
    """Deterministic stub that returns predictable output without AI calls.

    Useful for testing and pipeline dry-runs.  The category and descriptor are
    derived from the filename so results are repeatable. The matched legacy PCS
    code is translated into the new ``(top_parent, label)`` contract using the
    seed taxonomy.
    """

    MODEL_VERSION = "stub-1.0"

    def classify(
        self, image_path: Path, exif_data: dict[str, Any], taxonomy_snapshot: str = ""
    ) -> PhotoClassification:
        stem = image_path.stem.lower()
        pcs_code = 900  # default: miscellaneous

        # Whole-word matching: tokenize the stem into words (stems use "_"/"-"
        # as separators, so split on non-alphanumerics) and match each keyword
        # as an exact word. This avoids substring bleed (e.g. "cathedral" no
        # longer matches "cat").
        words = set(re.sub(r"[^a-z0-9]+", " ", stem).split())
        for pattern, code in _KEYWORD_MAP:
            if any(kw in words for kw in pattern.split("|")):
                pcs_code = code
                break

        cat = PCS_CATEGORIES.get(pcs_code) or PCS_CATEGORIES[900]
        top_parent = str((pcs_code // 100) * 100)

        # Build a simple descriptor from the first two non-trivial words of the stem
        stem_words = re.sub(r"[^a-z0-9]+", " ", stem).split()
        descriptor_words = [w for w in stem_words if len(w) > 1][:2]
        descriptor = " ".join(descriptor_words) if descriptor_words else "photo"

        return PhotoClassification(
            top_parent=top_parent,
            label=cat.name,
            descriptor=descriptor,
            caption=f"Stub classification for {image_path.name}",
            model_version=self.MODEL_VERSION,
        )


# ---------------------------------------------------------------------------
# OpenAI Vision classifier (optional; requires `pip install openai`)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a photo archivist that classifies images into an extensible taxonomy.

The taxonomy has 9 fixed top-level classes (codes 100-900). Beneath them are
sub-categories that grow over time. Here is the CURRENT taxonomy snapshot:

{taxonomy_snapshot}

Respond ONLY with a JSON object containing exactly these keys:
- top_parent (string): the code of the best fitting top-level class, e.g. "300"
- label      (string): a short category label. REUSE an existing sub-category
                       label from the snapshot when one fits; otherwise propose
                       a new concise label.
- sub_parent (string, optional): the code of an existing sub-category to nest a
                       new leaf beneath, or omit/null if not applicable.
- descriptor (string): 1-3 words, lowercase, hyphens, describing the content
- caption    (string): one sentence describing the image (max 120 chars)
- objects    (array of strings): notable objects detected in the image
- secondary_tags (array of strings): additional descriptive tags
- ocr_text   (string): any text visible in the image, or empty string

Rules:
- top_parent MUST be one of the 9 fixed class codes (100, 200, ... 900).
- descriptor uses only a-z, 0-9, and hyphens.
- Keep the JSON compact and valid.
"""

_USER_PROMPT = "Classify this image according to the taxonomy."


def _build_pcs_list() -> str:
    """Return a compact ``code: Label`` listing of the seed PCS taxonomy.

    Retained as a helper for tooling/tests; the OpenAI prompt now uses the live
    taxonomy snapshot instead.
    """
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

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        try:
            import openai as _openai  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for OpenAIClassifier. "
                "Install it with: pip install imageharbor[openai]"
            ) from exc

        self._openai = _openai
        # Local OpenAI-compatible servers (e.g. Ollama on the Jetson) usually
        # ignore the API key, but the SDK requires a non-empty value, so fall
        # back to a placeholder when none is supplied.
        self._client = _openai.OpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
            timeout=timeout,
        )
        self._model = model
        self.MODEL_VERSION = model

    def classify(
        self, image_path: Path, exif_data: dict[str, Any], taxonomy_snapshot: str = ""
    ) -> PhotoClassification:
        import base64

        with open(image_path, "rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("ascii")

        suffix = image_path.suffix.lower().lstrip(".")
        _MEDIA_TYPES = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        media_type = _MEDIA_TYPES.get(suffix, "image/jpeg")

        system_msg = _SYSTEM_PROMPT.format(taxonomy_snapshot=taxonomy_snapshot)

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

        top_parent = str(data.get("top_parent", "900"))
        label = str(data.get("label", "miscellaneous"))
        sub_parent_val = data.get("sub_parent")
        sub_parent = str(sub_parent_val) if sub_parent_val else None

        # Coerce list fields defensively: a JSON string would char-split under
        # list(), and a non-iterable (e.g. int) would raise. Only accept real
        # sequences; otherwise fall back to an empty list.
        objects_val = data.get("objects", [])
        objects = [str(x) for x in objects_val] if isinstance(objects_val, (list, tuple)) else []
        tags_val = data.get("secondary_tags", [])
        secondary_tags = [str(x) for x in tags_val] if isinstance(tags_val, (list, tuple)) else []

        return PhotoClassification(
            top_parent=top_parent,
            label=label,
            sub_parent=sub_parent,
            descriptor=str(data.get("descriptor", "photo")),
            caption=str(data.get("caption", "")),
            objects=objects,
            secondary_tags=secondary_tags,
            ocr_text=str(data.get("ocr_text", "")),
            model_version=self.MODEL_VERSION,
        )

    def adjudicate(self, label: str, candidates: list[str]) -> str | None:
        prompt = (
            f"Is '{label}' the same category as any of these: {candidates}? "
            f"Reply with the exact matching item, or NONE."
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
        )
        answer = (resp.choices[0].message.content or "").strip()
        for c in candidates:
            if c.lower() == answer.lower():
                return c
        return None
