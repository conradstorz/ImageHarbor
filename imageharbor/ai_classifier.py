"""AI perception interface and built-in implementations.

Public surface
--------------
ContentDescription    – dataclass holding pure perception output (no taxonomy)
AIClassifier          – abstract base class
StubClassifier        – deterministic stub (no network calls; good for tests)
OpenAIClassifier      – real backend backed by OpenAI Vision API (optional)

Design note
-----------
The AI only *describes* an image (``describe`` -> :class:`ContentDescription`).
Deciding which class the content belongs to is the organizer's job (see
``concept_map`` and the pipeline); when the concept-map misses, the AI is asked
to ``pick_class`` from the fixed top-level classes. Nothing here knows the
taxonomy shape.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ContentDescription:
    """Pure perception output — what is in the photo, no taxonomy knowledge."""

    primary_subject: str = "photo"  # 1-3 words, the main subject
    scene: str = ""  # setting/context
    objects: list[str] = field(default_factory=list)
    caption: str = ""
    tags: list[str] = field(default_factory=list)
    ocr_text: str = ""
    model_version: str = "stub-1.0"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AIClassifier(ABC):
    """Abstract AI backend: describes images and (on demand) picks a class."""

    @abstractmethod
    def describe(self, image_path: Path, exif_data: dict[str, Any]) -> ContentDescription:
        """Describe *image_path* and return a :class:`ContentDescription`.

        Pure perception: report what is visible, do NOT categorize.
        """

    def adjudicate(self, label: str, candidates: list[str]) -> str | None:
        """Return the candidate the label is a synonym of, or None.

        Default: no match. Subclasses backed by a real model may override this
        to let the AI decide whether a proposed *label* is the same category as
        one of the existing *candidates*.
        """
        return None

    def pick_class(self, content: "ContentDescription", classes: list[tuple[str, str]]) -> str:
        """Pick the best class CODE from `classes`. Default: miscellaneous."""
        return "900"


# ---------------------------------------------------------------------------
# Stub (deterministic, no network)
# ---------------------------------------------------------------------------


class StubClassifier(AIClassifier):
    """Deterministic stub that returns predictable output without AI calls.

    Useful for testing and pipeline dry-runs. The primary subject and tags are
    derived from the filename so results are repeatable. ``pick_class`` inherits
    the ABC default (900) so a miss stays deterministic and network-free.
    """

    MODEL_VERSION = "stub-1.0"

    def describe(self, image_path: Path, exif_data: dict[str, Any]) -> ContentDescription:
        stem = image_path.stem.lower()
        words = [w for w in re.sub(r"[^a-z0-9]+", " ", stem).split() if len(w) > 1]
        primary = words[0] if words else "photo"
        return ContentDescription(
            primary_subject=primary,
            caption=f"Stub description for {image_path.name}",
            tags=words[:3],
            model_version=self.MODEL_VERSION,
        )
    # pick_class inherits the ABC default (900) — deterministic, no network.


# ---------------------------------------------------------------------------
# OpenAI Vision backend (optional; requires `pip install openai`)
# ---------------------------------------------------------------------------

_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


class OpenAIClassifier(AIClassifier):
    """AI backend backed by the OpenAI Vision API.

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

    def describe(self, image_path: Path, exif_data: dict[str, Any]) -> ContentDescription:
        import base64

        with open(image_path, "rb") as fh:
            image_b64 = base64.b64encode(fh.read()).decode("ascii")
        suffix = image_path.suffix.lower().lstrip(".")
        media_type = _MEDIA_TYPES.get(suffix, "image/jpeg")
        system = (
            "You are a photo describer. Look at the image and respond ONLY with a "
            "JSON object with keys: primary_subject (1-3 words), scene (short), "
            "objects (array), caption (one sentence), tags (array), ocr_text (string). "
            "Do NOT categorize or classify — only describe what you see."
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
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
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
            ],
            max_tokens=400,
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("OpenAI describe returned invalid JSON: %s", raw)
            data = {}

        def _list(v: Any) -> list[str]:
            return [str(x) for x in v] if isinstance(v, (list, tuple)) else []

        return ContentDescription(
            primary_subject=str(data.get("primary_subject") or "photo"),
            scene=str(data.get("scene", "")),
            objects=_list(data.get("objects", [])),
            caption=str(data.get("caption", "")),
            tags=_list(data.get("tags", [])),
            ocr_text=str(data.get("ocr_text", "")),
            model_version=self.MODEL_VERSION,
        )

    def pick_class(self, content: ContentDescription, classes: list[tuple[str, str]]) -> str:
        options = "\n".join(f"{code}: {label}" for code, label in classes)
        prompt = (
            "Pick the single best top-level class for this photo description.\n"
            f"subject: {content.primary_subject}\nscene: {content.scene}\n"
            f"objects: {content.objects}\ncaption: {content.caption}\n"
            f"classes:\n{options}\n"
            "Reply with ONLY the class code (e.g. 500)."
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
        )
        ans = (resp.choices[0].message.content or "").strip()
        valid = {code for code, _ in classes}
        for tok in re.findall(r"\d+", ans):
            if tok in valid:
                return tok
        return "900"
