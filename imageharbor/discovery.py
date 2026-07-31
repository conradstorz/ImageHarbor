"""Image file discovery."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Supported image extensions (lowercase)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".heic",
        ".heif",
        ".avif",
        ".raw",
        ".cr2",
        ".cr3",
        ".nef",
        ".arw",
        ".dng",
    }
)


def discover_images(source: Path, recursive: bool = True) -> Iterator[Path]:
    """Yield all supported image files under *source*.

    Only regular files with supported extensions are yielded; the source
    directory itself is never modified.
    """
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")
    if not source.is_dir():
        # Single-file mode
        if source.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield source
        else:
            logger.warning("Skipping unsupported file: %s", source)
        return

    pattern = "**/*" if recursive else "*"
    for candidate in sorted(source.glob(pattern)):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
            logger.debug("Discovered: %s", candidate)
            yield candidate
        else:
            logger.debug("Skipping (unsupported extension): %s", candidate)
