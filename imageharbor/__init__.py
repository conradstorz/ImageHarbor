"""ImageHarbor: Classify. Verify. Preserve.

Copyright (C) 2026  Conrad Storz

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("imageharbor")
except PackageNotFoundError:  # source tree with no installed dist
    # Deliberately not PEP 440 and deliberately not digit-leading: a digit-leading
    # literal here would trip test_no_hardcoded_version_literals_remain, and this
    # branch is unreachable under the documented `uv run` workflow (the dist is
    # always installed). Do not "fix" this to look like a real version.
    __version__ = "unknown+uninstalled"

# This import is kept below the licence header + __version__ block so the two
# stay together at the top of the file (E402 -- module level import not at
# top of file -- is selected via "E4" in pyproject.toml's ruff config).
from .api import (  # noqa: E402
    AI,
    COPIED,
    DUPLICATE,
    ENRICHED,
    ERROR,
    FAILED,
    IO,
    OK,
    RENAMED,
    SKIPPED,
    TOTAL,
    Aborted,
    AIConfig,
    ConfigError,
    EnrichFailureRow,
    EnrichReport,
    ImageHarborError,
    ProcessReport,
    ProcessRow,
    VerifyReport,
    VerifyRow,
    enrich,  # NOTE: `enrich` here is the facade FUNCTION (api.enrich), which
    # shadows the `imageharbor.enrich` submodule as a package attribute.
    # Reach the module with `from .enrich import ...` (or
    # `importlib.import_module("imageharbor.enrich")`), never via
    # `from . import enrich` -- that now binds the function, not the module.
    process,
    stats,
    verify,
)

__all__ = [
    "process", "enrich", "verify", "stats",
    "AIConfig",
    "ProcessReport", "EnrichReport", "VerifyReport",
    "ProcessRow", "EnrichFailureRow", "VerifyRow",
    "COPIED", "DUPLICATE", "SKIPPED", "ERROR", "ENRICHED", "RENAMED", "TOTAL",
    "AI", "IO", "OK", "FAILED",
    "ImageHarborError", "ConfigError", "Aborted",
    "__version__",
]
