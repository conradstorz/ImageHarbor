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
