"""Resolve KINQHI_HOME for standalone skill scripts.

Skill scripts may run outside the Kinqhi process (e.g. system Python,
nix env, CI) where ``kinqhi_constants`` is not importable.  This module
provides the same ``get_kinqhi_home()`` and ``display_kinqhi_home()``
contracts as ``kinqhi_constants`` without requiring it on ``sys.path``.

When ``kinqhi_constants`` IS available it is used directly so that any
future enhancements (profile resolution, Docker detection, etc.) are
picked up automatically.  The fallback path replicates the core logic
from ``kinqhi_constants.py`` using only the stdlib.

All scripts under ``google-workspace/scripts/`` should import from here
instead of duplicating the ``KINQHI_HOME = Path(os.getenv(...))`` pattern.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from kinqhi_constants import display_kinqhi_home as display_kinqhi_home
    from kinqhi_constants import get_kinqhi_home as get_kinqhi_home
except (ModuleNotFoundError, ImportError):

    def get_kinqhi_home() -> Path:
        """Return the Kinqhi home directory (default: ~/.kinqhi).

        Mirrors ``kinqhi_constants.get_kinqhi_home()``."""
        val = os.environ.get("KINQHI_HOME", "").strip()
        return Path(val) if val else Path.home() / ".kinqhi"

    def display_kinqhi_home() -> str:
        """Return a user-friendly ``~/``-shortened display string.

        Mirrors ``kinqhi_constants.display_kinqhi_home()``."""
        home = get_kinqhi_home()
        try:
            return "~/" + str(home.relative_to(Path.home()))
        except ValueError:
            return str(home)
