from __future__ import annotations

import os
import sys


def _prepend(path: str) -> None:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_prepend(os.path.join(_REPO_ROOT, "third_party"))
_prepend(os.path.join(_REPO_ROOT, "berkeley-function-call-leaderboard"))
