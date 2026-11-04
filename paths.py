from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONVERT_ROOT = REPO_ROOT / "convert"


def ensure_sys_path() -> Path:
    """Put this folder and ``convert/`` on ``sys.path`` so ``sirc`` stays importable."""
    for root in (REPO_ROOT, CONVERT_ROOT):
        text = str(root)
        if text not in sys.path:
            sys.path.insert(0, text)
    return REPO_ROOT
