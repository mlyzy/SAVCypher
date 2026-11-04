from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from paths import ensure_sys_path

ensure_sys_path()

from sirc.parser import SIRCSyntaxError, parse_sirc

SIRC_CLAUSES = (
    "ANSWER",
    "GRAPH",
    "FILTER",
    "LET",
    "KEEP",
    "AGG",
    "HAVING",
    "DEDUP BY",
    "EXPAND",
    "OPTIONAL",
    "ANY",
    "SCOPE",
    "ORDER BY",
    "TOP",
    "TAKE",
    "LIMIT",
    "SKIP",
    "OFFSET",
)


def check_sirc_syntax(text: str) -> dict[str, Any]:
    """Return whether ``text`` is syntactically valid canonical SIRC.

    This is the parser from ``sirc.parser``: the query must start with
    ``ANSWER``, then only recognized SIRC clauses. Schema/scope checks are
    intentionally not applied here because inference schemas are compact JSON,
    not the validator's ``Relationships:`` text format.
    """
    stripped = str(text or "").strip()
    if not stripped:
        return {
            "valid": False,
            "errors": ["SIRCSyntaxError: empty SIRC"],
        }
    try:
        parse_sirc(stripped)
    except (SIRCSyntaxError, ValueError) as exc:
        return {
            "valid": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    return {"valid": True, "errors": []}


def retry_feedback(invalid_sirc: str, errors: list[str]) -> str:
    """Build a correction hint appended to the task-0 prompt."""
    error_lines = "\n".join(f"- {item}" for item in errors) or "- unknown syntax error"
    clauses = ", ".join(SIRC_CLAUSES)
    return (
        "The previous SIRC was syntactically invalid and must be rewritten.\n\n"
        f"Invalid SIRC:\n{invalid_sirc.strip()}\n\n"
        f"Syntax errors:\n{error_lines}\n\n"
        "Return canonical SIRC text only. The first line must be ANSWER. "
        f"Allowed clauses: {clauses}."
    )
