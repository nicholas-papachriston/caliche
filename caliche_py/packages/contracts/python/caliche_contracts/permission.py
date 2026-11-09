"""Wire permission mode helpers."""

from __future__ import annotations

from typing import Literal

PermissionMode = Literal[
    "default",
    "accept-edits",
    "accept-all",
    "deny-all",
    "interactive",
]


def parse_permission_mode(value: str) -> PermissionMode | None:
    """Return the permission-mode spelling, or None when it is unknown."""
    normalized = value.strip().lower()
    spellings = {
        "default": "default",
        "accept-edits": "accept-edits",
        "accept-all": "accept-all",
        "deny-all": "deny-all",
        "interactive": "interactive",
    }
    return spellings.get(normalized)  # type: ignore[return-value]
