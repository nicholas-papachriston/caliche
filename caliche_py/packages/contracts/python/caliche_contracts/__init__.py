"""
Pure wire contracts and parsers for Caliche.

Mirrors `caliche_contracts` (Rust). No native extension dependency.
"""

from __future__ import annotations

from caliche_contracts.context_ref import ContextRefKind, parse_context_ref_kind
from caliche_contracts.permission import PermissionMode, parse_permission_mode
from caliche_contracts.slash_intent import SlashIntent, parse_slash_intent

__all__ = [
    "ContextRefKind",
    "PermissionMode",
    "SlashIntent",
    "parse_context_ref_kind",
    "parse_permission_mode",
    "parse_slash_intent",
]
