#!/usr/bin/env python3
"""Validate key public exports from the root `caliche` facade.

Requires each token to appear in a public item declaration in `src/lib.rs`
(`pub use` / `pub mod` / `pub struct` / …), including multi-line `pub use`
blocks. Module doc comments are ignored.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB_RS = ROOT / "src" / "lib.rs"

REQUIRED_EXPORT_TOKENS = [
    "AgentBuilder",
    "AgentService",
    "AgentSession",
    "ProviderType",
    "Tool",
    "ToolRegistry",
    "HookMatcher",
    "PermissionMode",
]

# Optional / feature-gated MCP types: require when mentioned on a pub line.
OPTIONAL_EXPORT_TOKENS = [
    "McpServer",
    "InProcessMcpServer",
]

PUB_ITEM = re.compile(
    r"^\s*pub(?:\s*\([^)]*\))?\s+(?:use|mod|struct|enum|trait|type|fn|const)\b"
)


def public_code_text(source: str) -> str:
    """Return non-doc source lines that are part of public items."""
    out: list[str] = []
    in_block_doc = False
    in_pub_use = False
    brace_depth = 0

    for raw in source.splitlines():
        stripped = raw.strip()
        if stripped.startswith("/*!") or stripped.startswith("/**"):
            in_block_doc = True
        if in_block_doc:
            if "*/" in stripped:
                in_block_doc = False
            continue
        if stripped.startswith("//"):
            continue

        if PUB_ITEM.match(raw):
            out.append(raw)
            if re.search(r"\bpub(?:\s*\([^)]*\))?\s+use\b", raw) and "{" in raw and "}" not in raw:
                in_pub_use = True
                brace_depth = raw.count("{") - raw.count("}")
            continue

        if in_pub_use:
            out.append(raw)
            brace_depth += raw.count("{") - raw.count("}")
            if brace_depth <= 0:
                in_pub_use = False
                brace_depth = 0

    return "\n".join(out)


def main() -> int:
    source = LIB_RS.read_text(encoding="utf-8")
    haystack = public_code_text(source)
    missing = [token for token in REQUIRED_EXPORT_TOKENS if token not in haystack]
    # Optional MCP exports: only fail if neither root nor prelude exposes them
    # and the tokens appear nowhere as pub items (feature may be off in docs).
    for token in OPTIONAL_EXPORT_TOKENS:
        if token not in haystack and token not in source:
            missing.append(token)

    if not missing:
        print("Facade export check passed.")
        return 0

    print("Facade export check failed. Missing token(s) on pub item lines:")
    for token in missing:
        print(f"  - {token}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
