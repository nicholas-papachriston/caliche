#!/usr/bin/env python3
"""Validate lockstep versions for first-party Rust crates."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')

MANIFESTS = [
    ROOT / "Cargo.toml",
    ROOT / "crates" / "caliche_contracts" / "Cargo.toml",
    ROOT / "crates" / "caliche_orchestration" / "Cargo.toml",
    ROOT / "crates" / "caliche_policy" / "Cargo.toml",
    ROOT / "caliche_cli" / "Cargo.toml",
    ROOT / "caliche_server" / "Cargo.toml",
    ROOT / "caliche_web" / "Cargo.toml",
    ROOT / "caliche_sandbox" / "Cargo.toml",
    ROOT / "caliche_js" / "Cargo.toml",
    ROOT / "caliche_py" / "Cargo.toml",
]


def read_version(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = VERSION_PATTERN.match(line.strip())
        if match:
            return match.group(1)
    raise RuntimeError(f"Could not find explicit version in {path}")


def main() -> int:
    versions = {manifest: read_version(manifest) for manifest in MANIFESTS}
    unique_versions = sorted(set(versions.values()))
    if len(unique_versions) == 1:
        print(f"Lockstep version check passed: {unique_versions[0]}")
        return 0

    print("Lockstep version check failed. Found mismatched versions:")
    for manifest, version in versions.items():
        print(f"  - {manifest.relative_to(ROOT)} => {version}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
