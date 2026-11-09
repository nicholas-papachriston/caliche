#!/usr/bin/env python3
"""Validate Caliche first-party package versions are lockstep 0.1.0."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "0.1.0"
VERSION_PATTERN = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')

CARGO_MANIFESTS = [
    ROOT / "Cargo.toml",
    ROOT / "crates" / "caliche_contracts" / "Cargo.toml",
    ROOT / "crates" / "caliche_policy" / "Cargo.toml",
    ROOT / "caliche_cli" / "Cargo.toml",
    ROOT / "caliche_server" / "Cargo.toml",
    ROOT / "caliche_web" / "Cargo.toml",
    ROOT / "caliche_sandbox" / "Cargo.toml",
    ROOT / "caliche_js" / "Cargo.toml",
    ROOT / "caliche_py" / "Cargo.toml",
]

PACKAGE_JSON = [
    ROOT / "caliche_js" / "package.json",
    ROOT / "caliche_js" / "packages" / "contracts" / "package.json",
    ROOT / "caliche_js" / "packages" / "core" / "package.json",
    ROOT / "caliche_js" / "packages" / "runtime-node" / "package.json",
    ROOT / "caliche_js" / "packages" / "mcp" / "package.json",
    ROOT / "caliche_js" / "packages" / "hooks" / "package.json",
    ROOT / "caliche_js" / "packages" / "adapters-langchain" / "package.json",
    ROOT / "caliche_js" / "packages" / "adapters-ai-sdk" / "package.json",
]

PYPROJECT = [
    ROOT / "caliche_py" / "pyproject.toml",
    ROOT / "caliche_py" / "packages" / "contracts" / "pyproject.toml",
    ROOT / "caliche_py" / "packages" / "core" / "pyproject.toml",
    ROOT / "caliche_py" / "packages" / "runtime" / "pyproject.toml",
    ROOT / "caliche_py" / "packages" / "hooks" / "pyproject.toml",
    ROOT / "caliche_py" / "packages" / "mcp" / "pyproject.toml",
    ROOT / "caliche_py" / "packages" / "integrations-langchain" / "pyproject.toml",
]


def read_toml_version(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = VERSION_PATTERN.match(line.strip())
        if match:
            return match.group(1)
    raise RuntimeError(f"Could not find explicit version in {path}")


def read_package_json_version(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    version = data.get("version")
    if not isinstance(version, str):
        raise RuntimeError(f"Missing version in {path}")
    return version


def main() -> int:
    found: dict[Path, str] = {}
    for path in CARGO_MANIFESTS + PYPROJECT:
        if path.exists():
            found[path] = read_toml_version(path)
    for path in PACKAGE_JSON:
        if path.exists():
            found[path] = read_package_json_version(path)

    mismatches = {
        path: version for path, version in found.items() if version != EXPECTED
    }
    if not mismatches:
        print(f"Package version check passed: {EXPECTED} ({len(found)} manifests)")
        return 0

    print(f"Package version check failed. Expected {EXPECTED}:")
    for path, version in sorted(mismatches.items()):
        print(f"  - {path.relative_to(ROOT)} => {version}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
