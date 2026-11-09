#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import tomllib


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_FILES = [
    ROOT / "packages" / "contracts" / "pyproject.toml",
    ROOT / "packages" / "runtime" / "pyproject.toml",
    ROOT / "packages" / "core" / "pyproject.toml",
    ROOT / "packages" / "mcp" / "pyproject.toml",
    ROOT / "packages" / "hooks" / "pyproject.toml",
    ROOT / "packages" / "integrations-langchain" / "pyproject.toml",
]


def read_version(path: pathlib.Path) -> tuple[str, str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    version = data["project"]["version"]
    return path.parent.name, version


def main() -> int:
    pairs = [read_version(path) for path in PACKAGE_FILES]
    versions = {version for _, version in pairs}
    if len(versions) != 1:
        for name, version in pairs:
            print(f"{name}: {version}")
        print("Lockstep version mismatch detected.")
        return 1

    shared_version = next(iter(versions))
    print(f"Lockstep version check passed: {shared_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

