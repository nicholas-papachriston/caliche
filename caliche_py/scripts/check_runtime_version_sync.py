#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import tomllib


ROOT = pathlib.Path(__file__).resolve().parents[1]
CARGO_TOML = ROOT / "Cargo.toml"
RUNTIME_PYPROJECT = ROOT / "packages" / "runtime" / "pyproject.toml"


def main() -> int:
    cargo = tomllib.loads(CARGO_TOML.read_text(encoding="utf-8"))
    runtime = tomllib.loads(RUNTIME_PYPROJECT.read_text(encoding="utf-8"))
    cargo_version = cargo["package"]["version"]
    runtime_version = runtime["project"]["version"]
    if cargo_version != runtime_version:
        print(
            f"Runtime version mismatch: Cargo.toml={cargo_version}, "
            f"packages/runtime={runtime_version}"
        )
        return 1
    print(f"Runtime version check passed: {cargo_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

