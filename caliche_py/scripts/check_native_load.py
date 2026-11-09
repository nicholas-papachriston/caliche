#!/usr/bin/env python3
"""Fail when the native runtime module does not load.

A release build can link a module that the dynamic loader then rejects, so the
release scripts import it right after building.

    check_native_load.py            import the module installed in this interpreter
    check_native_load.py WHEEL...   import the module from each built wheel
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import zipfile

MODULE = "caliche_runtime._caliche_py"


def import_module(extra_path: str | None) -> subprocess.CompletedProcess[str]:
    prelude = f"import sys; sys.path.insert(0, {extra_path!r}); " if extra_path else ""
    return subprocess.run(
        [sys.executable, "-I", "-c", f"{prelude}import {MODULE}"],
        capture_output=True,
        text=True,
        check=False,
    )


def report(label: str, result: subprocess.CompletedProcess[str]) -> bool:
    if result.returncode == 0:
        print(f"Native module loads: {label}")
        return True
    print(f"Native module does not load: {label}", file=sys.stderr)
    print(result.stderr.strip(), file=sys.stderr)
    return False


def main(argv: list[str]) -> int:
    if not argv:
        return 0 if report(f"{MODULE} (installed)", import_module(None)) else 1
    ok = True
    for wheel in argv:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(tmp)
            ok = report(pathlib.Path(wheel).name, import_module(tmp)) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
