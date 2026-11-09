"""A Rust panic must reach Python as PanicException and not end the process."""

from __future__ import annotations

import subprocess
import sys

import pytest

import caliche

# Runs in a child process: if the extension aborts on panic, only the child dies.
PROBE = """
from caliche_runtime import _caliche_py
try:
    _caliche_py._panic_probe()
except BaseException as exc:
    print(type(exc).__name__, exc)
"""


def test_rust_panic_raises_panic_exception() -> None:
    runtime = caliche._runtime  # noqa: SLF001 - intentional feature probe
    if not hasattr(runtime, "_panic_probe"):
        pytest.skip("native extension built without panic-probe")
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "PanicException caliche panic probe"
