#!/usr/bin/env python3
"""
Deterministic REPL smoke test for terminal rendering integrity.

Runs the CLI in a pseudo-terminal, sends local-only commands (`/help`, `/exit`),
captures transcript, and fails if basic REPL invariants are broken.
"""

from __future__ import annotations

import argparse
import os
import pty
import re
import select
import signal
import sys
import tempfile
import time
from dataclasses import dataclass


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
PROMPT_RE = re.compile(r"\] >")
CURSOR_QUERY_RE = re.compile(rb"\x1b\[6n")
CRATE_FRAGMENT_RE = re.compile(r"\b[a-z][a-z0-9_]* v\d+\.\d")
BUILD_NOISE_TOKENS = (
    "Compiling ",
    "Finished `dev` profile",
    "Building Rust project",
    "Building CLI crate",
    "Building API crate",
)


@dataclass
class SmokeResult:
    ok: bool
    reason: str
    transcript: str


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "")


def _line_snippet(line: str, limit: int = 180) -> str:
    line = line.strip()
    return line if len(line) <= limit else line[:limit] + "..."


def _validate_repl_integrity(clean: str) -> str | None:
    lines = clean.splitlines()
    if not lines:
        return "empty transcript"

    banner_idx = next((i for i, line in enumerate(lines) if line.strip() == "Caliche"), -1)
    if banner_idx < 0:
        return "startup banner missing"

    # After REPL startup, build output should never appear again.
    for line in lines[banner_idx + 1 :]:
        if any(token in line for token in BUILD_NOISE_TOKENS):
            return f"build output leaked after REPL start: {_line_snippet(line)}"

    # Startup lines should not carry concatenated crate fragments.
    startup_window = lines[max(0, banner_idx - 4) : min(len(lines), banner_idx + 12)]
    for line in startup_window:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "Caliche":
            continue
        if stripped.startswith(
            (
                "Sandbox enabled:",
                "Loaded project context from",
                "Logs written to:",
                "Provider:",
                "Tools:",
                "Session:",
                "Startup:",
                "Type /help",
            )
        ):
            if any(token in stripped for token in BUILD_NOISE_TOKENS):
                return f"startup line contains build marker: {_line_snippet(stripped)}"
            if CRATE_FRAGMENT_RE.search(stripped):
                return f"startup line contains crate fragment: {_line_snippet(stripped)}"

    for line in lines[banner_idx:]:
        if "] >" in line:
            if any(token in line for token in BUILD_NOISE_TOKENS):
                return f"prompt line contains build marker: {_line_snippet(line)}"
            if CRATE_FRAGMENT_RE.search(line):
                return f"prompt line contains crate fragment: {_line_snippet(line)}"

    if "Ignored queued input(s)" in clean:
        return "queued-input warning appeared in transcript"

    if "Follow-up:" in clean:
        return "unexpected follow-up queue output appeared in transcript"

    return None


def run_smoke(
    binary: str,
    startup_timeout: float,
    step_timeout: float,
    shell_command: str,
    with_build_dev: bool,
) -> SmokeResult:
    temp_home = tempfile.TemporaryDirectory(prefix="caliche-repl-smoke-")
    env = os.environ.copy()
    env["HOME"] = temp_home.name
    env["CALICHE_DISABLE_STARTUP_PROMPTS"] = "1"
    os.makedirs(os.path.join(temp_home.name, ".caliche"), exist_ok=True)

    pid, fd = pty.fork()
    if pid == 0:
        if shell_command:
            os.execvpe("/bin/zsh", ["/bin/zsh", "-lc", shell_command], env)
        elif with_build_dev:
            cmd = f"./caliche build-dev && {binary} --debug"
            os.execvpe("/bin/zsh", ["/bin/zsh", "-lc", cmd], env)
        else:
            os.execvpe(binary, [binary, "--debug"], env)

    out = bytearray()
    prompt_count = 0
    sent_help = False
    sent_exit = False

    def decoded() -> str:
        return out.decode(errors="ignore")

    started_at = time.time()
    last_progress = started_at

    while time.time() - started_at < max(10.0, startup_timeout + step_timeout * 3):
        ready, _, _ = select.select([fd], [], [], 0.2)
        if ready:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            out.extend(chunk)
            last_progress = time.time()

            # Answer cursor-position queries so reedline does not consume
            # later keystrokes as a garbled CSI reply.
            while CURSOR_QUERY_RE.search(bytes(out)):
                os.write(fd, b"\x1b[24;1R")
                out[:] = CURSOR_QUERY_RE.sub(b"", bytes(out), count=1)

            text = decoded()
            clean = strip_ansi(text)
            prompt_count = len(PROMPT_RE.findall(clean))

            if not sent_help and prompt_count >= 1:
                os.write(fd, b"/help\n")
                sent_help = True
                continue

            if sent_help and not sent_exit and "Available Commands" in clean:
                os.write(fd, b"/exit\n")
                sent_exit = True
                continue

        if not sent_help and (time.time() - started_at) > startup_timeout:
            break
        if sent_help and not sent_exit and (time.time() - last_progress) > step_timeout:
            break

        if sent_exit:
            done, _ = os.waitpid(pid, os.WNOHANG)
            if done == pid:
                break

    transcript = decoded()
    clean = strip_ansi(transcript)

    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done == 0:
            os.kill(pid, signal.SIGTERM)
    except ChildProcessError:
        pass
    finally:
        temp_home.cleanup()

    if "Caliche" not in clean:
        return SmokeResult(False, "startup banner missing", transcript)
    if "Type /help" not in clean:
        return SmokeResult(False, "startup help line missing", transcript)
    if "Available Commands" not in clean:
        return SmokeResult(False, "/help did not print command list", transcript)
    if len(PROMPT_RE.findall(clean)) < 2:
        return SmokeResult(False, "did not observe two prompts (/help roundtrip failed)", transcript)
    if not with_build_dev and "Compiling " in clean:
        return SmokeResult(False, "build output interleaved into REPL transcript", transcript)
    if "\x00" in transcript:
        return SmokeResult(False, "NUL bytes detected in transcript", transcript)
    if "Ignored queued input(s)" in clean:
        return SmokeResult(False, "queued-input warning appeared in transcript", transcript)

    integrity_error = _validate_repl_integrity(clean)
    if integrity_error is not None:
        return SmokeResult(False, integrity_error, transcript)

    return SmokeResult(True, "repl smoke passed", transcript)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic REPL smoke test")
    parser.add_argument(
        "--binary",
        default="./target/debug/caliche",
        help="Path to caliche binary (default: ./target/debug/caliche)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=12.0,
        help="Seconds to wait for first prompt",
    )
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for /help roundtrip and /exit",
    )
    parser.add_argument(
        "--transcript-out",
        default="",
        help="Optional file path to write full transcript",
    )
    parser.add_argument(
        "--with-build-dev",
        action="store_true",
        help="Run `./caliche build-dev` before launching REPL in the same PTY",
    )
    parser.add_argument(
        "--shell-command",
        default="",
        help="Custom shell command to run (e.g. './caliche build-dev && ./target/debug/caliche --debug')",
    )
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}", file=sys.stderr)
        return 2

    result = run_smoke(
        args.binary,
        args.startup_timeout,
        args.step_timeout,
        args.shell_command,
        args.with_build_dev,
    )

    if args.transcript_out:
        with open(args.transcript_out, "w", encoding="utf-8") as f:
            f.write(result.transcript)

    print(result.reason)
    if not result.ok:
        print("---- transcript excerpt ----")
        excerpt = strip_ansi(result.transcript)
        print(excerpt[-2000:])
        print("---- end excerpt ----")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
