#!/usr/bin/env python3
"""ACP v1 fixture agent.

Stdout is JSON-RPC only. Diagnostics go to stderr.
"""

from __future__ import annotations

import json
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "happy"
PENDING_CANCEL = False
SESSION_COUNT = 0
PARKED_FOREIGN_NOTES = 37


def log(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def read_message():
    while True:
        line = sys.stdin.readline()
        if line == "":
            return None
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            log("NON_JSON_STDIN " + stripped[:200])
            raise SystemExit(2)


def result(message_id, payload: dict) -> None:
    send({"jsonrpc": "2.0", "id": message_id, "result": payload})


def text_update(session_id: str, kind: str, text: str) -> None:
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": kind,
                    "content": {"type": "text", "text": text},
                },
            },
        }
    )


def hermes_setup() -> dict:
    return {
        "id": "hermes-setup",
        "name": "Terminal setup",
        "description": "Run hermes --setup in a terminal",
        "type": "terminal",
        "args": ["--setup"],
    }


def opencode_login() -> dict:
    return {
        "id": "opencode-login",
        "name": "Login with opencode",
        "description": "Run `opencode auth login` in the terminal",
    }


def auth_methods() -> list:
    if MODE == "auth-terminal":
        return [hermes_setup()]
    if MODE == "auth-codex":
        return [
            {
                "id": "openai-codex",
                "name": "openai-codex runtime credentials",
                "description": "Authenticate Hermes using the currently configured openai-codex runtime credentials.",
                "type": "agent",
            },
            hermes_setup(),
        ]
    if MODE in {"auth-opencode", "auth-required"}:
        return [opencode_login()]
    if MODE == "auth-vendor":
        return [
            {
                "id": "cursor-account",
                "name": "Cursor account",
                "description": "Sign in with your Cursor subscription",
                "type": "agent",
            }
        ]
    if MODE == "auth-generic":
        return [
            {
                "id": "custom-login",
                "name": "Sign in",
                "description": "Complete authenticate before using the agent",
            }
        ]
    if MODE == "auth-openrouter":
        return [
            {
                "id": "openrouter",
                "name": "openrouter runtime credentials",
                "description": "Authenticate Hermes using the currently configured openrouter runtime credentials.",
            }
        ]
    if MODE == "auth-custom-terminal":
        return [
            {
                "id": "custom-terminal",
                "name": "Custom terminal",
                "description": "Run a terminal login",
                "type": "terminal",
            }
        ]
    return []


def log_client_auth(params: dict) -> None:
    caps = params.get("clientCapabilities") or {}
    auth = caps.get("auth") or {}
    meta = caps.get("_meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    log(f"client-auth-terminal {auth.get('terminal')!r}")
    log(f"client-terminal-auth {meta.get('terminal-auth')!r}")


def on_initialize(message_id, params: dict) -> None:
    version = params.get("protocolVersion")
    log(f"initialize protocolVersion={version}")
    log_client_auth(params)
    if MODE == "stdout-log":
        sys.stdout.write("not-json log line\n")
        sys.stdout.flush()
    answered = 2 if MODE == "version-2" else 1
    result(
        message_id,
        {
            "protocolVersion": answered,
            "agentCapabilities": {"loadSession": False},
            "agentInfo": {
                "name": "acp-fixture",
                "title": "ACP fixture",
                "version": "0.0.0",
            },
            "authMethods": auth_methods(),
        },
    )


def on_prompt(message_id, params: dict) -> None:
    global PENDING_CANCEL
    session_id = params.get("sessionId") or ""
    if MODE == "auth-required":
        send(
            {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {"code": -32000, "message": "Authentication required"},
            }
        )
        return
    if MODE == "permission":
        send(
            {
                "jsonrpc": "2.0",
                "id": 9001,
                "method": "session/request_permission",
                "params": {
                    "sessionId": session_id,
                    "toolCall": {"toolCallId": "call-1", "title": "read file"},
                    "options": [
                        {
                            "optionId": "allow",
                            "name": "Allow",
                            "kind": "allow_once",
                        },
                        {
                            "optionId": "reject",
                            "name": "Reject",
                            "kind": "reject_once",
                        },
                    ],
                },
            }
        )
        reply = read_message()
        if reply is None:
            return
        encoded = json.dumps(reply, separators=(",", ":"))
        log("permission-reply " + encoded)
        text_update(session_id, "agent_message_chunk", "permission closed")
        result(message_id, {"stopReason": "end_turn"})
        return
    if MODE == "cross-session":
        text_update(session_id, "agent_message_chunk", "local-turn")
        if session_id == "sess-a":
            for index in range(PARKED_FOREIGN_NOTES):
                text_update("sess-b", "agent_message_chunk", f"parked-{index:03d}")
        else:
            text_update(session_id, "agent_message_chunk", "live-b")
        result(message_id, {"stopReason": "end_turn"})
        return
    text_update(session_id, "agent_message_chunk", "hello from fixture")
    if MODE == "hold-for-cancel":
        while not PENDING_CANCEL:
            nxt = read_message()
            if nxt is None:
                return
            if nxt.get("method") == "session/cancel":
                PENDING_CANCEL = True
        PENDING_CANCEL = False
        result(message_id, {"stopReason": "cancelled"})
        return
    text_update(session_id, "agent_thought_chunk", "thinking")
    result(message_id, {"stopReason": "end_turn"})


def on_session_new(message_id, params: dict) -> None:
    global SESSION_COUNT
    cwd = params.get("cwd") or ""
    log(f"cwd {cwd}")
    if not str(cwd).startswith("/"):
        log("RELATIVE_CWD")
    if MODE == "cross-session":
        SESSION_COUNT += 1
        session_id = "sess-a" if SESSION_COUNT == 1 else "sess-b"
        result(message_id, {"sessionId": session_id})
        return
    result(message_id, {"sessionId": "sess-fixture"})


def main() -> None:
    global PENDING_CANCEL
    if MODE == "silent":
        while read_message() is not None:
            pass
        return
    while True:
        message = read_message()
        if message is None:
            return
        method = message.get("method")
        log(f"method {method}")
        message_id = message.get("id")
        params = message.get("params") or {}
        if method == "initialize":
            on_initialize(message_id, params)
            if MODE == "silent-after-init":
                while read_message() is not None:
                    pass
                return
            continue
        if method == "session/new":
            on_session_new(message_id, params)
            if MODE == "silent-after-session":
                while read_message() is not None:
                    pass
                return
            continue
        if method == "session/prompt":
            on_prompt(message_id, params)
            continue
        if method == "session/cancel":
            PENDING_CANCEL = True
            continue
        if message_id is not None:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        raise SystemExit(0)
