from __future__ import annotations

from caliche_contracts import (
    parse_context_ref_kind,
    parse_permission_mode,
    parse_slash_intent,
)


def test_parse_context_ref_kind_covers_common_mentions() -> None:
    assert parse_context_ref_kind("diff") == {"kind": "git_diff"}
    assert parse_context_ref_kind("clip") == {"kind": "clipboard"}
    assert parse_context_ref_kind("https://example.com") == {
        "kind": "url",
        "url": "https://example.com",
    }
    assert parse_context_ref_kind("src/main.rs") == {
        "kind": "file",
        "path": "src/main.rs",
    }
    assert parse_context_ref_kind("artifact:run-1") == {
        "kind": "artifact",
        "selector": "run-1",
    }
    assert parse_context_ref_kind("resource:mcp::file://x") == {
        "kind": "resource",
        "server": "mcp",
        "uri": "file://x",
    }


def test_parse_slash_intent_covers_shared_host_intents() -> None:
    assert parse_slash_intent("/plan") == {"intent": "plan"}
    assert parse_slash_intent("/agent research") == {
        "intent": "agent",
        "arg": "research",
    }
    assert parse_slash_intent("/showplan") == {"intent": "show_plan"}
    assert parse_slash_intent("/attach ./a.png ./b.pdf") == {
        "intent": "attach",
        "paths": ["./a.png", "./b.pdf"],
    }
    assert parse_slash_intent("/help") is None
    assert parse_slash_intent("hello") is None


def test_parse_permission_mode_accepts_one_spelling() -> None:
    assert parse_permission_mode("Accept-Edits") == "accept-edits"
    assert parse_permission_mode("deny-all") == "deny-all"
    assert parse_permission_mode("deny_all") is None
    assert parse_permission_mode("acceptall") is None
    assert parse_permission_mode("nope") is None
