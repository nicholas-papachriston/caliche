"""Tests for JSONL session trees and branch restore."""

from __future__ import annotations

from pathlib import Path

import pytest

from caliche import (
    JsonlSessionEntry,
    JsonlSessionHeader,
    JsonlSessionStore,
    JsonlSessionTree,
    new_entry_id,
    parse_jsonl_session,
)


def test_jsonl_session_branch_restore(tmp_path: Path):
    path = tmp_path / "chat.jsonl"
    store = JsonlSessionStore(str(path))
    store.create(JsonlSessionHeader("sess", "/proj"))
    store.append_entry(JsonlSessionEntry.message("u1", {"role": "user", "content": "one"}, None))
    store.append_entry(
        JsonlSessionEntry.message("a1", {"role": "assistant", "content": "two"}, "u1")
    )

    tree = store.load()
    branch = tree.branch_from("u1", "br", "retry")
    store.append_entry(tree.entries()[-1])
    store.append_entry(
        JsonlSessionEntry.message("a2", {"role": "assistant", "content": "alt"}, branch)
    )

    reloaded = store.load()
    main = reloaded.restore_branch_messages("a1")
    alt = reloaded.restore_branch_messages("a2")
    assert len(main) == 2
    assert len(alt) == 2
    assert main[1]["content"] == "two"
    assert alt[1]["content"] == "alt"
    assert branch == "br"


def test_parse_jsonl_session_and_new_entry_id():
    entry_id = new_entry_id()
    assert isinstance(entry_id, str)
    assert len(entry_id) >= 4

    text = "\n".join(
        [
            '{"type":"session","version":3,"id":"s1","timestamp":"2024-01-01T00:00:00Z"}',
            '{"type":"message","id":"m1","parentId":null,"timestamp":"2024-01-01T00:00:01Z","message":{"role":"user","content":"hi"}}',
            "",
        ]
    )
    tree = parse_jsonl_session(text)
    assert tree.header.id == "s1"
    assert tree.entry_count() == 1
    assert tree.leaves() == ["m1"]


def test_jsonl_session_tree_rejects_unknown_parent():
    tree = JsonlSessionTree(JsonlSessionHeader("s"))
    with pytest.raises(RuntimeError, match="unknown parentId"):
        tree.append(JsonlSessionEntry.message("x", {"role": "user", "content": "z"}, "missing"))


def test_jsonl_entry_factories_cover_compaction_and_model_change():
    header = JsonlSessionHeader("s")
    header.set_parent_session("parent-sess")
    assert header.parent_session == "parent-sess"

    tree = JsonlSessionTree(header)
    tree.append(JsonlSessionEntry.message("m1", {"role": "user", "content": "hi"}, None))
    tree.append(JsonlSessionEntry.compaction_summary("c1", "summarized", "m1", 120))
    tree.append(JsonlSessionEntry.model_change("mc1", "openai", "gpt-4o", "c1"))
    assert tree.entry_count() == 3
    assert tree.leaves() == ["mc1"]
