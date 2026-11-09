"""Tests for mutation style and apply_patch XOR."""

from __future__ import annotations

from pathlib import Path

from caliche import (
    AgentBuilder,
    MutationStyle,
    ProviderType,
    create_apply_patch_tool,
    create_fs_tool,
)


def test_mutation_style_factories():
    assert str(MutationStyle.EditWrite) == "EditWrite"
    assert str(MutationStyle.ApplyPatch) == "ApplyPatch"
    assert MutationStyle.from_string("apply_patch").__str__() == "ApplyPatch"


def test_create_fs_tool_apply_patch_style_strips_write_edit():
    tool = create_fs_tool(MutationStyle.ApplyPatch)
    assert tool.name() == "fs"
    params = tool.parameters()
    ops = params["properties"]["op"]["enum"]
    assert "read" in ops
    assert "write" not in ops
    assert "edit" not in ops

    result = tool.execute({"op": "write", "path": "x.txt", "content": "nope"})
    assert result["success"] is False
    assert "apply_patch" in (result.get("error") or "")


def test_create_apply_patch_tool_mutates_file(tmp_path: Path):
    target = tmp_path / "note.txt"
    patch = f"*** Begin Patch\n*** Add File: {target}\n+alpha\n+beta\n*** End Patch\n"
    tool = create_apply_patch_tool()
    assert tool.name() == "apply_patch"
    result = tool.execute({"patch": patch})
    assert result["success"] is True, result
    assert target.read_text() == "alpha\nbeta\n"


def test_agent_builder_accepts_mutation_and_inspection():
    assert hasattr(AgentBuilder, "mutation_style")
    assert hasattr(AgentBuilder, "enable_tool_inspection")
    assert hasattr(AgentBuilder, "enable_extended_default_tools")
    builder = AgentBuilder().mutation_style(MutationStyle.ApplyPatch)
    builder = builder.enable_tool_inspection("/tmp/ws")
    assert builder is not None


def test_builder_mutation_style_registers_apply_patch_on_built_agent():
    agent = (
        AgentBuilder()
        .provider(ProviderType.LMStudio)
        .model("local-model")
        .mutation_style(MutationStyle.ApplyPatch)
        .enable_tool("fs")
        .enable_tool("bash")
        .build()
    )
    tools = {tool["name"] for tool in agent.list_tools()}
    assert "apply_patch" in tools
    assert "fs" in tools
    assert "bash" in tools

    registry = agent.tool_registry()
    assert registry.is_active("apply_patch") is True
    schemas = registry.get_tool_schemas()
    fs_schema = next(s for s in schemas if s["name"] == "fs")
    ops = fs_schema["parameters"]["properties"]["op"]["enum"]
    assert "write" not in ops
    assert "edit" not in ops
