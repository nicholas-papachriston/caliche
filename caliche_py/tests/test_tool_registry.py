"""Tests for active/deferred tool registry controls."""

from __future__ import annotations

from caliche import (
    ToolRegistry,
    create_apply_patch_tool,
    create_calculator_tool,
)


def test_calculator_tool_executes_add():
    tool = create_calculator_tool()
    result = tool.execute({"operation": "add", "a": 2, "b": 2})
    assert result["success"] is True, result
    assert "4" in result["content"]


def test_inactive_tools_are_hidden_from_schemas_and_execution():
    registry = ToolRegistry()
    registry.register(create_calculator_tool())
    registry.register_deferred(create_apply_patch_tool())
    schemas = registry.get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "calculator"
    assert registry.is_active("apply_patch") is False

    denied = registry.execute(
        "apply_patch",
        {
            "patch": "*** Begin Patch\n*** Add File: /tmp/x\n+y\n*** End Patch\n",
        },
    )
    assert denied["success"] is False
    assert "not active" in (denied.get("error") or "")

    registry.activate(["apply_patch"])
    assert registry.is_active("apply_patch") is True
    assert len(registry.get_tool_schemas()) == 2
    registry.deactivate(["apply_patch"])
    assert registry.is_active("apply_patch") is False
    registry.activate_all()
    assert registry.is_active("apply_patch") is True
