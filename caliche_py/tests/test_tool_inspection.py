"""Tests for Goose-style tool inspection."""

from __future__ import annotations

from caliche import (
    InspectionAction,
    PathSecurityInspector,
    PermissionMode,
    PermissionModeInspector,
    ToolInspectionManager,
    default_inspection_manager,
)


def test_permission_mode_interactive_factory():
    assert str(PermissionMode.Interactive) == "Interactive"
    assert str(PermissionMode.from_string("interactive")) == "Interactive"


def test_path_security_inspector_denies_foreign_absolute():
    inspector = PathSecurityInspector("/tmp/my-workspace")
    allowed = inspector.inspect("fs", {"op": "read", "path": "src/lib.rs"}, "1")
    assert allowed[0].action.kind() == "allow"

    denied = inspector.inspect("fs", {"op": "read", "path": "/etc/passwd"}, "2")
    assert denied[0].action.kind() == "deny"


def test_permission_mode_inspector_deny_all():
    inspector = PermissionModeInspector(PermissionMode.DenyAll)
    results = inspector.inspect("fs", {"op": "read", "path": "."}, "c1")
    assert results[0].action.kind() == "deny"


def test_default_inspection_manager_and_tool_manager():
    manager = default_inspection_manager(PermissionMode.AcceptAll, "/tmp/ws")
    action = manager.inspect_one("fs", {"op": "read", "path": "a.rs"}, "x")
    assert action.kind() == "allow"

    built = ToolInspectionManager()
    built.add_permission_mode(PermissionMode.DenyAll)
    denied = built.inspect_one("bash", {"command": "ls"}, "b")
    assert denied.kind() == "deny"
    assert "permission_mode" in built.inspector_names()


def test_inspection_action_helpers():
    assert InspectionAction.allow().kind() == "allow"
    assert InspectionAction.deny("no").reason() == "no"
    assert InspectionAction.require_approval("ask").kind() == "require_approval"
