"""Tests for WorkflowBuilder and workflow JSON export."""

import importlib
import json

import pytest

from caliche import WorkflowBuilder
from caliche._runtime import load_runtime


def test_workflow_builder_uses_shared_native_runtime():
    """Root and runtime import paths must resolve to one extension module."""
    runtime = load_runtime()
    root_native = importlib.import_module("caliche._caliche_py")
    assert runtime is root_native


def test_workflow_builder_build_and_to_json():
    b = WorkflowBuilder("unit_wf")
    b.add_agent("step_a", "Say {{input}}")
    b.set_entry_point("step_a")
    wf = b.build()
    assert wf.name() == "unit_wf"
    raw = wf.to_json()
    data = json.loads(raw)
    assert data.get("name") == "unit_wf"
    assert any(n.get("name") == "step_a" for n in data.get("nodes", []))


def test_collect_merge_without_fan_out_errors():
    b = WorkflowBuilder("bad_merge")
    b.add_agent("only", "x")
    with pytest.raises(RuntimeError, match="fan_out"):
        b.collect_merge("m", "concatenate")


def test_add_parallel_requires_branch_dicts():
    b = WorkflowBuilder("para")
    with pytest.raises(RuntimeError, match="at least one branch"):
        b.add_parallel([])
