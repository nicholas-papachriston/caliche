"""
Workflow Orchestration Wrapper

Provides Python-friendly wrappers for workflow orchestration features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from caliche_runtime import _caliche_py as _runtime
else:
    from caliche._runtime import load_runtime

    _runtime = load_runtime()

PyWorkflow = _runtime.PyWorkflow
PyWorkflowBuilder = _runtime.PyWorkflowBuilder
PyWorkflowEngine = _runtime.PyWorkflowEngine


class WorkflowBuilder:
    """Builder for creating DAG-based workflows."""

    def __init__(self):
        self._builder = PyWorkflowBuilder()

    def add_agent(self, name: str, prompt_template: str) -> WorkflowBuilder:
        """Add an agent node to the workflow."""
        self._builder.add_agent(name, prompt_template)
        return self

    def add_agent_with_capability(
        self, name: str, capability: str, prompt_template: str
    ) -> WorkflowBuilder:
        """Add an agent node with a specific capability."""
        self._builder.add_agent_with_capability(name, capability, prompt_template)
        return self

    def add_condition(self, name: str, predicate: str) -> WorkflowBuilder:
        """Add a condition node for branching."""
        self._builder.add_condition(name, predicate)
        return self

    def add_parallel(self, name: str) -> WorkflowBuilder:
        """Add a parallel execution node."""
        self._builder.add_parallel(name)
        return self

    def add_merge(self, name: str, strategy: str = "concatenate") -> WorkflowBuilder:
        """Add a merge node for combining parallel results."""
        self._builder.add_merge(name, strategy)
        return self

    def add_loop(self, name: str, max_iterations: int, condition: str) -> WorkflowBuilder:
        """Add a loop node for iterative execution."""
        self._builder.add_loop(name, max_iterations, condition)
        return self

    def add_human_interrupt(
        self,
        name: str,
        message: str,
        required_fields: Optional[List[str]] = None,
    ) -> WorkflowBuilder:
        """Add a human-in-the-loop interrupt point."""
        self._builder.add_human_interrupt(name, message, required_fields or [])
        return self

    def connect_nodes(
        self, from_node: str, to_node: str, condition: Optional[str] = None
    ) -> WorkflowBuilder:
        """Connect two nodes in the workflow."""
        self._builder.connect_nodes(from_node, to_node, condition)
        return self

    def set_entry_point(self, name: str) -> WorkflowBuilder:
        """Set the entry point of the workflow."""
        self._builder.set_entry_point(name)
        return self

    def build(self) -> Workflow:
        """Build the workflow."""
        workflow = self._builder.build()
        return Workflow(workflow)


class Workflow:
    """Workflow instance."""

    def __init__(self, workflow: PyWorkflow):
        self._workflow = workflow

    def create_engine(self) -> WorkflowEngine:
        """Create a workflow engine to execute this workflow."""
        engine = self._workflow.create_engine()
        return WorkflowEngine(engine)


class WorkflowEngine:
    """Workflow engine for executing workflows."""

    def __init__(self, engine: PyWorkflowEngine):
        self._engine = engine

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cancel workflow if still running."""
        try:
            state = await self.get_state()
            if state not in ("completed", "cancelled", "failed"):
                await self.cancel()
        except Exception:
            pass  # Ignore errors during cleanup
        return False

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow with input data."""
        return await self._engine.execute(input_data)

    async def pause(self) -> None:
        """Pause workflow execution."""
        await self._engine.pause()

    async def resume(self, human_input: Optional[Dict[str, Any]] = None) -> None:
        """Resume workflow execution with optional human input."""
        await self._engine.resume(human_input or {})

    async def cancel(self) -> None:
        """Cancel workflow execution."""
        await self._engine.cancel()

    async def get_state(self) -> str:
        """Get current workflow state."""
        return await self._engine.get_state()

    async def get_pending_interrupts(self) -> List[Dict[str, Any]]:
        """Get pending interrupts requiring human input."""
        return await self._engine.get_pending_interrupts()


def create_workflow_builder() -> WorkflowBuilder:
    """Create a new workflow builder."""
    return WorkflowBuilder()
