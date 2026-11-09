"""
Workflow Orchestration Example

Demonstrates how to create and execute DAG-based workflows with
conditional branching, parallel execution, loops, and human-in-the-loop.
"""

import asyncio
from caliche import AgentBuilder, WorkflowBuilder


async def main():
    print("Workflow Orchestration Example\n")

    # Create agents for different tasks
    research_agent = await AgentBuilder().set_provider("openai").set_model("gpt-4").set_system_prompt(
        "You are a research assistant."
    ).build()

    writing_agent = await AgentBuilder().set_provider("openai").set_model("gpt-4").set_system_prompt(
        "You are a technical writer."
    ).build()

    review_agent = await AgentBuilder().set_provider("openai").set_model("gpt-4").set_system_prompt(
        "You are a quality reviewer."
    ).build()

    # Build a workflow
    workflow = (
        WorkflowBuilder()
        # Add agent nodes
        .add_agent("research", "Research the topic: {{topic}}. Provide key findings.")
        .add_agent("write", "Write a comprehensive article based on: {{research_output}}")
        .add_agent("review", "Review and improve the article: {{write_output}}")
        # Add a condition node
        .add_condition("needs_revision", "{{review_score}} < 0.8")
        # Add a loop for revisions
        .add_loop("revision_loop", 3, "{{needs_revision}}")
        # Add a human-in-the-loop checkpoint
        .add_human_interrupt(
            "final_approval",
            "Please review the final article and approve or request changes.",
            ["approved", "feedback"],
        )
        # Connect nodes
        .connect_nodes("research", "write")
        .connect_nodes("write", "review")
        .connect_nodes("review", "needs_revision")
        .connect_nodes("needs_revision", "revision_loop", "needs_revision == true")
        .connect_nodes("needs_revision", "final_approval", "needs_revision == false")
        .connect_nodes("revision_loop", "review")
        # Set entry point
        .set_entry_point("research")
        .build()
    )

    # Create workflow engine
    engine = workflow.create_engine()

    # Execute workflow
    print("Executing workflow...")
    result = await engine.execute({"topic": "Artificial Intelligence in Healthcare"})

    print(f"Workflow state: {result['state']}")
    print(f"Output: {result.get('output', {})}")

    # Check for pending interrupts
    interrupts = await engine.get_pending_interrupts()
    if interrupts:
        print("\nPending interrupts:")
        for interrupt in interrupts:
            print(f"  - {interrupt['node_name']}: {interrupt['message']}")

        # Resume with human input
        print("\nResuming with approval...")
        await engine.resume({"approved": True, "feedback": "Looks good!"})

        # Check final state
        final_state = await engine.get_state()
        print(f"Final state: {final_state}")


if __name__ == "__main__":
    asyncio.run(main())

