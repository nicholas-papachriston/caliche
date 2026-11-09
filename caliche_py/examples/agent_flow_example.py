#!/usr/bin/env python3
"""
Agent Flow Example for Caliche Python bindings

This example demonstrates:
- Creating multiple specialized agents
- Building agent flows with FlowBuilder
- Executing multi-agent workflows with handoffs
- Generating Mermaid flow diagrams

uv run examples/agent_flow_example.py

Note: This example requires provider credentials to be set in environment variables.
"""

import caliche
import json
import os

def main():
    print("=== Agent Flow Example ===\n")

    # Check if provider is configured
    if not os.getenv("AGENT_PROVIDER") and not os.getenv("AGENT_MODEL"):
        print("Note: Set AGENT_PROVIDER and AGENT_MODEL environment variables")
        print("      to run this example with actual agents.\n")
        print("Example:")
        print("  export AGENT_PROVIDER=bedrock")
        print("  export AGENT_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0\n")
        return

    try:
        # Create specialized agents
        print("1. Creating specialized agents...")
        
        # Research agent - specializes in research tasks
        research_builder = caliche.AgentBuilder()
        research_builder.system_prompt(
            "You are a research specialist. When you need to write content, "
            "transfer to the writer agent using transfer_to tool."
        )
        research_builder.enable_tool("transfer_to")
        research_agent = research_builder.build()
        
        print("   Research agent created")

        # Writer agent - specializes in writing
        writer_builder = caliche.AgentBuilder()
        writer_builder.system_prompt(
            "You are a writing specialist. Focus on creating clear, "
            "well-structured content."
        )
        writer_agent = writer_builder.build()
        
        print("   Writer agent created\n")

        # Build the flow
        print("2. Building agent flow...")
        flow_builder = caliche.FlowBuilder()
        flow_builder.add_node("researcher", research_agent)
        flow_builder.add_node("writer", writer_agent)
        flow_builder.set_entry_point("researcher")
        flow = flow_builder.build()
        
        print(f"   Flow entry point: {flow.entry_point()}")
        print(f"   Available nodes: {flow.node_names()}\n")

        # Generate Mermaid diagram
        print("3. Flow visualization (Mermaid):")
        mermaid = flow.to_mermaid()
        print(mermaid)
        print()

        # Execute the flow (commented out as it requires actual LLM calls)
        print("4. Flow execution:")
        print("   (To execute the flow, uncomment the code below)")
        print("   result = flow.run('Research Rust benefits and write a summary')")
        print("   print(result['final_response'])")

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This example requires valid provider credentials.")

    print("\n=== Example Complete ===")
    print("\nTips:")
    print("- Use FlowBuilder to chain multiple agents together")
    print("- Agents can hand off to each other using transfer_to tool")
    print("- Use to_mermaid() to visualize your agent flows")

if __name__ == "__main__":
    main()

