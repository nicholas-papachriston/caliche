"""
Agent Coordinator Example

Demonstrates multi-agent coordination with load balancing,
capability-based routing, and inter-agent messaging.
"""

import asyncio
from caliche import AgentBuilder, AgentCoordinator


async def main():
    print("Agent Coordinator Example\n")

    # Create coordinator with load balancing strategy
    coordinator = AgentCoordinator.with_strategy("least_connections")

    # Create specialized agents
    research_agent = await AgentBuilder().set_provider("openai").set_model("gpt-4").set_system_prompt(
        "You are a research specialist."
    ).build()

    coding_agent = await AgentBuilder().set_provider("openai").set_model("gpt-4").set_system_prompt(
        "You are a coding specialist."
    ).build()

    writing_agent = await AgentBuilder().set_provider("openai").set_model("gpt-4").set_system_prompt(
        "You are a writing specialist."
    ).build()

    # Register agents with capabilities
    research_agent_id = coordinator.register_agent(
        "Research Agent", "Research Specialist", ["research", "data_analysis"], research_agent
    )
    print(f"Registered: {research_agent_id}")

    coding_agent_id = coordinator.register_agent(
        "Coding Agent", "Software Developer", ["coding"], coding_agent
    )
    print(f"Registered: {coding_agent_id}")

    writing_agent_id = coordinator.register_agent(
        "Writing Agent", "Technical Writer", ["writing"], writing_agent
    )
    print(f"Registered: {writing_agent_id}\n")

    # Delegate tasks based on capability
    print("Delegating research task...")
    research_result = coordinator.delegate_task(
        "Research the latest trends in machine learning", "research"
    )
    print(f"Agent {research_result['agent_id']} responded:")
    print(f"{research_result['response'][:200]}...\n")

    print("Delegating coding task...")
    coding_result = coordinator.delegate_task(
        "Write a Python function to calculate fibonacci numbers", "coding"
    )
    print(f"Agent {coding_result['agent_id']} responded:")
    print(f"{coding_result['response'][:200]}...\n")

    # Send message between agents
    print("Sending message between agents...")
    coordinator.send_message(
        research_agent_id, writing_agent_id, "Here are the research findings: [findings]"
    )
    print("Message sent\n")

    # Broadcast to all agents
    print("Broadcasting announcement...")
    coordinator.broadcast(
        research_agent_id, "Important: New research guidelines have been updated."
    )
    print("Broadcast sent\n")

    # Get coordinator statistics
    stats = coordinator.get_stats()
    print("Coordinator Statistics:")
    print(f"  Total agents: {stats['total_agents']}")
    print(f"  Total load: {stats['total_load']}")
    print(f"  Total capacity: {stats['total_capacity']}")
    print("\n  Agent Stats:")
    for agent_stat in stats["agent_stats"]:
        print(
            f"    {agent_stat['name']}: {agent_stat['current_load']}/{agent_stat['max_concurrent_tasks']}"
        )

    # Get message bus statistics
    message_bus = coordinator.get_message_bus()
    bus_stats = message_bus.get_stats()
    print("\nMessage Bus Statistics:")
    print(f"  Queue size: {bus_stats['queue_size']}")
    print(f"  Subscribers: {bus_stats['subscriber_count']}")


if __name__ == "__main__":
    asyncio.run(main())

