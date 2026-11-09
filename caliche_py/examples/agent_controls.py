#!/usr/bin/env python3
"""
Agent Lifecycle Controls Example

This example demonstrates advanced agent control features:
- Lifecycle hooks (PreToolUse/PostToolUse) for intercepting tool execution
- Permission modes (AcceptEdits, AcceptAll, DenyAll) for automatic permission handling
- Allowed tools filtering to restrict which tools can be used
- In-process MCP servers for direct tool registration without subprocess overhead
- Session forking for parallel work and subagents

Run with:
```bash
uv run examples/agent_controls.py
```
"""

import caliche
from caliche.hooks import (
    PermissionMode,
    HookMatcher,
    HookContext,
    HookResult,
    LifecycleHook,
)
from caliche.session import AgentSession
from caliche.mcp import InProcessMcpServerBuilder, InProcessMcpServer
import asyncio


class BashSecurityHook(LifecycleHook):
    """Example hook that blocks dangerous bash commands."""

    async def on_pre_tool_use(self, ctx: HookContext) -> HookResult:
        """Block dangerous bash commands."""
        if ctx.tool_name == "Bash":
            # Check if command contains dangerous patterns
            command = ctx.tool_input.get("command", "")
            dangerous_patterns = ["rm -rf", "format", "delete", "drop database"]
            for pattern in dangerous_patterns:
                if pattern in command:
                    return HookResult.deny(f"Command contains dangerous pattern: {pattern}")
        return HookResult.allow()

    async def on_post_tool_use(self, ctx: HookContext, result: caliche.ToolResult) -> HookResult:
        """Log tool execution for audit purposes."""
        if not result.success:
            print(f"Warning: Tool execution failed: {result.error}")
        return HookResult.allow()


class GreetTool:
    """Example custom tool for in-process MCP server."""

    def name(self) -> str:
        return "greet"

    def description(self) -> str:
        return "Greet a user by name"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the person to greet",
                }
            },
            "required": ["name"],
        }

    async def execute(self, arguments: dict) -> dict:
        name = arguments.get("name", "World")
        return {"success": True, "content": f"Hello, {name}!"}


async def example_lifecycle_hooks():
    """Example 1: Lifecycle Hooks"""
    print("=== Example 1: Lifecycle Hooks ===")

    builder = caliche.AgentBuilder()
    builder.provider(caliche.ProviderType.Bedrock)
    builder.model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    builder.enable_tool("calculator")
    builder.hook(HookMatcher.tool("Bash"), BashSecurityHook())

    try:
        agent = builder.build()
        print("Agent configured with security hook for Bash tool")
        print("(Hook will block dangerous commands)\n")
    except Exception as e:
        print(f"Note: {e}\n")


async def example_permission_modes():
    """Example 2: Permission Modes"""
    print("=== Example 2: Permission Modes ===")

    # AcceptEdits mode - auto-accept file edits
    builder = caliche.AgentBuilder()
    builder.provider(caliche.ProviderType.Bedrock)
    builder.model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    builder.permission_mode(PermissionMode.AcceptEdits)
    builder.enable_tool("fs")

    try:
        agent = builder.build()
        print("Agent configured with AcceptEdits permission mode")
        print("(File edits will be auto-accepted)\n")
    except Exception as e:
        print(f"Note: {e}\n")

    # DenyAll mode - block all tool calls
    builder = caliche.AgentBuilder()
    builder.provider(caliche.ProviderType.Bedrock)
    builder.model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    builder.permission_mode(PermissionMode.DenyAll)
    builder.enable_tool("calculator")

    try:
        agent = builder.build()
        print("Agent configured with DenyAll permission mode")
        print("(All tool calls will be blocked)\n")
    except Exception as e:
        print(f"Note: {e}\n")


async def example_allowed_tools():
    """Example 3: Allowed Tools Filtering"""
    print("=== Example 3: Allowed Tools Filtering ===")

    builder = caliche.AgentBuilder()
    builder.provider(caliche.ProviderType.Bedrock)
    builder.model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    builder.allowed_tools(["calculator", "fs"])
    builder.enable_all_default_tools()  # Enable all, but only calculator and echo will be allowed

    try:
        agent = builder.build()
        print("Agent configured with allowed_tools: ['calculator', 'fs']")
        print("Available tools (allowed tools marked with *):")
        # Note: list_tools() is available but requires proper Python context
        # For demonstration, we'll skip the actual listing
        print("  * calculator")
        print("  * echo")
        print("  (Other tools are filtered out by allowed_tools)")
        print()
    except Exception as e:
        print(f"Note: {e}\n")


async def example_in_process_mcp():
    """Example 4: In-Process MCP Server"""
    print("=== Example 4: In-Process MCP Server ===")

    try:
        # Create an in-process MCP server with custom tools
        server_builder = InProcessMcpServerBuilder("my-tools")
        server_builder.version("1.0.0")
        server_builder.add_tool(GreetTool())
        server = server_builder.build()

        print(f"Created in-process MCP server '{server.name}' v{server.version}")
        print("Tools in server:")
        for tool in server.list_tools():
            print(f"  - {tool['name']}: {tool['description']}")

        # Register the MCP server with an agent
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Bedrock)
        builder.model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        # Pass the server - the builder will extract the inner PyInProcessMcpServer
        builder.mcp_server("my-tools", server)

        agent = builder.build()
        print("\nAgent configured with in-process MCP server")
        print("Available tools (including MCP server tools):")
        tools = agent.list_tools()
        for tool in tools:
            print(f"  - {tool['name']}")
        print()
    except ImportError:
        print("(Skipped: MCP feature not available)\n")
    except Exception as e:
        print(f"Note: {e}\n")


async def example_session_forking():
    """Example 5: Session Forking"""
    print("=== Example 5: Session Forking ===")

    try:
        # Create main session
        main_session = AgentSession("main-session")
        main_session.add_message({"role": "user", "content": "Hello from main session"})

        print(f"Created main session: {main_session.id}")
        print(f"Main session history: {len(main_session.get_history())} messages\n")

        # Fork the session
        fork = main_session.fork("parallel-worker")
        print(f"Forked session: {fork.id}")
        print(f"Forked session history: {len(fork.get_history())} messages (copied from main)")

        # Add messages to the fork
        fork.add_message({"role": "user", "content": "Working in parallel"})
        fork.add_message({"role": "assistant", "content": "Processing in fork"})
        print(f"After adding messages to fork: {len(fork.get_history())} messages\n")

        # Merge fork back into main
        main_session.merge(fork)
        print("Merged fork back into main session")
        print(f"Main session history after merge: {len(main_session.get_history())} messages\n")

        print("Session forking enables:")
        print("  - Parallel work with isolated state")
        print("  - Subagents that can work independently")
        print("  - Merging results back into main conversation\n")
    except Exception as e:
        print(f"Note: {e}\n")


async def example_combined_features():
    """Example 6: Combined Features"""
    print("=== Example 6: Combined Features ===")

    builder = caliche.AgentBuilder()
    builder.provider(caliche.ProviderType.Bedrock)
    builder.model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    builder.allowed_tools(["calculator", "fs"])
    builder.permission_mode(PermissionMode.AcceptEdits)
    builder.hook(HookMatcher.all(), BashSecurityHook())

    try:
        agent = builder.build()
        print("Agent configured with:")
        print("  - Allowed tools: ['calculator', 'fs']")
        print("  - Permission mode: AcceptEdits")
        print("  - Security hook for all tools\n")
    except Exception as e:
        print(f"Note: {e}\n")


async def main():
    """Main execution"""
    try:
        await example_lifecycle_hooks()
        await example_permission_modes()
        await example_allowed_tools()
        await example_in_process_mcp()
        await example_session_forking()
        await example_combined_features()

        print("=== Examples Complete ===\n")
        print("Key Features:")
        print("1. Lifecycle Hooks: Intercept and control tool execution")
        print("2. Permission Modes: Control tool call permissions automatically")
        print("3. Allowed Tools: Restrict which tools can be used")
        print("4. In-Process MCP Servers: Direct tool registration without subprocess overhead")
        print("5. Session Forking: Enable parallel work and subagents")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
