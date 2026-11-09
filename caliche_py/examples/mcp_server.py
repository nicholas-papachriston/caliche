#!/usr/bin/env python3
"""
Example: Creating a custom MCP server with Python

This example demonstrates how to create a custom MCP server that exposes
custom tools via the Model Context Protocol.

uv run examples/mcp_server.py
"""

from caliche.mcp import McpServerBuilder, McpTool


class CalculatorTool:
    """A simple calculator tool for MCP."""

    def name(self) -> str:
        return "calculator"

    def description(self) -> str:
        return "Performs basic arithmetic operations (add, subtract, multiply, divide)"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform",
                },
                "a": {
                    "type": "number",
                    "description": "First number",
                },
                "b": {
                    "type": "number",
                    "description": "Second number",
                },
            },
            "required": ["operation", "a", "b"],
        }

    async def execute(self, arguments: dict) -> dict:
        operation = arguments.get("operation")
        a = arguments.get("a")
        b = arguments.get("b")

        try:
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return {"success": False, "content": "", "error": "Division by zero"}
                result = a / b
            else:
                return {"success": False, "content": "", "error": f"Unknown operation: {operation}"}

            return {
                "success": True,
                "content": f"Result: {result}",
            }
        except Exception as e:
            return {"success": False, "content": "", "error": str(e)}


class WeatherTool:
    """A mock weather tool for MCP."""

    def name(self) -> str:
        return "get_weather"

    def description(self) -> str:
        return "Gets the current weather for a location (mock implementation)"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for",
                },
            },
            "required": ["location"],
        }

    async def execute(self, arguments: dict) -> dict:
        location = arguments.get("location", "")
        # Mock weather data
        return {
            "success": True,
            "content": f"Weather in {location}: Sunny, 72°F",
        }


class FileInfoTool:
    """A file info tool that demonstrates reading file metadata."""

    def name(self) -> str:
        return "get_file_info"

    def description(self) -> str:
        return "Gets information about a file (mock implementation)"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to get information about",
                },
            },
            "required": ["path"],
        }

    async def execute(self, arguments: dict) -> dict:
        path = arguments.get("path", "")
        # Mock file info
        return {
            "success": True,
            "content": f"File: {path}\nSize: 1024 bytes\nType: text/plain\nModified: 2024-01-01T00:00:00Z",
        }


def main():
    """Create and run an MCP server."""
    try:
        # Create server builder
        server = (
            McpServerBuilder()
            .name("example-mcp-server")
            .version("1.0.0")
            .enable_tools(True)
            .enable_prompts(False)
            .enable_resources(False)
            .add_tool(CalculatorTool())
            .add_tool(WeatherTool())
            .add_tool(FileInfoTool())
            .build()
        )

        import json
        import os
        import sys
        
        print("Starting MCP server on stdio...", file=sys.stderr)
        print("Server name: example-mcp-server", file=sys.stderr)
        print("Server version: 1.0.0", file=sys.stderr)
        print("Available tools: calculator, get_weather, get_file_info", file=sys.stderr)
        print("", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print("MCP Client Configuration", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print("", file=sys.stderr)
        print("Add this to your MCP client configuration (e.g., Claude Desktop):", file=sys.stderr)
        print("", file=sys.stderr)
        
        # Get the script path
        script_path = os.path.abspath(__file__)
        python_executable = sys.executable
        
        config = {
            "mcpServers": {
                "example-mcp-server": {
                    "command": python_executable,
                    "args": [script_path]
                }
            }
        }
        
        print(json.dumps(config, indent=2), file=sys.stderr)
        print("", file=sys.stderr)
        print("", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print("Send JSON-RPC requests to stdin", file=sys.stderr)

        # Run the server (this blocks until stdin is closed)
        server.run_stdio()
    except Exception as e:
        print(f"Error running MCP server: {e}", file=__import__("sys").stderr)
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()

