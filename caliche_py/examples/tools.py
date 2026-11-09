#!/usr/bin/env python3
"""
Tools Example

This example demonstrates:
- Built-in tools (calculator, echo, file operations, HTTP, etc.)
- Implementing custom tools using the Tool interface
- Using toolkits to bundle related tools
- Tool credentials management

Best practices shown:
- Proper error handling in tool execution
- Clear tool descriptions and parameters
- Explicit tool selection for security
- Credential management patterns

Run with:
```bash
uv run examples/tools.py
"""

import json
import caliche
from caliche.tools import Tool, FunctionTool, tool
from caliche import ToolResult
from pydantic import BaseModel, Field


class ReverseStringTool(Tool):
    """Example tool that reverses a string."""
    
    def name(self) -> str:
        return "reverse_string"
    
    def description(self) -> str:
        return "Reverses a string. Use this tool to reverse any text string. Takes a 'text' parameter and returns the reversed text."
    
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to reverse"
                }
            },
            "required": ["text"]
        }
    
    def execute(self, arguments: dict) -> ToolResult:
        text = arguments.get("text", "")
        if not text:
            return ToolResult(
                success=False,
                content="",
                error="Missing 'text' parameter"
            )
        
        reversed_text = text[::-1]
        result = {
            "original": text,
            "reversed": reversed_text
        }
        return ToolResult(
            success=True,
            content=json.dumps(result),
            error=None
        )


class WordCountTool(Tool):
    """Example tool that counts words in text."""
    
    def name(self) -> str:
        return "count_words"
    
    def description(self) -> str:
        return "Counts the number of words, characters, and lines in a text string. Use this tool to analyze text statistics."
    
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze"
                }
            },
            "required": ["text"]
        }
    
    def execute(self, arguments: dict) -> ToolResult:
        text = arguments.get("text", "")
        if not text:
            return ToolResult(
                success=False,
                content="",
                error="Missing 'text' parameter"
            )
        
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.splitlines())
        
        result = {
            "text": text,
            "words": word_count,
            "characters": char_count,
            "lines": line_count
        }
        return ToolResult(
            success=True,
            content=json.dumps(result),
            error=None
        )


class CalculatorToolkit:
    """A custom toolkit that bundles calculator-related tools"""
    
    def name(self):
        return "calculator"
    
    def get_tools(self):
        class AddTool:
            def name(self):
                return "add"
            def description(self):
                return "Adds two numbers together. Use this tool to perform addition."
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["a", "b"]
                }
            def execute(self, arguments):
                result = arguments.get("a", 0) + arguments.get("b", 0)
                return ToolResult(
                    success=True,
                    content=json.dumps({"result": result})
                )
        
        class MultiplyTool:
            def name(self):
                return "multiply"
            def description(self):
                return "Multiplies two numbers together. Use this tool to perform multiplication."
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["a", "b"]
                }
            def execute(self, arguments):
                result = arguments.get("a", 0) * arguments.get("b", 0)
                return ToolResult(
                    success=True,
                    content=json.dumps({"result": result})
                )
        
        return [AddTool(), MultiplyTool()]


def built_in_tools():
    """Example 1: Built-in Tools"""
    print("=== Example 1: Built-in Tools ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Bedrock)
        builder.model("anthropic.claude-3-sonnet-20240229-v1:0")
        builder.enable_tool("calculator")
        builder.enable_tool("fs")
        agent = builder.build()
        
        print("Available built-in tools:")
        tools = agent.list_tools()
        for tool in tools:
            print(f"  - {tool['name']}")
        
        response, _, _ = agent.chat("What is 42 multiplied by 17?")
        print(f"\nQuery: What is 42 multiplied by 17?")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def custom_tools():
    """Example 2: Custom Tools"""
    print("=== Example 2: Custom Tools ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Bedrock)
        builder.model("anthropic.claude-3-sonnet-20240229-v1:0")
        builder.add_tool(ReverseStringTool())
        builder.add_tool(WordCountTool())
        builder.enable_tool("calculator")
        # Don't enable Python execution to ensure custom tools are used
        agent = builder.build()
        
        # Use explicit prompts that mention the tool names
        response1, _, _ = agent.chat(
            "Use the reverse_string tool to reverse this text: 'Hello, World!'"
        )
        print("Query: Use the reverse_string tool to reverse this text: 'Hello, World!'")
        print(f"Response: {response1}\n")
        
        response2, _, _ = agent.chat(
            "Use the count_words tool to count the words, characters, and lines in this text: 'The quick brown fox jumps over the lazy dog'"
        )
        print("Query: Use the count_words tool to analyze the text")
        print(f"Response: {response2}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def toolkits():
    """Example 3: Toolkits"""
    print("=== Example 3: Toolkits ===\n")
    
    toolkit = CalculatorToolkit()
    print(f"Toolkit name: {toolkit.name()}")
    tools = toolkit.get_tools()
    print(f"Tools in toolkit: {', '.join([t.name() for t in tools])}\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Bedrock)
        builder.model("anthropic.claude-3-sonnet-20240229-v1:0")
        builder.register_toolkit(toolkit)
        # Don't enable built-in calculator to ensure toolkit tools are used
        agent = builder.build()
        
        print("Toolkit registered successfully")
        print(f"Agent created: {agent.provider}, {agent.model}\n")
        
        # Use explicit prompt mentioning the tool name
        response, _, _ = agent.chat("Use the add tool to calculate 5 + 3")
        print("Query: Use the add tool to calculate 5 + 3")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def decorator_tools():
    """Example 4: @tool Decorator and FunctionTool"""
    print("=== Example 4: @tool Decorator and FunctionTool ===\n")
    
    # Example 4a: Using @tool decorator with type hints
    @tool
    def get_weather(city: str, unit: str = "celsius") -> str:
        """Get the current weather for a specified city. Use this tool to retrieve weather information."""
        # Simulate weather data
        temp = 22 if unit == "celsius" else 72
        return json.dumps({
            "location": city,
            "temperature": temp,
            "units": unit,
            "condition": "sunny"
        })
    
    # Example 4b: Using @tool decorator with custom name and description
    @tool(
        name="math_operation",
        description="Perform basic math operations (add, subtract, multiply, divide) on two numbers. Use this tool for mathematical calculations."
    )
    def calculate(a: int, b: int, operation: str = "add") -> str:
        """Calculate two numbers using the specified operation."""
        if operation == "add":
            result = a + b
        elif operation == "multiply":
            result = a * b
        elif operation == "subtract":
            result = a - b
        elif operation == "divide":
            if b == 0:
                return json.dumps({"error": "Division by zero"})
            result = a / b
        else:
            return json.dumps({"error": f"Unknown operation: {operation}"})
        
        return json.dumps({"result": result, "operation": operation, "a": a, "b": b})
    
    # Example 4c: Using FunctionTool directly with Pydantic model
    class WeatherArgs(BaseModel):
        city: str = Field(description="The city name")
        unit: str = Field(default="celsius", description="Temperature unit")
    
    def get_weather_pydantic(city: str, unit: str = "celsius") -> str:
        """Get weather using Pydantic validation."""
        temp = 22 if unit == "celsius" else 72
        return json.dumps({
            "location": city,
            "temperature": temp,
            "units": unit,
            "condition": "sunny"
        })
    
    weather_pydantic_tool = FunctionTool(
        get_weather_pydantic,
        args_schema=WeatherArgs
    )
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Bedrock)
        builder.model("anthropic.claude-3-sonnet-20240229-v1:0")
        builder.add_tool(get_weather)  # Decorated function becomes a tool
        builder.add_tool(calculate)  # Another decorated tool
        builder.add_tool(weather_pydantic_tool)  # FunctionTool with Pydantic
        # Don't enable built-in tools to ensure custom tools are used
        agent = builder.build()
        
        print("Tools created using @tool decorator and FunctionTool")
        print(f"Weather tool name: {get_weather.name()}")
        print(f"Math tool name: {calculate.name()}\n")
        
        # Use explicit prompts that mention the tool names
        response1, _, _ = agent.chat(
            "Use the get_weather tool to get the weather in Tokyo in celsius"
        )
        print("Query: Use the get_weather tool to get the weather in Tokyo")
        print(f"Response: {response1}\n")
        
        response2, _, _ = agent.chat(
            "Use the math_operation tool to multiply 12 by 8"
        )
        print("Query: Use the math_operation tool to multiply 12 by 8")
        print(f"Response: {response2}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def multiple_tools():
    """Example 5: Multiple Tools"""
    print("=== Example 5: Multiple Tools ===\n")
    
    try:
        builder = caliche.AgentBuilder()
        builder.provider(caliche.ProviderType.Bedrock)
        builder.model("anthropic.claude-3-sonnet-20240229-v1:0")
        builder.add_tool(ReverseStringTool())
        builder.add_tool(WordCountTool())
        builder.enable_tool("calculator")
        # Don't enable Python execution to ensure custom tools are used
        agent = builder.build()
        
        # Use explicit prompt mentioning tool names
        response, _, _ = agent.chat(
            "First use the calculator tool to calculate 5 * 7, then use the reverse_string tool to reverse the result as text, and finally use the count_words tool to count the words in the reversed text"
        )
        print("Query: Combine multiple tools (calculator, reverse_string, count_words)")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"(Skipped: {e})\n")


def main():
    """Main execution"""
    print("=== Tools Examples ===\n")
    
    built_in_tools()
    custom_tools()
    toolkits()
    decorator_tools()
    multiple_tools()
    
    print("=== Examples Complete ===")
    print("\nTips:")
    print("- Use built-in tools when possible for reliability")
    print("- Create custom tools for domain-specific functionality")
    print("- Use @tool decorator for quick tool creation from functions")
    print("- Use FunctionTool with Pydantic models for advanced validation")
    print("- Use toolkits to bundle related tools together")
    print("- Always validate tool parameters and handle errors gracefully")


if __name__ == "__main__":
    main()

