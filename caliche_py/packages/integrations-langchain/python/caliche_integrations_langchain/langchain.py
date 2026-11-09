"""
LangChain Tool and chain adapters for Caliche.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from caliche_core import ToolResult


class LangChainTool:
    """Adapter that wraps a LangChain tool for Caliche."""

    def __init__(self, tool: Any) -> None:
        self._tool = tool

    def name(self) -> str:
        return self._tool.name

    def description(self) -> str:
        return self._tool.description

    def parameters(self) -> dict[str, Any]:
        if hasattr(self._tool, "args_schema") and self._tool.args_schema is not None:
            try:
                args_schema = self._tool.args_schema
                if hasattr(args_schema, "model_json_schema"):
                    schema = args_schema.model_json_schema()
                elif hasattr(args_schema, "schema"):
                    schema = args_schema.schema()
                elif hasattr(self._tool, "input_schema"):
                    schema = self._tool.input_schema
                    if callable(schema):
                        schema = schema()
                else:
                    schema = getattr(args_schema, "__schema__", {})
                    if not schema:
                        schema = {"type": "object", "properties": {}, "required": []}

                if isinstance(schema, dict) and "type" in schema:
                    return schema
                return {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                }
            except Exception:
                return {"type": "object", "properties": {}, "required": []}

        if hasattr(self._tool, "input_schema"):
            try:
                schema = self._tool.input_schema
                if callable(schema):
                    schema = schema()
                if isinstance(schema, dict):
                    return schema
            except Exception:
                pass

        return {"type": "object", "properties": {}, "required": []}

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            if hasattr(self._tool, "invoke"):
                result = self._tool.invoke(arguments)
            elif hasattr(self._tool, "run"):
                result = self._tool.run(arguments)
            else:
                result = self._tool(arguments)

            content = result if isinstance(result, str) else json.dumps(result)
            return ToolResult(success=True, content=content, error=None)
        except Exception as exc:
            return ToolResult(success=False, content="", error=str(exc))


class LangChainMemoryAdapter:
    """Adapter that bridges LangChain memory to Caliche memory strategies."""

    def __init__(self, langchain_memory: Any) -> None:
        self._memory = langchain_memory

    def get_messages(self) -> list[dict[str, Any]]:
        try:
            if hasattr(self._memory, "chat_memory"):
                messages = self._memory.chat_memory.messages
            elif hasattr(self._memory, "messages"):
                messages = self._memory.messages
            else:
                return []

            caliche_messages: list[dict[str, Any]] = []
            for msg in messages:
                role = "user"
                if hasattr(msg, "type"):
                    if msg.type == "human":
                        role = "user"
                    elif msg.type == "ai":
                        role = "assistant"
                    elif msg.type == "system":
                        role = "system"

                content = str(getattr(msg, "content", ""))
                caliche_messages.append({"role": role, "content": content})

            return caliche_messages
        except Exception:
            return []

    def add_message(self, role: str, content: str) -> None:
        try:
            if hasattr(self._memory, "chat_memory"):
                if role == "user":
                    self._memory.chat_memory.add_user_message(content)
                elif role == "assistant":
                    self._memory.chat_memory.add_ai_message(content)
        except Exception:
            pass


class LangChainRetrieverAdapter:
    """Adapter that uses LangChain retrievers for RAG in Caliche."""

    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever

    async def get_relevant_documents(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        try:
            if hasattr(self._retriever, "aget_relevant_documents"):
                docs = await self._retriever.aget_relevant_documents(query)
            elif hasattr(self._retriever, "get_relevant_documents"):
                loop = asyncio.get_running_loop()
                docs = await loop.run_in_executor(
                    None, lambda: self._retriever.get_relevant_documents(query)
                )
            else:
                return []

            results: list[dict[str, Any]] = []
            for doc in docs[:top_k]:
                if hasattr(doc, "page_content"):
                    content = doc.page_content
                elif hasattr(doc, "content"):
                    content = doc.content
                elif isinstance(doc, str):
                    content = doc
                else:
                    content = ""
                metadata = getattr(doc, "metadata", {})
                results.append({"content": content, "metadata": metadata})
            return results
        except Exception:
            return []


class LangChainChainAdapter:
    """Adapter that wraps LangChain chains as Caliche chains."""

    def __init__(self, chain: Any) -> None:
        self._chain = chain

    async def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            if hasattr(self._chain, "ainvoke"):
                result = await self._chain.ainvoke(inputs)
            elif hasattr(self._chain, "invoke"):
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: self._chain.invoke(inputs))
            elif hasattr(self._chain, "arun"):
                result = await self._chain.arun(**inputs)
            elif hasattr(self._chain, "run"):
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: self._chain.run(**inputs))
            else:
                raise ValueError("Chain does not expose invoke/ainvoke/run/arun")

            if isinstance(result, dict):
                return result
            if isinstance(result, str):
                return {"output": result}
            return {"output": str(result)}
        except Exception as exc:
            raise RuntimeError(f"LangChain chain execution failed: {exc}") from exc
