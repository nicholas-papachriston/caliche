"""
LangChain LLM Provider Adapter for Caliche.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LangChainLLMProvider:
    """Adapter that wraps a LangChain chat model."""

    def __init__(
        self, llm: Any, supports_native_tools: bool = True, supports_streaming: bool = True
    ) -> None:
        self.llm = llm
        self.supports_native_tools = supports_native_tools
        self.supports_streaming = supports_streaming
        self._model_name = (
            getattr(llm, "model_name", None) or getattr(llm, "model", None) or "langchain-llm"
        )

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        request_obj = self._normalize_request(request)
        messages_data = self._extract_messages(request_obj)
        langchain_messages = self._convert_messages_to_langchain(messages_data)

        llm_to_use = self.llm
        if request_obj.get("tools"):
            tools_json = request_obj.get("tools")
            tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
            langchain_tools = self._convert_tools_to_langchain(tools)
            if langchain_tools and hasattr(llm_to_use, "bind_tools"):
                llm_to_use = llm_to_use.bind_tools(langchain_tools)

        if request_obj.get("temperature") is not None and hasattr(llm_to_use, "temperature"):
            llm_to_use.temperature = request_obj["temperature"]
        if request_obj.get("max_tokens") is not None:
            if hasattr(llm_to_use, "max_tokens"):
                llm_to_use.max_tokens = request_obj["max_tokens"]
            elif hasattr(llm_to_use, "maxTokens"):
                llm_to_use.maxTokens = request_obj["max_tokens"]

        try:
            if hasattr(llm_to_use, "invoke") and callable(getattr(llm_to_use, "invoke", None)):
                response = llm_to_use.invoke(langchain_messages)
            else:
                raise ValueError(f"LLM {type(llm_to_use)} does not have invoke method")
        except Exception as exc:
            logger.exception("Exception in LLM invocation")
            raise RuntimeError(f"LangChain LLM invocation failed: {exc}") from exc

        return self._convert_langchain_response(response)

    def stream(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        request_obj = self._normalize_request(request)
        messages_data = self._extract_messages(request_obj)
        langchain_messages = self._convert_messages_to_langchain(messages_data)

        llm_to_use = self.llm
        if request_obj.get("tools"):
            tools_json = request_obj.get("tools")
            tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
            langchain_tools = self._convert_tools_to_langchain(tools)
            if langchain_tools and hasattr(llm_to_use, "bind_tools"):
                llm_to_use = llm_to_use.bind_tools(langchain_tools)

        if request_obj.get("temperature") is not None and hasattr(llm_to_use, "temperature"):
            llm_to_use.temperature = request_obj["temperature"]
        if request_obj.get("max_tokens") is not None:
            if hasattr(llm_to_use, "max_tokens"):
                llm_to_use.max_tokens = request_obj["max_tokens"]
            elif hasattr(llm_to_use, "maxTokens"):
                llm_to_use.maxTokens = request_obj["max_tokens"]

        events: list[dict[str, Any]] = []
        try:
            if hasattr(llm_to_use, "stream") and callable(getattr(llm_to_use, "stream", None)):
                for chunk in llm_to_use.stream(langchain_messages):
                    event = self._convert_langchain_chunk(chunk)
                    if event:
                        events.append(event)
            else:
                raise ValueError("LLM does not support streaming")
        except Exception as exc:
            raise RuntimeError(f"LangChain streaming failed: {exc}") from exc

        events.append({"type": "done"})
        return events

    def _normalize_request(self, request: dict[str, Any] | Any) -> dict[str, Any]:
        if isinstance(request, str):
            try:
                request_obj = json.loads(request)
            except json.JSONDecodeError:
                request_obj = {"prompt": request}
        elif isinstance(request, dict):
            request_obj = request
        else:
            request_obj = {}

        nested = request_obj.get("request")
        if isinstance(nested, dict):
            return nested
        return request_obj

    def _extract_messages(self, request_obj: dict[str, Any]) -> list[dict[str, Any]]:
        raw_messages = request_obj.get("messages", request_obj.get("input"))

        if isinstance(raw_messages, str):
            try:
                parsed = json.loads(raw_messages)
            except json.JSONDecodeError:
                return [{"role": "user", "content": raw_messages}]
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
            return []

        if isinstance(raw_messages, list):
            return copy.deepcopy(raw_messages)
        if isinstance(raw_messages, dict):
            return [copy.deepcopy(raw_messages)]

        prompt = request_obj.get("prompt")
        if isinstance(prompt, str) and prompt:
            return [{"role": "user", "content": prompt}]

        return []

    def _convert_messages_to_langchain(self, messages_data: Any) -> list[Any]:
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        except ImportError as exc:
            raise ImportError(
                "langchain-core is required. Install with: pip install langchain-core"
            ) from exc

        langchain_messages: list[Any] = []
        for msg_data in messages_data:
            role = msg_data.get("role")
            content = msg_data.get("content")
            text = (
                content.get("text", content.get("content", ""))
                if isinstance(content, dict)
                else str(content or "")
            )

            if role == "user":
                langchain_messages.append(HumanMessage(content=text))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=text))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=text))
            elif role == "tool":
                tool_call_id = msg_data.get("tool_call_id", "")
                langchain_messages.append(ToolMessage(content=text, tool_call_id=tool_call_id))

        return langchain_messages

    def _convert_tools_to_langchain(self, tools: Any) -> list[dict[str, Any]] | None:
        if not tools:
            return None

        langchain_tools: list[dict[str, Any]] = []
        for tool in tools:
            tool_name = tool.get("function", {}).get("name", "")
            tool_desc = tool.get("function", {}).get("description", "")
            tool_params = tool.get("function", {}).get("parameters", {})
            langchain_tools.append(
                {"name": tool_name, "description": tool_desc, "parameters": tool_params}
            )

        return langchain_tools if langchain_tools else None

    def _convert_langchain_response(self, response: Any) -> dict[str, Any]:
        content = str(getattr(response, "content", ""))
        tool_calls = None
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls = []
            for tc in response.tool_calls:
                tool_calls.append(
                    {
                        "id": getattr(tc, "id", ""),
                        "name": getattr(tc, "name", ""),
                        "arguments": json.dumps(getattr(tc, "args", {})),
                    }
                )

        result: dict[str, Any] = {
            "content": content,
            "finish_reason": getattr(response, "response_metadata", {}).get("finish_reason")
            if hasattr(response, "response_metadata")
            else None,
            "usage": None,
        }
        if tool_calls:
            result["tool_calls"] = tool_calls
            result["content"] = json.dumps({"content": content, "tool_calls": tool_calls})

        return result

    def _convert_langchain_chunk(self, chunk: Any) -> dict[str, Any] | None:
        if hasattr(chunk, "content"):
            return {"type": "text_chunk", "content": str(chunk.content)}
        return None
