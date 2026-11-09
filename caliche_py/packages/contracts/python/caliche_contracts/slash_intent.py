"""Shared slash-command intent parsing for CLI/UI hosts."""

from __future__ import annotations

from typing import Literal, TypedDict, Union


class PlanIntent(TypedDict):
    intent: Literal["plan"]


class AskIntent(TypedDict):
    intent: Literal["ask"]


class AgentIntent(TypedDict):
    intent: Literal["agent"]
    arg: str | None


class ModeIntent(TypedDict):
    intent: Literal["mode"]


class ShowPlanIntent(TypedDict):
    intent: Literal["show_plan"]


class ResumeIntent(TypedDict):
    intent: Literal["resume"]
    id: str | None


class AttachIntent(TypedDict):
    intent: Literal["attach"]
    paths: list[str]


SlashIntent = Union[
    PlanIntent,
    AskIntent,
    AgentIntent,
    ModeIntent,
    ShowPlanIntent,
    ResumeIntent,
    AttachIntent,
]


def parse_slash_intent(input_text: str) -> SlashIntent | None:
    """Parse shared host intents from a slash-command line."""
    trimmed = input_text.strip()
    if not trimmed.startswith("/"):
        return None

    parts = trimmed.split()
    raw = parts[0].lower() if parts else ""
    args = parts[1:]

    if raw == "/plan":
        return {"intent": "plan"}
    if raw == "/ask":
        return {"intent": "ask"}
    if raw == "/agent":
        return {"intent": "agent", "arg": args[0] if args else None}
    if raw == "/mode":
        return {"intent": "mode"}
    if raw in {"/show-plan", "/showplan"}:
        return {"intent": "show_plan"}
    if raw == "/resume":
        return {"intent": "resume", "id": args[0] if args else None}
    if raw == "/attach":
        return {"intent": "attach", "paths": list(args)}
    return None
