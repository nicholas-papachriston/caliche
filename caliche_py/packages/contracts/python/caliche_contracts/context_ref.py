"""Shared `@` context-reference parsing (ROADMAP 2.10 / 2.12)."""

from __future__ import annotations

from typing import Literal, TypedDict, Union


class FileRef(TypedDict):
    kind: Literal["file"]
    path: str


class DirectoryRef(TypedDict):
    kind: Literal["directory"]
    path: str


class UrlRef(TypedDict):
    kind: Literal["url"]
    url: str


class SymbolRef(TypedDict):
    kind: Literal["symbol"]
    name: str


class ClipboardRef(TypedDict):
    kind: Literal["clipboard"]


class GitDiffRef(TypedDict):
    kind: Literal["git_diff"]


class GitStatusRef(TypedDict):
    kind: Literal["git_status"]


class ArtifactRef(TypedDict):
    kind: Literal["artifact"]
    selector: str


class ResourceRef(TypedDict):
    kind: Literal["resource"]
    server: str | None
    uri: str


ContextRefKind = Union[
    FileRef,
    DirectoryRef,
    UrlRef,
    SymbolRef,
    ClipboardRef,
    GitDiffRef,
    GitStatusRef,
    ArtifactRef,
    ResourceRef,
]


def parse_context_ref_kind(input_text: str) -> ContextRefKind | None:
    """Parse a context reference from a string without the leading `@`."""
    trimmed = input_text.strip()
    if not trimmed:
        return None

    if trimmed.startswith("artifact:"):
        return {"kind": "artifact", "selector": trimmed[len("artifact:") :]}

    if trimmed.startswith("resource:"):
        return _parse_resource_kind(trimmed[len("resource:") :])

    if trimmed in {"clipboard", "clip"}:
        return {"kind": "clipboard"}
    if trimmed in {"git:diff", "diff"}:
        return {"kind": "git_diff"}
    if trimmed in {"git:status", "status"}:
        return {"kind": "git_status"}

    if trimmed.startswith("http://") or trimmed.startswith("https://"):
        return {"kind": "url", "url": trimmed}

    if trimmed.endswith("/"):
        return {"kind": "directory", "path": trimmed}
    if "." in trimmed or "/" in trimmed:
        return {"kind": "file", "path": trimmed}
    return {"kind": "symbol", "name": trimmed}


def _parse_resource_kind(selector: str) -> ContextRefKind | None:
    trimmed = selector.strip()
    if not trimmed:
        return None
    if "::" in trimmed:
        server, uri = trimmed.split("::", 1)
        return {"kind": "resource", "server": server, "uri": uri}
    return {"kind": "resource", "server": None, "uri": trimmed}
