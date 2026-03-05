"""Events yielded by the agent and consumed by the CLI and Textual TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Event:
    """Base class for all harness events. `step` is the 0-based turn index."""
    step: int


@dataclass
class SessionStart(Event):
    """Emitted once at the start: the workspace and its visible files."""
    workspace_dir: str
    files: List[str]
    test_file: Optional[str] = None


@dataclass
class Reasoning(Event):
    """A chunk of the assistant's natural-language reasoning for this turn.

    Tool-call XML is stripped out before emission; the structured call is
    delivered separately as a `ToolCall` event.
    """
    text: str


@dataclass
class ToolCall(Event):
    """A parsed tool call the agent decided to make."""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult(Event):
    """The observation returned by executing a `ToolCall`."""
    name: str
    result: str


@dataclass
class DebuggerState(Event):
    """Structured pdb state, read directly off the live session (not parsed).

    Any field may be None when unchanged / unavailable.
    """
    position: Optional[Dict[str, Any]] = None       # {"file": str, "line": int}
    breakpoints: Optional[List[Dict[str, Any]]] = None  # [{"file","line"}]
    locals: Optional[Dict[str, str]] = None


@dataclass
class TestResult(Event):
    """Outcome of a test run detected in a tool observation."""
    passed: bool
    summary: str = ""


@dataclass
class AgentError(Event):
    """A non-fatal error (model request failed, tool raised, etc.)."""
    message: str
    code: str = "error"


@dataclass
class SessionComplete(Event):
    """Terminal event. `success` reflects the last observed test signal."""
    success: bool
    steps_taken: int
