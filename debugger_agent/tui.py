"""Alt-screen 4-pane Textual TUI for the debugging agent.

Layout:
  ┌───────────────────────────┬──────────────────┐
  │  Code (gutter: ● bp, ▶ pc)│  Assistant       │
  │  syntax hl, current-line  ├──────────────────┤
  │                           │  Locals          │
  ├───────────────────────────┴──────────────────┤
  │  Tool output (replaced each step)             │
  └───────────────────────────────────────────────┘

Consumes the same `agent.run()` event stream as the CLI. The agent (blocking
network + tool I/O) runs on a worker thread; events are marshalled onto the UI
thread via `call_from_thread`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Set

from rich.syntax import Syntax
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import RichLog, Static

from . import events
from .agent import DebugAgent


def _basename(path: Optional[str]) -> Optional[str]:
    return os.path.basename(path) if path else None


def render_code(code: str, current_line: Optional[int], bp_lines: Set[int]) -> Text:
    """Render source with a gutter (● breakpoint, ▶ current line) + syntax hl."""
    try:
        highlighted = Syntax(code, "python", theme="ansi_dark").highlight(code)
        line_texts = highlighted.split("\n")
    except Exception:  # noqa: BLE001 — fall back to plain text
        line_texts = [Text(l) for l in code.splitlines()]

    out = Text()
    for i, line_text in enumerate(line_texts, start=1):
        if i in bp_lines:
            marker, gstyle = "●", "bold red"
        elif i == current_line:
            marker, gstyle = "▶", "bold yellow"
        else:
            marker, gstyle = " ", "dim"
        row = Text.assemble(Text(f"{marker} {i:>4} ", style=gstyle), line_text)
        if i == current_line:
            row.stylize("on grey23")
        out.append(row)
        out.append("\n")
    return out


class DebuggerTUI(App):
    CSS = """
    #top { height: 1fr; }
    #code { width: 62%; border: round $primary; padding: 0 1; }
    #right { width: 38%; }
    #assistant { height: 1fr; border: round $secondary; padding: 0 1; }
    #locals { height: 12; border: round $accent; padding: 0 1; }
    #output { height: 34%; border: round $warning; padding: 0 1; }
    """

    BINDINGS = [("q", "quit_app", "Quit")]

    def __init__(self, agent: DebugAgent):
        super().__init__()
        self.agent = agent
        self.success = False
        self._current_file: Optional[str] = None
        self._current_line: Optional[int] = None
        self._bps: Dict[str, Set[int]] = {}

    # ── layout ─────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="top"):
                with VerticalScroll(id="code"):
                    yield Static("", id="code_body")
                with Vertical(id="right"):
                    yield RichLog(id="assistant", wrap=True, markup=True)
                    yield Static("(no locals yet)", id="locals")
            yield RichLog(id="output", wrap=True, markup=True, highlight=False)

    def on_mount(self) -> None:
        self.query_one("#code").border_title = "Code"
        self.query_one("#assistant").border_title = "Assistant"
        self.query_one("#locals").border_title = "Locals"
        self.query_one("#output").border_title = "Tool output"
        self.title = "debug"
        self._run_agent()

    def action_quit_app(self) -> None:
        self.exit(0 if self.success else 1)

    # ── agent worker ───────────────────────────────────────────────

    @work(thread=True)
    def _run_agent(self) -> None:
        for ev in self.agent.run():
            self.call_from_thread(self._handle, ev)

    def _handle(self, ev: events.Event) -> None:
        assistant = self.query_one("#assistant", RichLog)
        output = self.query_one("#output", RichLog)

        if isinstance(ev, events.SessionStart):
            self.sub_title = ev.workspace_dir
            show = ev.test_file or (ev.files[0] if ev.files else None)
            if show:
                self._show_file(show)
        elif isinstance(ev, events.Reasoning):
            assistant.write(Text.assemble((f"● step {ev.step}\n", "bold cyan"),
                                          (ev.text, "")))
        elif isinstance(ev, events.ToolCall):
            params = ", ".join(f"{k}={v}" for k, v in ev.parameters.items())
            assistant.write(Text(f"→ {ev.name}({params})", style="bold blue"))
            if ev.name in ("edit_file", "write_file"):
                fp = _basename(ev.parameters.get("file_path"))
                if fp:
                    self._show_file(fp)
        elif isinstance(ev, events.ToolResult):
            output.clear()
            output.write(Text(f"{ev.name}\n", style="bold"))
            output.write(ev.result.rstrip())
        elif isinstance(ev, events.DebuggerState):
            self._apply_state(ev)
        elif isinstance(ev, events.TestResult):
            style = "bold green" if ev.passed else "bold red"
            output.write(Text(f"\n{'PASS' if ev.passed else 'FAIL'}  {ev.summary}",
                              style=style))
        elif isinstance(ev, events.AgentError):
            output.write(Text(f"error [{ev.code}]: {ev.message}", style="bold red"))
        elif isinstance(ev, events.SessionComplete):
            self.success = ev.success
            tag = "[green]✔ fixed[/]" if ev.success else "[yellow]✗ not fixed[/]"
            self.sub_title = f"{tag} — {ev.steps_taken} steps · press q to quit"

    # ── state → code pane ──────────────────────────────────────────

    def _apply_state(self, ev: events.DebuggerState) -> None:
        if ev.breakpoints is not None:
            self._bps = {}
            for bp in ev.breakpoints:
                f = _basename(bp.get("file"))
                if f and bp.get("line") is not None:
                    self._bps.setdefault(f, set()).add(int(bp["line"]))
        if ev.position:
            f = _basename(ev.position.get("file"))
            self._current_line = ev.position.get("line")
            if f:
                self._show_file(f, self._current_line)
        elif ev.breakpoints is not None and self._current_file:
            self._show_file(self._current_file, self._current_line)
        if ev.locals:
            body = "\n".join(f"[bold]{k}[/] = {v}" for k, v in ev.locals.items())
            self.query_one("#locals", Static).update(body or "(no locals)")

    def _show_file(self, filename: str, current_line: Optional[int] = None) -> None:
        name = _basename(filename)
        self._current_file = name
        if current_line is not None:
            self._current_line = current_line
        path = Path(self.agent.codebase_dir) / name
        try:
            code = path.read_text()
        except OSError:
            code = f"# (could not read {name})"
        line = self._current_line if name == _basename(self._current_file) else None
        bps = self._bps.get(name, set())
        self.query_one("#code_body", Static).update(render_code(code, line, bps))
        code_pane = self.query_one("#code")
        bp_hint = f"  ●{sorted(bps)}" if bps else ""
        code_pane.border_title = f"Code · {name}{bp_hint}"


def run_tui(agent: DebugAgent) -> int:
    """Run the TUI to completion; returns a process exit code."""
    app = DebuggerTUI(agent)
    app.run()
    return 0 if app.success else 1
