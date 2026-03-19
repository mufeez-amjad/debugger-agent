"""`debug` CLI entry point.

Default (non-alt-screen): stream the agent's reasoning + tool calls + tool
results to stdout — scriptable, and the surface other agents (opencode, Claude
Code) shell out to. `--view` launches the Textual 4-pane TUI instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Iterator

from .agent import DEFAULT_MODEL_NAME, DEFAULT_MODEL_URL, AgentConfig, DebugAgent
from . import events


# ── ANSI (stdlib only; auto-off when not a tty or NO_COLOR set) ────────

class _C:
    def __init__(self, enabled: bool):
        self.on = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def dim(self, s): return self._w("2", s)
    def bold(self, s): return self._w("1", s)
    def cyan(self, s): return self._w("36", s)
    def blue(self, s): return self._w("34;1", s)
    def green(self, s): return self._w("32;1", s)
    def red(self, s): return self._w("31;1", s)
    def yellow(self, s): return self._w("33", s)


def _truncate(text: str, max_lines: int = 40) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    hidden = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n… ({hidden} more lines)"


def _fmt_params(params: dict) -> str:
    parts = []
    for k, v in params.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "…"
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def render_stdout(event_stream: Iterator[events.Event], *, color: bool) -> int:
    """Render the agent event stream to stdout. Returns a process exit code."""
    c = _C(color)
    exit_code = 1
    for ev in event_stream:
        if isinstance(ev, events.SessionStart):
            files = ", ".join(ev.files) if ev.files else "(none)"
            print(c.bold("● debugging ") + c.dim(ev.workspace_dir))
            print(c.dim(f"  files: {files}"))
            print()
        elif isinstance(ev, events.Reasoning):
            print(c.cyan(ev.text))
        elif isinstance(ev, events.ToolCall):
            print(c.blue(f"→ {ev.name}(") + c.dim(_fmt_params(ev.parameters)) + c.blue(")"))
        elif isinstance(ev, events.ToolResult):
            body = _truncate(ev.result.strip())
            print(c.dim("  " + body.replace("\n", "\n  ")))
        elif isinstance(ev, events.DebuggerState):
            if ev.position:
                print(c.dim(f"  · at {ev.position['file']}:{ev.position['line']}"))
        elif isinstance(ev, events.TestResult):
            tag = c.green("PASS") if ev.passed else c.red("FAIL")
            print(f"  {tag} {c.dim(ev.summary)}")
        elif isinstance(ev, events.AgentError):
            print(c.red(f"error [{ev.code}]: {ev.message}"), file=sys.stderr)
        elif isinstance(ev, events.SessionComplete):
            print()
            if ev.success:
                print(c.green(f"✔ fixed — tests pass ({ev.steps_taken} steps)"))
                exit_code = 0
            else:
                print(c.yellow(f"✗ not fixed ({ev.steps_taken} steps)"))
                exit_code = 1
    return exit_code


def render_json(event_stream: Iterator[events.Event]) -> int:
    """Emit each event as a JSON line (JSONL).

    This is the machine-readable stream consumed by an external UI over a pipe, and by other agents (opencode, Claude Code) that
    shell out and parse structured output. Returns a process exit code.
    """
    exit_code = 1
    for ev in event_stream:
        record = {"type": type(ev).__name__, **dataclasses.asdict(ev)}
        sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        if isinstance(ev, events.SessionComplete):
            exit_code = 0 if ev.success else 1
    return exit_code


# ── argument parsing ───────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="debug",
        description="Run the debugging agent against a failing test.",
    )
    p.add_argument("-p", "--prompt", required=True, help="What to debug/fix.")
    p.add_argument("--view", action="store_true",
                   help="Launch the alt-screen 4-pane TUI instead of stdout.")
    p.add_argument("--json", action="store_true",
                   help="Emit the event stream as JSONL (for external UIs / agents).")
    p.add_argument("--dir", default=None,
                   help="Directory to debug in (default: current working dir).")
    p.add_argument("--model-url", default=os.environ.get("LLM_DEBUGGER_URL", DEFAULT_MODEL_URL),
                   help="OpenAI-compatible base URL. Defaults to $LLM_DEBUGGER_URL, "
                        "else %(default)s. Set it to your Modal endpoint after "
                        "`modal deploy`.")
    p.add_argument("--model-name", default=os.environ.get("LLM_DEBUGGER_MODEL", DEFAULT_MODEL_NAME),
                   help="Model name in the request (default: %(default)s).")
    p.add_argument("--api-key", default=None,
                   help="Bearer token for the endpoint. Defaults to $VLLM_API_KEY. "
                        "Required by the Modal deployment; a local server needs none.")
    p.add_argument("--max-turns", type=int, default=50,
                   help="Turn budget (default: %(default)s). The model paces "
                        "itself over about 40 turns, so a lower cap can stop a "
                        "session before it finishes.")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--tokenizer", default=os.environ.get("LLM_DEBUGGER_TOKENIZER"),
                   help="tokenizer.json for exact prompt-token counting: a file, "
                        "a directory, or an HF repo id (a repo id may download). "
                        "Defaults to $LLM_DEBUGGER_TOKENIZER, else the local HF "
                        "cache. Without one the context fit is estimated from "
                        "character counts, which is conservative and drops "
                        "history earlier than necessary.")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI color.")
    p.add_argument("--no-guided-decoding", action="store_true",
                   help="Decode tool calls without the xgrammar structural tag. "
                        "Use only to measure raw format robustness.")
    p.add_argument("--context-limit", type=int, default=None,
                   help="Server context window in tokens. Defaults to the "
                        "deployed endpoint's. Only ever an upper bound: if the "
                        "server reports a smaller window the agent adopts that "
                        "instead, so this rarely needs setting.")
    p.add_argument("--no-hints", action="store_true",
                   help="Do not append the observation hints (for example, a "
                        "wrong path or an empty pytest run). The model saw these "
                        "hints in training; use this only to measure behaviour "
                        "without them.")
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    codebase_dir = os.path.abspath(args.dir) if args.dir else os.getcwd()
    api_key = args.api_key or os.environ.get("VLLM_API_KEY")
    if not api_key and "modal" in args.model_url:
        sys.stderr.write(
            "error: the Modal endpoint requires an API key, but neither --api-key\n"
            "       nor $VLLM_API_KEY is set. It lives in the Modal secret\n"
            "       `debugger-agent-vllm-key` (key VLLM_API_KEY).\n"
            "       Use a local server instead with --model-url http://localhost:8000\n"
        )
        sys.exit(2)
    config = AgentConfig(
        model_url=args.model_url,
        model_name=args.model_name,
        api_key=api_key,
        guided_decoding=not args.no_guided_decoding,
        hints=not args.no_hints,
        **({"context_limit": args.context_limit} if args.context_limit else {}),
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tokenizer=args.tokenizer,
    )
    agent = DebugAgent(codebase_dir=codebase_dir, prompt=args.prompt, config=config)

    if args.view:
        try:
            from .tui import run_tui  # imported lazily (TUI dependency)
        except ImportError:
            sys.stderr.write(
                "error: --view needs Textual. Install it with:\n"
                "  pip install 'debugger-agent[tui]'   (or: pip install textual)\n"
            )
            sys.exit(2)
        sys.exit(run_tui(agent))

    if args.json:
        sys.exit(render_json(agent.run()))

    color = (not args.no_color) and sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    sys.exit(render_stdout(agent.run(), color=color))


if __name__ == "__main__":
    main()
