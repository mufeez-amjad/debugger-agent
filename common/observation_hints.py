"""Append the training-time hints to tool observations during inference.

Hints use tool arguments, results, and session progress. The frame tracker is
per-session; reset the hinter before reusing it for another session."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Debugger tools whose output can carry a pdb stop-frame header, and whose
# arguments can name the expression that raised. The stepping ops belong here
# even though they read no state: a step_over result reports the frame it landed
# in, and that frame is what tells the AttributeError hint whether the session
# is paused inside __init__ before the assignment.
INSPECTION_DEBUG_TOOLS = {
    "print_variable",
    "examine_locals",
    "examine_globals",
    "list_code",
    "where_am_i",
    "inspect_stack",
    "run_expression",
    "step_over",
    "step_into",
    "step_out",
    "continue_execution",
    "add_breakpoint",
    "remove_breakpoint",
}

# Full stop-frame header including the function name:
# ``> /path/file.py(18)__init__()`` (function may be ``<module>``).
PDB_FRAME_RE = re.compile(r">\s*([^()\s]+?)\((\d+)\)([\w<>]+)\(\)")

MAX_OBSERVATION_CHARS = 4000


def pytest_collected_nothing(observation: Optional[str]) -> bool:
    """True when a pytest invocation executed ZERO tests (collection error).

    `pytest wrong/path.py` prints "collected 0 items", "no tests ran in ...s"
    and "ERROR: file or directory not found" (exit 4). Such a run carries no
    test evidence: the model saw neither a pass nor a fail. The caller must not
    count it as a pytest run, and it must not draw the "tests are still
    failing -> edit" hint. The fix is the path, not the code.
    """
    obs = (observation or "").lower()
    if not obs:
        return False
    return (
        "no tests ran" in obs
        or "error: file or directory not found" in obs
        or ("collected 0 items" in obs and " passed" not in obs and " failed" not in obs)
    )


def pytest_collection_errored(observation: Optional[str]) -> bool:
    """True when pytest FAILED TO COLLECT (exit 2): an import/syntax error in
    the code under test broke collection ("error during collection",
    "ERROR collecting", "Interrupted:").

    This differs from pytest_collected_nothing (exit 4: wrong test path, zero
    tests). Here the model's own last edit likely broke an import or syntax.
    The failing-test signal from the previous real run must stay (it is not a
    pass). The caller must not count this run as a pytest run, or a rerun guard
    would block every later pytest against an empty cache. The model must fix
    the file, not chase a path.
    """
    obs = (observation or "").lower()
    if not obs:
        return False
    return (
        "error during collection" in obs
        or "errors during collection" in obs
        or "error collecting" in obs
        or "interrupted:" in obs
        or "collection error" in obs
    )


def first_collection_error_lines(
    observation: Optional[str], max_lines: int = 3
) -> str:
    """Pull the most informative lines out of a pytest collection-error dump so
    the COLLECTION ERROR hint names the actual import/syntax problem.
    Prefers pytest's 'E   ' error lines and lines mentioning Error, else the
    'ERROR collecting' header."""
    text = observation or ""
    picked: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if (
            line.startswith("E ")
            or line.startswith("E   ")
            or "error collecting" in low
            or low.endswith("error")
            or "error:" in low
            or "importerror" in low
            or "modulenotfounderror" in low
            or "syntaxerror" in low
            or "indentationerror" in low
        ):
            picked.append(line)
        if len(picked) >= max_lines:
            break
    if not picked:
        # Fall back to the first non-empty lines.
        picked = [ln.strip() for ln in text.splitlines() if ln.strip()][:max_lines]
    return " | ".join(picked)[:300]


class ObservationHinter:
    """Append hints using the workspace path and caller-owned turn counters.

    set_progress supplies the counters used to detect test reruns without edits."""

    def __init__(self, workspace_path: str) -> None:
        self.workspace_path = workspace_path
        self._pdb_last_frame: Optional[Tuple[str, int, str]] = None
        self._pdb_recent_lines: List[Tuple[str, int]] = []
        self._turn_last_pytest = 0
        self._turn_last_edit = 0

    def reset(self) -> None:
        """Clear per-session pdb state. Call between sessions."""
        self._pdb_last_frame = None
        self._pdb_recent_lines = []

    def set_progress(self, turn_last_pytest: int, turn_last_edit: int) -> None:
        """Record when the caller last saw a pytest run and a successful edit."""
        self._turn_last_pytest = int(turn_last_pytest or 0)
        self._turn_last_edit = int(turn_last_edit or 0)

    # ── the composed entry point ───────────────────────────────────

    def format_observation(
        self,
        tool_name: str,
        result: str,
        tool_args: Optional[Dict[str, Any]] = None,
        hints_enabled: bool = True,
        wrap: bool = True,
        max_chars: int = MAX_OBSERVATION_CHARS,
    ) -> str:
        """Track pdb state, apply the hints, and cap the result.

        Frame tracking runs even on an unhinted call, so later hints keep
        the current frame.

        ``wrap`` prepends the env's ``Tool: {name}\\nResult: {...}`` header. The
        CLI and eval harness carry the tool name in the native tool message
        instead, so they pass ``wrap=False`` and the cap then applies to the
        content alone.
        """
        self.track_frame(tool_name, result)
        if hints_enabled:
            improved = self.improve_error_message(tool_name, result, tool_args)
        else:
            improved = result
        formatted = f"Tool: {tool_name}\nResult: {improved}" if wrap else improved
        return truncate_observation(formatted, max_chars)

    # ── pdb state tracking ─────────────────────────────────────────

    def track_frame(self, tool_name: Optional[str], result: str) -> None:
        """Cache the most recent pdb stop frame from a debugger observation.

        Any debugger-tool result that prints a stop-frame header
        (``> file.py(18)__init__()``) updates the cached frame; the LAST
        header in the result wins (it is the current position after the op).
        Consumed by the paused-before-assignment AttributeError hint, which
        needs to know whether the session is currently stopped inside
        ``__init__``.
        """
        if not result:
            return
        if tool_name not in INSPECTION_DEBUG_TOOLS and tool_name != "start_debugger":
            return
        matches = PDB_FRAME_RE.findall(result)
        if not matches:
            return
        file_part, line_str, func = matches[-1]
        try:
            line_no = int(line_str)
        except ValueError:
            return
        self._pdb_last_frame = (file_part.rsplit("/", 1)[-1], line_no, func)

    def revisit_hint(self, tool_name: Optional[str], result: str) -> str:
        """Revisit nudges are disabled. Always returns an empty string."""
        return ""

    def _attribute_timing_hint(
        self,
        tool_name: Optional[str],
        tool_args: Optional[Dict[str, Any]],
        attr_name: Optional[str],
    ) -> Optional[str]:
        """Timing-aware hint for an AttributeError on ``self.<attr>`` in pdb.

        The generic "read the class definition" hint creates a contradiction:
        the model reads the class, sees the assignment right there, and
        invents a descriptor theory to explain why the object lacks it. The
        actual cause is pdb's stop-before-execute semantics. When the session
        stops at or before the assignment line in ``__init__``, the attribute
        does not exist yet. This hint says that.
        """
        if tool_name not in INSPECTION_DEBUG_TOOLS:
            return None
        args = tool_args if isinstance(tool_args, dict) else {}
        expr = ""
        # print_variable accepts several aliases (see the pdb harness);
        # run_expression uses `expression`.
        for key in ("expression", "variable", "name", "variable_name", "var_name"):
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                expr = val.strip()
                break
        if "self." not in expr:
            return None
        attr_ref = f"'{attr_name}'" if attr_name else "that attribute"
        frame = self._pdb_last_frame
        if frame and frame[2] == "__init__":
            return (
                f"Hint: stopped inside __init__ ({frame[0]}:{frame[1]}); pdb pauses "
                "BEFORE the reported line, so self."
                f"{attr_name or '<attr>'} isn't assigned YET. step_over past the "
                "assignment first."
            )
        return (
            f"Hint: {attr_ref} does not exist on this object here. pdb stops BEFORE "
            "the reported line — if you are in the method that assigns it, "
            "step_over past the assignment; otherwise read the class definition "
            "for its attributes."
        )

    # ── the error-message diagnoses ────────────────────────────────

    def improve_error_message(
        self,
        tool_name: Optional[str],
        result: str,
        tool_args: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Append an actionable diagnosis to a tool result that needs one.

        Returns the result unchanged when nothing matches — most observations
        take that path, and appending noise to a clean result teaches the model
        that hints are noise.
        """
        # Special handling for pytest runs: discourage blind re-runs without edits.
        if tool_name and tool_name.lower() == "bash":
            cmd = ""
            if isinstance(tool_args, dict):
                cmd = (tool_args.get("command") or "").lower()

            # `cd` to a nonexistent absolute directory is the opening move of
            # a wrong-workspace-root loop (the model guesses /root,
            # /usr/src/workspace, ... for many turns). Remind it with the
            # actual workspace path the caller already knows.
            _res_lower = (result or "").lower()
            if (
                cmd
                and re.search(r"(?:^|&&|;)\s*cd\s+/", cmd)
                and "no such file or directory" in _res_lower
                and "cd:" in _res_lower
            ):
                return (
                    f"{result}\n\n"
                    "Reminder: your workspace is "
                    f"{self.workspace_path} — every bash command "
                    "already starts there. Use relative paths; do not `cd` to "
                    "absolute directories you have not listed."
                )

            if cmd and "pytest" in cmd:
                # A pytest COLLECTION error (exit 2: the last edit broke an
                # import or syntax in the code under test) needs its own hint.
                # Name the actual import/syntax problem and tell the model to
                # fix the file. Check it BEFORE pytest_collected_nothing,
                # because a broken import can also print "collected 0 items".
                if pytest_collection_errored(result):
                    return (
                        "COLLECTION ERROR: your last edit likely broke an import "
                        f"or syntax — {first_collection_error_lines(result)}. Fix "
                        f"the file before rerunning.\n\n{result}"
                    )
                # A wrong test path (exit 4) is NOT a test failure. A "tests are
                # still failing -> edit the code" hint would send the model
                # into edit loops when the only problem is the path.
                if pytest_collected_nothing(result):
                    # Blaming only the filename reinforces filename-guessing
                    # loops when the real problem is a `cd` to a directory
                    # without the test files (e.g. `cd /workspace`).
                    return (
                        f"{result}\n\n"
                        "Hint: pytest ran 0 tests — the test path doesn't exist "
                        "from your CURRENT directory. Your workspace is "
                        f"{self.workspace_path} and every bash command "
                        "starts there; rerun pytest with the test file's relative "
                        "path (no `cd` needed). Path problem, not a test failure."
                    )
            # Empty sed reads: the model chases stale line numbers after its
            # own write_file shrank the file, and `sed -n '71p'` past EOF
            # silently prints nothing (exit 0). Say why.
            if (
                cmd
                and re.search(r"\bsed\b", cmd)
                and "no output" in (result or "").lower()
            ):
                return (
                    f"{result}\n\n"
                    "Hint: sed printed nothing — the line range is likely past "
                    "end of file (line numbers shift after edits). Check length "
                    "with `wc -l <file>` and re-read in range."
                )

        if not result or "error" not in result.lower():
            return result

        tool_args = tool_args or {}

        # FileNotFoundError - suggest listing files / metadata checks
        if (
            "no such file or directory" in result.lower()
            or "filenotfounderror" in result.lower()
        ):
            # Try to extract filename from error
            filename_match = re.search(r"'([^']+)'|\"([^\"]+)\"|:\s+(\S+):", result)
            if filename_match:
                filename = (
                    filename_match.group(1)
                    or filename_match.group(2)
                    or filename_match.group(3)
                )
                hint = (
                    f"{result}\n\nHint: File '{filename}' not found. Run `ls` or "
                    f"`find . -maxdepth 1` to list files, then `wc -l -c {filename}` to confirm the path "
                    "before calling edit_file."
                )
                return hint
            return f"{result}\n\nHint: File not found. Use `ls` and `wc -l -c <file>` to verify the exact path before editing."

        if (
            tool_name in {"edit_file", "write_file"}
            and "string not found" in result.lower()
        ):
            # The file interface always inlines the numbered file or
            # closest-match region that the model can copy old_string from
            # verbatim. A second scan here could serve stale line numbers.
            # Searching for the model's own invented string finds nothing.
            # Add nothing.
            return result

        # AttributeError — timing-aware for pdb inspections of self.<attr>
        # (paused-before-assignment), generic class-definition pointer
        # otherwise.
        if "attributeerror" in result.lower() or "has no attribute" in result.lower():
            attr_match = re.search(
                r"has no attribute '([^']+)'", result, re.IGNORECASE
            )
            attr_name = attr_match.group(1) if attr_match else None
            timing_hint = self._attribute_timing_hint(
                tool_name, tool_args, attr_name
            )
            if timing_hint:
                return f"{result}\n\n{timing_hint}"
            if attr_name:
                return f"{result}\n\nHint: Method/attribute '{attr_name}' doesn't exist. Read the class definition to see available methods."
            return f"{result}\n\nHint: Method or attribute doesn't exist. Read the class/object definition to check available attributes."

        # SyntaxError - point to the location
        if "syntaxerror" in result.lower():
            # The heredoc/EOF hint only fits a bash command that used a
            # heredoc. On other SyntaxErrors (such as pdb run_expression
            # output) it would mislead.
            bash_cmd = (
                (tool_args.get("command") or "")
                if isinstance(tool_args, dict)
                else ""
            )
            if tool_name and tool_name.lower() == "bash" and "<<" in bash_cmd:
                return (
                    f"{result}\n\nHint: Check your code syntax. If using cat > "
                    "with a heredoc, ensure 'EOF' markers are correct and Python "
                    "indentation is valid (prefer edit_file/write_file over heredoc)."
                )
            return f"{result}\n\nHint: Check the code syntax near the reported line."

        # Timeout: suggest a non-interactive approach. Match only the
        # harness's own timeout messages, not the bare word "timeout". A
        # successful sed dump of source code that merely contains "timeout"
        # (retry/backoff modules) would get a false "Command timed out" hint,
        # which teaches the model that hints are noise.
        lowered_result = result.lower()
        if (
            "tool execution timed out" in lowered_result
            or "read timeout after" in lowered_result
            or "test execution timed out" in lowered_result
            or "command timed out" in lowered_result
        ):
            if "pdb" in lowered_result or "interactive" in lowered_result:
                return f"{result}\n\nHint: Interactive debugger commands timeout. Use non-interactive debugging: read code with sed, add print statements, or use debugger tools."
            return f"{result}\n\nHint: Command timed out (30s limit). Try simpler commands or read files in smaller chunks."

        return result


def truncate_observation(text: str, max_chars: int = MAX_OBSERVATION_CHARS) -> str:
    """Cap one observation, keeping a 2/3 head and a 1/3 tail.

    Keeping the head preserves the command echo and the start of a traceback;
    keeping the tail preserves the pytest summary line, which is where the
    failure count lives. Every caller must use this same split — the policy has
    never seen a truncated observation of any other shape.
    """
    if len(text) <= max_chars:
        return text
    keep = max_chars - 80
    head_len = keep * 2 // 3
    tail_len = keep - head_len
    omitted = len(text) - head_len - tail_len
    return (
        text[:head_len]
        + f"\n\n[... {omitted} chars omitted ...]\n\n"
        + text[-tail_len:]
    )
