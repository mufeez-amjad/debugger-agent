"""Control pdb sessions through a pexpect terminal and expose them as tool calls."""

import hashlib
import logging
import os
import re
import shlex
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pexpect

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.models import ToolCall, TOOLS_DESCRIPTIONS

logger = logging.getLogger(__name__)

# Appended by PdbSession._clean_pdb_output when pdb reruns a finished test
# program in-place; PdbHarness strips it from the "test result" tail so the
# missed-breakpoint diagnostic doesn't repeat it.
RESTARTED_IN_PLACE_NOTE = (
    "The program finished and was RESTARTED from the beginning; your "
    "breakpoints are still set. continue_execution runs the test again."
)


def _numbered_source_window(
    file_path: str, line_no: int, marker: str, context_lines: int = 3
) -> str:
    """Numbered source window around ``line_no`` with ``marker`` on that line.
    Empty string when the file is unreadable or the line is out of range.
    Shared by PdbSession (frame-return messages, where pdb hands us an
    absolute path) and PdbHarness._source_window (which resolves paths
    first)."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return ""
    if not 1 <= line_no <= len(lines):
        return ""
    lo = max(1, line_no - context_lines)
    hi = min(len(lines), line_no + context_lines)
    pad = " " * len(marker)
    rows = []
    for i in range(lo, hi + 1):
        m = marker if i == line_no else pad
        rows.append(f"{m} {i:>4} {lines[i - 1].rstrip()}")
    return "Code context ({}):\n{}".format(
        os.path.basename(file_path), "\n".join(rows)
    )


class PdbSession:
    """A pdb process and its current frame, breakpoints, and test selection."""

    def __init__(self, existing_file_path: str, test_selection: Optional[str] = None):
        self.existing_file_path = existing_file_path
        self.filename = os.path.basename(existing_file_path)
        # Select the failing test so a breakpoint does not stop in an earlier passing test.
        self.test_selection = (test_selection or "").strip() or None
        # Node ids the preflight collected, for the start message and for
        # resolving the selection. Empty until start() runs, and only ever
        # populated for a test file.
        self._collected: List[str] = []
        # "node" (exact node id) or "k" (substring/expression), decided by
        # _resolve_selection against the collected list.
        self._selection_mode = "k"
        self.child = None  # pexpect.spawn object
        self.execution_state = "not_started"
        self.current_line = 1
        self.current_position = {"file": None, "line": None}  # Updated after every PDB command
        self.breakpoints = []  # List of (file, line) tuples — supports cross-file breakpoints
        self.breakpoint_numbers = {}  # Maps breakpoint_number -> (file, line)
        # How many times this session has been auto-restarted after the program
        # ran to completion without stopping at a breakpoint. Bounded so a model
        # that keeps `continue`-ing past a never-hit breakpoint can't loop the
        # harness forever (see PdbHarness._recover_after_missed_breakpoint).
        self._auto_restart_count = 0
        # Propagate the active pdb trace function into worker threads so
        # breakpoints inside ThreadPoolExecutor / threading.Thread targets are
        # honored during the pytest run (default on; empirically verified to
        # make worker-thread breakpoints fire — see _write_thread_trace_plugin).
        # Set LLM_DEBUGGER_THREAD_TRACE=0 to disable.
        self.thread_tracing_enabled = os.environ.get(
            "LLM_DEBUGGER_THREAD_TRACE", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        # Temp dir holding the generated pytest thread-trace plugin, if any.
        self._trace_plugin_dir: Optional[str] = None

    def __del__(self):
        """Cleanup: close the pexpect session if it's still running."""
        if self.child and self.child.isalive():
            try:
                self.child.close(force=True)
            except:
                pass  # Ignore errors during cleanup
        self._cleanup_trace_plugin()

    # Source of the auto-generated pytest plugin that forwards the active pdb
    # trace function into worker threads. Patching Thread.start (which
    # ThreadPoolExecutor also uses) at the moment .start() is called on the main
    # thread captures pdb's live trace_dispatch and installs it in the worker,
    # so breakpoints inside thread targets fire. No-op when no trace is active.
    _THREAD_TRACE_PLUGIN_SRC = (
        '"""Auto-generated: forward the active pdb trace into worker threads."""\n'
        "import sys\n"
        "import threading\n"
        "\n"
        "_orig_start = threading.Thread.start\n"
        "\n"
        "\n"
        "def _start_with_trace(self):\n"
        "    tr = sys.gettrace()\n"
        "    if tr is not None:\n"
        "        _run = self.run\n"
        "\n"
        "        def _run_traced(*args, **kwargs):\n"
        "            sys.settrace(tr)\n"
        "            return _run(*args, **kwargs)\n"
        "\n"
        "        self.run = _run_traced\n"
        "    return _orig_start(self)\n"
        "\n"
        "\n"
        'if not getattr(threading.Thread.start, "_pdb_thread_trace", False):\n'
        "    _start_with_trace._pdb_thread_trace = True\n"
        "    threading.Thread.start = _start_with_trace\n"
    )

    def _cleanup_trace_plugin(self) -> None:
        """Remove the temp dir holding the generated thread-trace plugin."""
        d = getattr(self, "_trace_plugin_dir", None)
        if not d:
            return
        try:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
        self._trace_plugin_dir = None

    def _write_thread_trace_plugin(self) -> Tuple[Optional[str], Optional[str]]:
        """Write the worker-thread trace-propagation pytest plugin to a temp dir.

        Returns ``(dir_on_pythonpath, module_name)`` or ``(None, None)`` on
        failure so the caller can fall back to running pytest without it.
        """
        # Drop any plugin dir from a start before the restart.
        self._cleanup_trace_plugin()
        try:
            import tempfile
            d = tempfile.mkdtemp(prefix="pdb_thread_trace_")
            mod = "_pdb_thread_trace_plugin"
            with open(os.path.join(d, mod + ".py"), "w", encoding="utf-8") as fh:
                fh.write(self._THREAD_TRACE_PLUGIN_SRC)
            self._trace_plugin_dir = d
            return d, mod
        except Exception:
            return None, None

    def is_alive(self) -> bool:
        """Check if the PDB session is still running."""
        return self.child is not None and self.child.isalive()

    def _preflight_collect(self, env) -> Optional[str]:
        """Return an honest start error when pytest cannot collect the test
        file (syntax error, import error, no tests), or None when collection
        succeeds. Best-effort: any preflight hiccup defers to the real session.

        Always collects the WHOLE file, never the selection: the test list is
        what lets `_resolve_selection` turn a partial name into a match and name
        the real tests when it cannot.
        """
        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest", "--collect-only", "-q",
                    "--no-header", self.existing_file_path,
                ],
                cwd=os.path.dirname(self.existing_file_path) or ".",
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception:
            return None
        # pytest exit codes: 0 collected; 2 interrupted / collection error;
        # 4 usage error; 5 no tests collected.
        if proc.returncode in (0, 1):
            self._collected = [
                ln.strip().split("::", 1)[1]
                for ln in (proc.stdout or "").splitlines()
                if "::" in ln and not ln.startswith(" ")
            ]
            return self._resolve_selection()
        detail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if len(detail) > 800:
            detail = detail[:800].rstrip() + " …[truncated]"
        msg = (
            "Error: Failed to start debugging session — pytest could not "
            "collect the test file (collection/syntax error, or no tests "
            "collected)."
        )
        if detail:
            msg += f"\n{detail}"
        return msg

    def _resolve_selection(self) -> Optional[str]:
        """Decide how to pass the requested test to pytest, or explain why not.

        Resolved against the collected test list rather than guessed from the
        string's shape, because a name that merely LOOKS like a node id is not
        one unless the file actually contains it. A partial name ("failing",
        "lopsided") is the natural thing to reach for and would otherwise
        hard-fail as an exact node id that does not exist, so it resolves to a
        `-k` substring match instead. When nothing matches, name the tests that
        do exist — the model cannot guess them and would otherwise burn turns.
        """
        if not self.test_selection:
            return None
        sel = self.test_selection
        if sel in self._collected:
            self._selection_mode = "node"
            return None
        if self._NODE_ID_RE.match(sel):
            if any(sel in name for name in self._collected):
                self._selection_mode = "k"
                return None
            available = ", ".join(self._collected[:12]) or "(none)"
            more = "" if len(self._collected) <= 12 else f" (+{len(self._collected) - 12} more)"
            return (
                f"Error: no test matching '{sel}' in {self.filename}. "
                f"Tests in this file: {available}{more}."
            )
        # An expression (spaces, and/or, globs): pytest owns the matching, and a
        # miss surfaces as its own "no tests ran" on the real run.
        self._selection_mode = "k"
        return None

    def _is_test_file(self) -> bool:
        """Check if the target file is a pytest test file."""
        name = os.path.basename(self.existing_file_path)
        return name.startswith("test_") or name.endswith("_test.py")

    # A bare node id: `test_name`, `TestClass::test_method`, `test_x[param]`.
    # Anything else (spaces, `or`/`and`, globs) is a -k expression.
    _NODE_ID_RE = re.compile(r"^[A-Za-z_][\w\[\]\-.:]*$")

    def _selection_note(self) -> str:
        """The sentence in the start message saying what is about to run.

        When the caller did not select, and the file holds more than one test,
        say so and name the parameter. A breakpoint fires in the FIRST test that
        reaches it, so on a file whose failing test is not first the model
        otherwise stops in a passing test with nothing in the observation to
        say so — it reads correct behaviour and concludes the code is fine.
        """
        if self.test_selection:
            return f", running only `{self.test_selection}`."
        count = len(self._collected)
        if count <= 1:
            return "."
        return (
            f", running all {count} tests in it, in file order. A breakpoint "
            f"stops in the FIRST test that reaches it — pass "
            f'test="<test_name>" to debug one test instead.'
        )

    def _pytest_target_args(self) -> List[str]:
        """The pytest argv selecting what to run: the file, or one test in it.

        ``_resolve_selection`` has already decided between the two, so this only
        renders it. An exact hit becomes a node id (`file.py::test_name`)
        because that cannot match anything else; everything else goes to `-k`.
        """
        if not self.test_selection:
            return [self.existing_file_path]
        if self._selection_mode == "node":
            return [f"{self.existing_file_path}::{self.test_selection}"]
        return [self.existing_file_path, "-k", self.test_selection]

    def start(self) -> str:
        """Start the real PDB debugging session using pexpect."""
        try:
            if not os.path.exists(self.existing_file_path):
                return f"Error: Target file {self.existing_file_path} does not exist"

            env = os.environ.copy()

            # For test files, run pytest under an OUTER pdb (`python -m pdb -m
            # pytest`): the first prompt appears BEFORE pytest executes at all,
            # so breakpoints set (or re-applied) at that prompt fire anywhere —
            # module import, @pytest.fixture setup, and the test body.
            # `pytest --trace` only wraps the test CALL phase, so code run by
            # fixtures would run before the first prompt existed, and
            # breakpoints there could never fire. Also, when the program ends,
            # pdb reruns it in-process instead of exiting, so breakpoints
            # survive without a respawn. Regular files use plain pdb.
            if self._is_test_file():
                # Fail fast when pytest cannot even collect the test file.
                # The pre-run prompt appears BEFORE pytest executes, so a
                # syntax/import/collection error would otherwise surface only
                # on the first continue — after the model already placed
                # breakpoints. Checking at start is cheap.
                collect_err = self._preflight_collect(env)
                if collect_err:
                    self.execution_state = "not_started"
                    return collect_err
                plugin_arg = ""
                if self.thread_tracing_enabled:
                    plugin_dir, plugin_mod = self._write_thread_trace_plugin()
                    if plugin_dir:
                        env["PYTHONPATH"] = (
                            plugin_dir + os.pathsep + env.get("PYTHONPATH", "")
                        )
                        plugin_arg = f" -p {plugin_mod}"
                target = " ".join(shlex.quote(a) for a in self._pytest_target_args())
                cmd = (
                    f"{sys.executable} -u -m pdb -m pytest{plugin_arg}"
                    f" {target}"
                    f" -x -s --tb=short --no-header -q"
                )
            else:
                cmd = f"{sys.executable} -u -m pdb {self.existing_file_path}"

            # echo=False: without it every command is echoed back into the
            # output the model sees ("pp locals()" as the first result line),
            # and readline's redisplay of a >80-col command produces garbled
            # horizontal-scroll artifacts ("<dited, ...").
            # Wide pty dimensions are a second layer of defense for the same
            # wrap artifacts. send_command() also strips any echo that leaks
            # through textually.
            self.child = pexpect.spawn(
                cmd,
                encoding='utf-8',
                codec_errors='replace',
                cwd=os.path.dirname(self.existing_file_path) or ".",
                timeout=10,
                env=env,
                echo=False,
                dimensions=(48, 2000),
            )

            # Wait for initial PDB prompt
            initial_output = self._wait_for_prompt()

            # If the child exited before we ever reached an interactive prompt,
            # the session did NOT start (collection error, syntax error, or no
            # tests were collected). Return an honest error rather than a message
            # that claims the session started and then reports failure.
            if not self.child.isalive():
                self.execution_state = "not_started"
                detail = (initial_output or "").strip()
                msg = (
                    "Error: Failed to start debugging session — the process "
                    "exited before the debugger prompt (collection/syntax error, "
                    "or no tests collected)."
                )
                if detail and detail != "Process terminated":
                    msg += f"\n{detail}"
                return msg

            self.execution_state = "started"

            if self._is_test_file():
                # The raw initial output is pdb's position inside pytest's
                # bootstrap — noise that misleads the model. State the actual
                # contract of the pre-run prompt.
                scope = self._selection_note()
                return (
                    f"Debugging session started for {self.filename}{scope} The "
                    f"program is paused BEFORE the test run — no test or library "
                    f"code has executed yet. Set breakpoints with add_breakpoint "
                    f"(they can stop anywhere: library code, pytest fixtures, and "
                    f"module import), then call continue_execution to run the test."
                )

            # Return startup message with initial output if available
            if initial_output and initial_output != "PDB ready":
                return f"Debugging session started for {self.filename}\n{initial_output}"
            else:
                return f"Debugging session started for {self.filename}"

        except pexpect.exceptions.EOF as e:
            return f"Error: PDB process ended unexpectedly: {str(e)}"
        except pexpect.exceptions.TIMEOUT as e:
            return f"Error: Timeout waiting for PDB to start: {str(e)}"
        except Exception as e:
            return f"Error starting debugging session: {str(e)}"

    def restart(self, reapply_breakpoints: bool = True) -> Dict[str, Any]:
        """Re-spawn the pdb/pytest process after the child died.

        Under `-m pdb -m pytest` the child normally SURVIVES the end of a test
        run (pdb reruns the target in-process with breakpoints intact), so this
        is the safety net for a genuinely dead child — a crash, a `q`, or an
        edit that requires reloading code. Without it, every subsequent tool
        call returns the "no active debugging session" error and the session is
        effectively dead even though the model may still want to (re)place a
        breakpoint and inspect. We tear down the dead child, re-spawn the same
        command, and re-apply the breakpoints we were tracking so the model
        lands back in a usable state at the pre-run prompt.

        Returns a dict describing what happened (alive flag + which breakpoints
        were successfully re-applied) so callers can build an accurate message.
        """
        saved_breakpoints = list(self.breakpoints) if reapply_breakpoints else []

        # Tear down the dead child if anything is left.
        try:
            if self.child and self.child.isalive():
                self.child.close(force=True)
        except Exception:
            pass
        self.child = None
        self.breakpoints = []
        self.breakpoint_numbers = {}
        self.current_position = {"file": None, "line": None}

        start_output = self.start()
        if not self.is_alive():
            return {"alive": False, "reapplied": [], "start_output": start_output}

        reapplied: List = []
        for bp in saved_breakpoints:
            try:
                file_name, line_num = bp
            except (TypeError, ValueError):
                continue
            if file_name and file_name != self.filename:
                cmd = f"b {file_name}:{line_num}"
            else:
                # Qualify with the session file's path: at the pre-run prompt
                # the current frame is pytest's bootstrap, so a bare
                # `b {line}` would land in the wrong file.
                cmd = f"b {os.path.abspath(self.existing_file_path)}:{line_num}"
            resp = self.send_command(cmd)
            match = re.search(r'Breakpoint (\d+) at', resp or "")
            if match:
                bp_entry = (file_name, line_num)
                if bp_entry not in self.breakpoints:
                    self.breakpoints.append(bp_entry)
                self.breakpoint_numbers[int(match.group(1))] = bp_entry
                reapplied.append(bp_entry)
        return {"alive": True, "reapplied": reapplied, "start_output": start_output}

    def line_is_inside_function(
        self, line_num: int, file_path: Optional[str] = None
    ) -> Optional[bool]:
        """Return True if ``line_num`` lies inside a function/method body.

        Returns False if the line is module-level or sits directly in a class
        body, and None when the file can't be parsed (so callers can stay
        silent rather than emit a misleading warning).

        Why this matters: module-level / class-body lines execute once at
        *import* time (during test collection), so a breakpoint there stops in
        import machinery rather than in the test's call into the code — almost
        never what the model wants. Useful breakpoints sit on executable lines
        *inside* the function the test calls (``__init__`` body, the method
        under test). This lets ``add_breakpoint`` warn the
        model at placement time.
        """
        target = file_path or self.existing_file_path
        try:
            import ast
            with open(target, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
        except Exception:
            return None

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if start is None or end is None:
                    continue
                # Ignore the def/decorator header lines themselves; a breakpoint
                # on a `def` line behaves like the body for our purposes, but the
                # body is what actually executes per-call.
                if start <= line_num <= end:
                    return True
        return False

    def suggest_executable_line(
        self, line_num: int, file_path: Optional[str] = None
    ) -> Optional[Tuple[str, int]]:
        """Suggest a concrete executable line to break on near ``line_num``.

        When a model breaks on a module/class-body line, "set it inside the
        function" is not actionable on its own. This returns ``(func_name,
        first_body_line)`` for the nearest function — the one whose body
        contains ``line_num`` if any, else the closest function starting at or
        after ``line_num``, else the last function before it — so the warning
        can name an exact line. Returns None when the file can't be parsed or
        has no functions. First body line skips the def/decorator header and
        any leading docstring.
        """
        target = file_path or self.existing_file_path
        try:
            import ast

            with open(target, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
        except Exception:
            return None

        funcs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if start is None or end is None or not node.body:
                    continue
                body0 = node.body[0]
                first = getattr(body0, "lineno", start)
                # Skip a leading string-literal docstring.
                if (
                    isinstance(body0, ast.Expr)
                    and isinstance(getattr(body0, "value", None), ast.Constant)
                    and isinstance(body0.value.value, str)
                    and len(node.body) > 1
                ):
                    first = getattr(node.body[1], "lineno", first)
                funcs.append((node.name, start, end, first))

        if not funcs:
            return None

        # Prefer the enclosing function, then the nearest one starting at/after,
        # then the last one before.
        enclosing = [f for f in funcs if f[1] <= line_num <= f[2]]
        if enclosing:
            f = min(enclosing, key=lambda x: x[2] - x[1])
            return f[0], f[3]
        after = sorted([f for f in funcs if f[1] >= line_num], key=lambda x: x[1])
        if after:
            return after[0][0], after[0][3]
        before = sorted([f for f in funcs if f[1] < line_num], key=lambda x: -x[1])
        return (before[0][0], before[0][3]) if before else None

    def resolve_breakpoint_line(
        self, line_num: int, file_path: Optional[str] = None
    ) -> Optional[int]:
        """Snap a requested breakpoint line to the nearest line PDB will actually
        stop on during the test run.

        Returns the line to break on, which may equal ``line_num`` (no change
        needed) or a nearby line, and ``None`` only when the file can't be parsed
        so the caller keeps the requested line and the raw PDB behaviour.

        The valid targets are statement lines that live strictly *inside* a
        function body — never a blank/comment line (which PDB rejects), a
        ``def``/decorator header, or a module/class-body line (which runs once
        at import during collection, so a breakpoint there stops in import
        machinery instead of the test's call). When the requested line isn't already such a
        line we snap: to the nearest statement inside the enclosing function if
        the line falls within one, otherwise to the first statement of the
        nearest function (matching :meth:`suggest_executable_line`).
        """
        target = file_path or self.existing_file_path
        try:
            import ast
            with open(target, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
        except Exception:
            return None

        func_ranges = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if start is not None and end is not None:
                    func_ranges.append((start, end))
        if not func_ranges:
            return line_num

        # Statement lines inside some function body (strictly after its def
        # header). Skip docstring string-literal statements — PDB rejects a
        # breakpoint on a triple-quoted line just like a comment.
        valid = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.stmt):
                continue
            ln = getattr(node, "lineno", None)
            if ln is None:
                continue
            if (
                isinstance(node, ast.Expr)
                and isinstance(getattr(node, "value", None), ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if any(s < ln <= e for (s, e) in func_ranges):
                valid.add(ln)
        if not valid or line_num in valid:
            return line_num

        # If the requested line falls inside a function, snap within that
        # function (innermost), preferring the next statement at/after it.
        enclosing = [(s, e) for (s, e) in func_ranges if s < line_num <= e]
        if enclosing:
            s, e = min(enclosing, key=lambda x: x[1] - x[0])
            cands = sorted(l for l in valid if s < l <= e)
            if cands:
                forward = [l for l in cands if l >= line_num]
                return forward[0] if forward else cands[-1]
            return line_num

        # Module/class-body line: snap to the first statement of the nearest
        # function starting at/after it, else the last function before it.
        after = sorted([(s, e) for (s, e) in func_ranges if s >= line_num])
        before = sorted(
            [(s, e) for (s, e) in func_ranges if s < line_num], key=lambda x: -x[0]
        )
        for s, e in (after or before):
            cands = sorted(l for l in valid if s < l <= e)
            if cands:
                return cands[0]
        return line_num

    def file_line_count(self, file_path: Optional[str] = None) -> Optional[int]:
        """Number of lines in ``file_path`` (defaults to the session's target
        file), or None when the file can't be read."""
        target = file_path or self.existing_file_path
        try:
            with open(target, "r", encoding="utf-8", errors="replace") as handle:
                return sum(1 for _ in handle)
        except OSError:
            return None

    def function_name_for_line(
        self, line_num: int, file_path: Optional[str] = None
    ) -> Optional[str]:
        """Name of the innermost function whose body contains ``line_num``, or
        None for a module/class-body line or when the file can't be parsed.

        Used to decide whether a breakpoint snap crossed a function boundary
        (a loud, semantics-changing move) versus a small in-function nudge.
        """
        target = file_path or self.existing_file_path
        try:
            import ast
            with open(target, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
        except Exception:
            return None
        best = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if start is None or end is None:
                    continue
                if start <= line_num <= end and (best is None or start > best[0]):
                    best = (start, node.name)
        return best[1] if best else None

    def _update_position_from_output(self, raw_output: str) -> None:
        """Extract current execution position from raw PDB output and update current_position.

        PDB emits lines like: > /path/to/file.py(42)<module>()
        This is emitted on start, continue, step_over, step_into, and program restart.
        """
        if not raw_output:
            return
        # Match the PDB position pattern: > /path/file.py(LINE)FUNC()
        # Also capture the frame's function name: comprehension frames
        # (<genexpr>/<listcomp>/...) need dedicated NameError guidance.
        match = re.search(r'> (.+?)\((\d+)\)(<?[\w<>]*>?)', raw_output)
        if match:
            file_path = match.group(1)
            line_num = int(match.group(2))
            frame_func = (match.group(3) or "").strip() or None
            self.current_position = {
                "file": file_path,
                "line": line_num,
                "function": frame_func,
            }
            self.current_line = line_num

    def _wait_for_prompt(self, timeout: float = 10.0) -> str:
        """
        Wait for PDB prompt using pexpect - robust and reliable.

        Args:
            timeout: Maximum time to wait for prompt in seconds

        Returns:
            Output before the prompt, cleaned and filtered
        """
        try:
            # Wait for either the PDB prompt or EOF
            index = self.child.expect_exact(['(Pdb) ', pexpect.EOF], timeout=timeout)

            if index == 0:  # Normal prompt
                # self.child.before contains everything before '(Pdb) '
                output = self.child.before or ""
                # Update position tracking from raw output before cleaning
                self._update_position_from_output(output)
                return self._clean_pdb_output(output)

            elif index == 1:  # EOF - program ended
                output = self.child.before or ""
                if output:
                    self._update_position_from_output(output)
                    return self._clean_pdb_output(output)
                return "Process terminated"

        except pexpect.TIMEOUT:
            # Try to get partial output
            partial = self.child.before if hasattr(self.child, 'before') else ""
            if partial:
                return f"Timeout waiting for PDB response. Partial output:\n{partial}"
            return "Timeout waiting for PDB response"

        except pexpect.EOF:
            return "Debugger session ended unexpectedly"

    def send_command(self, command: str, timeout: float = 10.0) -> str:
        """Send a command to PDB and return the response using pexpect."""
        if not self.child or not self.child.isalive():
            return (
                "Error: No active debugging session (program finished or session "
                "ended). Call start_debugger on the TEST file, then add_breakpoint "
                "before continue_execution."
            )

        try:
            # Send command to PDB
            self.child.sendline(command)

            # Wait for response
            return self._strip_command_echo(command, self._wait_for_prompt(timeout))

        except pexpect.exceptions.EOF:
            return "Debugger session ended unexpectedly"
        except pexpect.exceptions.TIMEOUT:
            return f"Timeout waiting for response to command: {command}"
        except Exception as e:
            return f"Error sending command '{command}': {str(e)}"
    
    @staticmethod
    def _strip_command_echo(command: str, output: str) -> str:
        """Remove a leaked terminal echo of ``command`` from the head of
        ``output``.

        The pty is spawned with ``echo=False``, so normally nothing is echoed —
        this is the textual fallback for environments where the echo (or
        readline's redisplay of it) still leaks through. Two shapes are
        handled: the plain echoed command as the first line, and readline's
        horizontal-scroll artifact for an over-width command (``<tail-of-cmd``,
        such as "<dited, ..."). Only the FIRST line is ever
        considered, so genuine output can't be eaten.
        """
        if not output or not command:
            return output
        lines = output.split("\n")
        first = lines[0].strip().rstrip("\r")
        cmd = command.strip()
        is_echo = first == cmd or (
            first.startswith("<") and len(first) > 4 and first[1:] in cmd
        )
        if is_echo:
            return "\n".join(lines[1:]).lstrip("\n")
        return output

    def _clean_pdb_output(self, output: str) -> str:
        """Clean PDB output by removing prompt artifacts that confuse the LLM."""
        # Strip ANSI escape / color codes (e.g. pytest output surfaced via `c`)
        # so the model gets plain text instead of `\x1b[31mF\x1b[0m` noise.
        output = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', output)
        # Remove trailing prompt characters
        output = output.rstrip('(Pdb) >\n ')
        # Remove any remaining standalone prompt markers
        output = output.replace('\n(Pdb) ', '\n')
        output = output.replace('(Pdb) ', '')
        # Remove standalone > prompt markers (from manual command mode)
        lines = output.split('\n')
        cleaned_lines = [line for line in lines if line.strip() != '>']
        cleaned_output = '\n'.join(cleaned_lines).strip()

        # Normalize EOF and program restart messages to be clearer.
        # NOTE: `--Return--` means we stepped OUT of a frame (a function
        # returned) — NOT the end of the program. Only genuine end-of-program
        # markers map to the completion message; a frame return reports where we
        # returned to so the model doesn't think execution ended and stop early.
        if "End of file" in cleaned_output and "Breakpoint" not in cleaned_output:
            return "Program execution completed. Use step_over to restart from beginning."

        if "--Return--" in cleaned_output and "Breakpoint" not in cleaned_output:
            return self._describe_frame_return(cleaned_output)

        # pdb's post-run banners: under `-m pdb` the child does NOT exit when
        # the program ends — pdb prints a banner, reruns the target in-process,
        # and prompts again at the first line with all breakpoints intact.
        # pytest always ends via SystemExit, so its banner is the sys.exit()
        # one. The program's own output (the test result) arrives in the same
        # read — keep it and normalize only the banner, instead of swallowing
        # the whole thing.
        if (
            "The program finished and will be restarted" in cleaned_output
            or "The program exited via sys.exit()" in cleaned_output
        ):
            drop_prefixes = (
                "The program finished and will be restarted",
                "The program exited via sys.exit().",
                "Restarting ",
            )
            kept_lines = []
            for line in cleaned_output.split("\n"):
                stripped = line.strip()
                if any(stripped.startswith(p) for p in drop_prefixes):
                    continue
                # pdb's post-restart position marker + pending-line echo.
                if re.match(r'> .+\(\d+\)', stripped) or stripped.startswith("-> "):
                    continue
                kept_lines.append(line)
            kept = "\n".join(kept_lines).strip()
            if self._is_test_file():
                note = RESTARTED_IN_PLACE_NOTE
            else:
                note = (
                    "Program finished without hitting any breakpoint. Restarted "
                    "at beginning. Hint: if you are debugging a library file, "
                    "start the debugger on the TEST file instead so the function "
                    "is actually called."
                )
            if kept:
                kept = self._filter_verbose_output(kept)
            return f"{kept}\n{note}".strip() if kept else note

        # Apply aggressive filtering to remove verbose Python internals
        return self._filter_verbose_output(cleaned_output)

    def _describe_frame_return(self, cleaned_output: str) -> str:
        """Render a compact message for a pdb ``--Return--`` event (stepping out
        of a frame). pdb prints, e.g.::

            --Return--
            > /path/file.py(10)worker()->42
            -> return x + 1

        We name the function that returned, its location, and the return value
        (when present), plus the pending source line. It does not report this
        as "Program execution completed", which would mislead.
        """
        m = re.search(
            r">\s*(.+?)\((\d+)\)(<?\w+>?)\(\)(?:->(.*))?", cleaned_output
        )
        pending = ""
        for line in cleaned_output.split("\n"):
            if line.strip().startswith("->"):
                pending = line.strip()
                break
        if not m:
            base = (cleaned_output.replace("--Return--", "").strip())
            return base or (
                "Returned from the current function (stepped out of the frame). "
                "Execution has not ended; continue stepping or inspect state."
            )
        path, line_no, func = m.group(1), m.group(2), m.group(3)
        retval = (m.group(4) or "").strip()
        fname = os.path.basename(path)
        msg = (
            f"Returned from {func}() at {fname}:{line_no} (stepped out of the "
            f"frame — execution has NOT ended)."
        )
        if retval:
            msg += f" Return value: {retval}"
        if pending:
            msg += f"\n{pending}"
        # The rewrite above drops pdb's raw `> file(line)` marker, so the
        # generic _with_position_context gate never fires on return events —
        # append the same numbered window here. Without it, the model loses
        # orientation at frame returns.
        context = _numbered_source_window(path, int(line_no), "->")
        if context:
            msg += f"\n{context}"
        return msg

    def _filter_verbose_output(self, output: str) -> str:
        """
        Aggressively filter verbose Python internals from PDB output.

        Removes:
        - All dunder variables (__builtins__, __name__, etc.)
        - Python copyright and license text
        - Module system internals
        - Excessively large variable dumps

        Args:
            output: Raw PDB output string

        Returns:
            Filtered output with only relevant debugging information
        """
        # Handle None/empty
        if output is None:
            return ""
        if not output:
            return output

        # Check if this looks like output that needs filtering
        has_dunders = '__' in output
        has_copyright = 'Copyright' in output or 'Reserved' in output
        is_dict_dump = '{' in output and '}' in output and ':' in output

        # Check for builtin class/function patterns (even in fragments)
        has_builtin_classes = re.search(r"<class '\w+(Error|Warning|Exception|bool|int|str|list|dict|tuple)'>", output)
        has_builtin_functions = '<built-in function' in output
        has_builtin_patterns = has_builtin_classes or has_builtin_functions

        needs_filtering = has_dunders or has_copyright or is_dict_dump or has_builtin_patterns

        # If output is very short and doesn't need filtering, return as-is
        if len(output) < 100 and not needs_filtering:
            return output

        # Max size limits (bytes)
        MAX_OUTPUT_SIZE = 4096  # 4KB max for any output
        MAX_VARIABLE_SIZE = 200  # 200 chars per variable representation

        lines = output.split('\n')
        filtered_lines = []
        current_size = 0
        variables_shown = 0
        MAX_VARIABLES = 30  # Show max 30 variables

        # Patterns to aggressively remove
        remove_patterns = [
            # Copyright and license
            r'Copyright',
            r'All Rights Reserved',
            r'Reserved\.',
            r'Licensed to PSF',
            r'Thanks to.*?python\.org',
            r'Stichting Mathematisch Centrum',

            # Dunder variables and Python internals
            r"'__builtins__':\s*\{.*?\}",  # The entire builtins dict
            r"'__builtins__':\s*<module.*?>",
            r"'__name__':",
            r"'__doc__':",
            r"'__package__':",
            r"'__loader__':",
            r"'__spec__':",
            r"'__annotations__':",
            r"'__cached__':",
            r"'__file__':",
            r"'__dict__':",

            # Common verbose built-in references
            r"<class 'dict'>",
            r"<class 'list'>",
            r"<class 'str'>",
            r"<class 'int'>",
            r"<class 'float'>",
            r"<built-in function.*?>",
            r"<built-in method.*?>",

            # Help and usage text
            r"Type help\(\).*?",
            r"Use exit\(\).*?",
            r"Use quit\(\).*?",

            # Built-in exception classes (key-value pairs from __builtins__)
            r"'\w+Error':\s*<class '\w+Error'>",
            r"'\w+Warning':\s*<class '\w+Warning'>",
            r"'\w+Exception':\s*<class '\w+Exception'>",

            # Built-in types (key-value pairs)
            r"'(bool|int|float|str|list|dict|tuple|set|frozenset|bytes|bytearray)':\s*<class '\1'>",
            r"'(type|object|super|property|staticmethod|classmethod)':\s*<class '\1'>",
            r"'(enumerate|filter|map|zip|reversed|range)':\s*<class '\1'>",
            r"'(slice|memoryview|complex)':\s*<class '\1'>",

            # Built-in functions (key-value pairs)
            r"'(abs|all|any|ascii|bin|callable|chr|compile|delattr|dir|divmod)':\s*<built-in function",
            r"'(eval|exec|format|getattr|globals|hasattr|hash|hex|id|input)':\s*<built-in function",
            r"'(isinstance|issubclass|iter|len|locals|max|min|next|oct|ord)':\s*<built-in function",
            r"'(open|pow|print|repr|round|setattr|sorted|sum|vars)':\s*<built-in function",

            # Built-in helper objects
            r"'(help|credits|license|copyright|quit|exit)':\s*",

            # Built-in constants
            r"'__(debug|import)__':\s*(True|False)",
            r"'(True|False|None|Ellipsis|NotImplemented)':\s*",

            # Module docstrings (often very long)
            r"'__doc__':\s*'.*?identifiers of Python",
            r"This module provides direct access",
            r"identifiers of Python",
            r"built-in function len\(\)",
            r"built-in of that name is also needed",
            r"objects with the same name as a built-in value",
            r"This module is not normally accessed explicitly",

            # Fragment patterns (incomplete lines from verbose output)
            r"^\s*'?\w+Error'>?,?\s*$",  # Fragment like "eError'>,"
            r"^\s*'objects\.",  # Fragment docstring text
        ]

        # Check if this looks like a locals()/globals() dump
        is_dict_dump = '{' in output and '}' in output and ':' in output

        # First pass: remove entire problematic sections that might span lines
        # Handle __builtins__ dictionary dumps which can be massive
        # Use a more robust pattern that handles nested braces
        def collapse_builtins(match):
            """Collapse __builtins__ dict to a simple marker."""
            return match.group(1) + ": <filtered>"

        # Match __builtins__ with balanced braces using recursive pattern simulation
        # This handles nested dicts/objects within __builtins__
        output = re.sub(r"('__builtins__'|\"__builtins__\"):\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
                       collapse_builtins, output, flags=re.DOTALL)

        # Also handle __builtins__ as a module reference
        output = re.sub(r"('__builtins__'|\"__builtins__\"):\s*<module 'builtins'[^>]*>",
                       r"\1: <filtered>", output)

        # Remove multi-line credits/copyright text
        output = re.sub(r"'credits':\s*Thanks to.*?information\.,?", "", output, flags=re.DOTALL)
        output = re.sub(r'"credits":\s*Thanks to.*?information\.,?', '', output, flags=re.DOTALL)

        # Remove license and copyright objects
        output = re.sub(r"'(license|copyright)':\s*Copyright.*?(?=,\s*'|\})", "", output, flags=re.DOTALL)
        output = re.sub(r'"(license|copyright)":\s*Copyright.*?(?=,\s*"|\})', '', output, flags=re.DOTALL)

        # Re-split lines after removing multi-line sections
        lines = output.split('\n')

        consecutive_builtins = 0
        MAX_CONSECUTIVE_BUILTINS = 3  # If we see 3+ consecutive builtin lines, start aggressive filtering

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Check against removal patterns
            should_remove = False
            for pattern in remove_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    should_remove = True
                    break

            # Track consecutive builtin-looking lines
            is_builtin_line = (
                re.search(r"<class '\w+'>", line) or
                re.search(r"<built-in function", line) or
                re.search(r"'\w+(Error|Warning|Exception)':", line) or
                re.search(r"'__(debug|import)__':", line) or
                # Quoted multiline text (docstring fragments)
                (re.match(r"^\s*'[^']*'$", line) and len(line.strip()) > 40)
            )

            if is_builtin_line:
                consecutive_builtins += 1
                # If we've seen too many consecutive builtins, start filtering more aggressively
                if consecutive_builtins > MAX_CONSECUTIVE_BUILTINS:
                    should_remove = True
            else:
                consecutive_builtins = 0

            if should_remove:
                continue

            # For dictionary dumps, filter individual entries
            if is_dict_dump:
                # Skip lines with __builtins__: <filtered> markers (we don't need them)
                if '<filtered>' in line and '__builtins__' in line:
                    continue

                # Skip lines that are just dunder variable definitions
                # Match exactly __word__ patterns (must have double underscores on both sides)
                if re.match(r"\s*['\"]__[a-zA-Z_]+__['\"]:", line):
                    continue

                # Skip very long variable representations
                if len(line) > MAX_VARIABLE_SIZE:
                    # Try to extract just the variable name
                    match = re.match(r"(\s*'[^']+'):\s*", line)
                    if match:
                        var_name = match.group(1)
                        truncated_line = f"{var_name}: <truncated, {len(line)} chars>"
                        filtered_lines.append(truncated_line)
                        current_size += len(truncated_line)
                        variables_shown += 1
                    continue

            # Check size limits
            if current_size + len(line) > MAX_OUTPUT_SIZE:
                # Add truncation marker
                remaining_bytes = len('\n'.join(lines)) - current_size
                output_hash = hashlib.sha1(output.encode()).hexdigest()[:8]
                truncation_msg = (
                    f"\n<OUTPUT_TRUNCATED "
                    f"remaining_bytes={remaining_bytes} "
                    f"total_lines={len(lines)} "
                    f"shown_lines={len(filtered_lines)} "
                    f"sha1={output_hash}>"
                )
                filtered_lines.append(truncation_msg)
                break

            # Check variable limit for dict dumps
            if is_dict_dump and variables_shown >= MAX_VARIABLES:
                remaining_vars = len([l for l in lines if ':' in l]) - variables_shown
                filtered_lines.append(f"\n<VARIABLES_TRUNCATED remaining={remaining_vars}>")
                break

            # Keep this line
            filtered_lines.append(line)
            current_size += len(line)
            if ':' in line and is_dict_dump:
                variables_shown += 1

        filtered_output = '\n'.join(filtered_lines).strip()

        # Final cleanup: remove orphaned quoted string fragments from docstrings
        # These are lines that are just quoted strings (likely from multi-line docstrings)
        if filtered_output:
            final_lines = []
            for line in filtered_output.split('\n'):
                # Skip lines that are just single-quoted or double-quoted strings
                # These are orphaned fragments of filtered docstrings
                stripped = line.strip()
                is_orphaned_string = (
                    re.match(r"^'[^']*'$", stripped) or
                    re.match(r'^"[^"]*"$', stripped) or
                    stripped in ["'\\n'", '"\\n"', "''", '""']
                )
                if not is_orphaned_string:
                    final_lines.append(line)

            filtered_output = '\n'.join(final_lines).strip()

        # If filtering removed everything, provide a helpful message
        if not filtered_output and output:
            return "<All output filtered - contained only Python internals>"

        return filtered_output
    
    async def cleanup_async(self):
        """Clean up the PDB process asynchronously."""
        if self.child and self.child.isalive():
            try:
                # Send quit command first
                self.child.sendline("q")
                self.child.expect(pexpect.EOF, timeout=2)
            except:
                pass
            finally:
                if self.child.isalive():
                    try:
                        self.child.close(force=True)
                    except:
                        pass
                self.child = None
        self._cleanup_trace_plugin()

    def cleanup(self) -> None:
        """Synchronous cleanup for callers that aren't async."""
        if not self.child:
            return
        try:
            if self.child.isalive():
                try:
                    self.child.sendline("q")
                    self.child.expect(pexpect.EOF, timeout=2)
                except Exception:
                    pass
        finally:
            if self.child and self.child.isalive():
                try:
                    self.child.close(force=True)
                except Exception:
                    pass
            self.child = None
        self._cleanup_trace_plugin()


class PdbHarness:
    """
    Harness for conducting real pdb debugging sessions.
    
    This class provides programmatic access to pdb functionality,
    maintaining session state and providing authentic debugging outputs.
    Includes translation layer for file operations.
    """
    
    def __init__(self, working_directory: str = "."):
        self.current_session: Optional[PdbSession] = None
        self.session_history: List[str] = []
        self.working_directory = Path(working_directory).resolve()
        # Import FileInterfaceHarness for file operations. It ships alongside
        # this module (harness/file_interface/), so an ImportError is a
        # packaging bug — fail loudly rather than silently running the whole
        # session with file operations disabled.
        import sys
        harness_dir = Path(__file__).parent.parent
        file_interface_path = harness_dir / "file_interface"
        sys.path.insert(0, str(file_interface_path))
        from file_interface import FileInterfaceHarness
        self.file_interface = FileInterfaceHarness(str(self.working_directory))
        
    def start_session(self, *args, test_selection: Optional[str] = None) -> str:
        """Start a new debugging session.

        Supports two call signatures:
        - start_session(file_path: str)
        - start_session(code: str, filename: str)

        ``test_selection`` names one test inside the file to run instead of all
        of them. Keyword-only so it cannot be confused with the ``filename``
        of the two-positional form.
        """
        if self.current_session:
            self.current_session.cleanup()

        # Determine call signature
        if len(args) == 1:
            target_file_path = args[0]
            if not isinstance(target_file_path, str):
                return "Error: start_session(file_path) expects a string path"
            # Convert to absolute path relative to working directory
            if not os.path.isabs(target_file_path):
                target_file_path = str(self.working_directory / target_file_path)
        elif len(args) == 2:
            code, filename = args
            if not isinstance(code, str) or not isinstance(filename, str):
                return "Error: start_session(code, filename) expects two strings"
            # Write code to working directory/filename
            target_path = self.working_directory / filename
            try:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(code)
                target_file_path = str(target_path)
            except Exception as e:
                return f"Error writing code to {target_path}: {e}"
        else:
            return "Error: start_session expects 1 or 2 arguments"

        self.current_session = PdbSession(target_file_path, test_selection=test_selection)
        result = self.current_session.start()
        self.session_history.append(f"[SESSION_START] {result}")
        return result
    
    def execute_tool_call(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result."""
        # Validate tool call against common definitions.
        # This harness also supports a few harness-only tools that aren't part of the
        # global debugger-rl toolset (used mainly for local harness tests).
        harness_only_tools = {"run_program", "propose_fix", "verify_output"}
        if tool_call.name not in TOOLS_DESCRIPTIONS and tool_call.name not in harness_only_tools:
            error_msg = f"Unknown tool '{tool_call.name}'. Available tools: {list(TOOLS_DESCRIPTIONS.keys())[:5]}..."
            self.session_history.append(f"[ERROR] {error_msg}")
            return error_msg
            
        # Check if it's a file operation tool call
        file_operations = {'edit_file', 'write_file'}
        
        if tool_call.name in file_operations:
            # Delegate to file interface harness
            result = self.file_interface.execute_tool_call(tool_call)
            self.session_history.append(f"[{tool_call.name.upper()}] {result[:100]}...")
            return result
        
        # Handle PDB-specific operations
        if not self.current_session and tool_call.name not in ['start_debugger']:
            return "Error: No debugging session active. Use start_debugger first."
            
        method_name = f"_execute_{tool_call.name}"
        if hasattr(self, method_name):
            try:
                result = getattr(self, method_name)(tool_call)
                self.session_history.append(f"[{tool_call.name.upper()}] {result}")
                return result
            except Exception as e:
                error_msg = f"Error executing {tool_call.name}: {str(e)}"
                self.session_history.append(f"[ERROR] {error_msg}")
                return error_msg
        else:
            error_msg = f"Tool {tool_call.name} not implemented in PdbHarness"
            self.session_history.append(f"[ERROR] {error_msg}")
            return error_msg
    
    def _execute_start_debugger(self, tool_call: ToolCall) -> str:
        """Start the pdb debugger, restarting the session if one is running."""
        # Allow both file_path and file.
        file_path = tool_call.parameters.get('file_path') or tool_call.parameters.get('file')
        test_selection = (
            tool_call.parameters.get('test')
            or tool_call.parameters.get('test_name')
        )
        if isinstance(test_selection, str):
            test_selection = test_selection.strip() or None
        elif test_selection is not None:
            return "Error: `test` expects a test name or -k expression, as a string."

        # Check if we have an active session with a running process
        has_active_process = (
            self.current_session and
            self.current_session.is_alive()
        )

        if has_active_process:
            # Restart instead of a no-op. Models call start_debugger
            # mid-session to rerun the test after an edit. A "Debugger already
            # running" reply would waste the turn and leave them inside the
            # stale run.
            requested = file_path
            if requested and not os.path.isabs(requested):
                requested = str(self.working_directory / requested)
            same_target = not requested or (
                os.path.abspath(requested)
                == os.path.abspath(self.current_session.existing_file_path)
            )
            # A changed selection is a changed target even on the same file:
            # restarting in place would silently keep running the previous set of
            # tests and land the breakpoints back where they were.
            if same_target and test_selection != self.current_session.test_selection:
                same_target = False
            if same_target:
                outcome = self.current_session.restart(reapply_breakpoints=True)
                if outcome.get("alive"):
                    self.current_session.execution_state = "debugging"
                    reapplied = outcome.get("reapplied") or []
                    bp_note = ""
                    if reapplied:
                        bp_desc = ", ".join(
                            f"{f or self.current_session.filename}:{ln}"
                            for f, ln in reapplied
                        )
                        bp_note = f" Re-applied breakpoints: {bp_desc}."
                    return (
                        f"Debugger restarted from the beginning of "
                        f"{self.current_session.filename}.{bp_note}\n"
                        f"{outcome.get('start_output', '')}"
                    ).rstrip()
                return (
                    "Error: Failed to restart debugger: "
                    f"{outcome.get('start_output', '')}"
                )
            # Different target file: fall through — start_session() below tears
            # down the current session and starts fresh on the new file.

        # Need to start a new session
        if not file_path:
            return "Error: No file_path provided. You must specify which file to debug (e.g. the test file)."

        # Verify file exists before trying to start
        if not os.path.isabs(file_path):
            file_path = str(self.working_directory / file_path)

        if not os.path.exists(file_path):
            return f"Error: File does not exist: {file_path}"

        # Start the debugging session
        result = self.start_session(file_path, test_selection=test_selection)
        self.session_history.append(f"[AUTO_START] {result}")

        # Verify the session actually started
        if self.current_session and self.current_session.is_alive():
            self.current_session.execution_state = "debugging"
            return result
        # start_session already returns an honest "Error: ..." on a failed start
        # (collection/syntax error, no tests). Don't wrap it in a second,
        # self-contradictory "Failed to start debugger: Debugging session
        # started ..." prefix — surface it directly.
        if isinstance(result, str) and result.lstrip().startswith("Error"):
            return result
        return f"Error: Failed to start debugger: {result}"
    
    def _execute_add_breakpoint(self, tool_call: ToolCall) -> str:
        """Set breakpoints using real PDB"""
        def _format_breakpoint_failure(line_num: int, response: str, *, file_hint: Optional[str] = None) -> str:
            resp = (response or "").replace("\r", "").strip()
            resp_lower = resp.lower()

            file_part = ""
            if file_hint:
                file_part = f" in {os.path.basename(str(file_hint))}"

            if "blank or comment" in resp_lower:
                msg = (
                    f"Error: Cannot set breakpoint at line {line_num}{file_part} because it is blank/comment-only. "
                    f"Pick an executable line (e.g., an assert or a function call). (PDB: {resp})"
                )
            elif "***" in resp or "error" in resp_lower:
                msg = f"Error: Failed to set breakpoint at line {line_num}{file_part}. (PDB: {resp})"
            else:
                msg = f"Error: Breakpoint failed at line {line_num}{file_part}. (PDB: {resp})"

            # Show the code around the rejected line so picking a valid one
            # doesn't require a follow-up read.
            context = self._source_window(
                self._resolve_bp_context_file(file_hint), line_num, "! "
            )
            if context:
                msg += f"\n{context}"
            return msg

        # Support simple 'line' parameter for convenience
        if 'line' in tool_call.parameters:
            try:
                line_num = int(tool_call.parameters.get('line'))
                condition = tool_call.parameters.get('condition')
                # Reject an unhittable line (blank/comment, def header, or
                # module/class-body scope) outright — no auto-adjust; the
                # model must read the code and pick a real statement.
                target_line, adjust_note, bp_error = self._resolve_bp_line(line_num, None)
                if bp_error:
                    return bp_error
                # Always file-qualify: at the pre-run prompt the current frame
                # is pytest's bootstrap, so a bare `b {line}` would land in the
                # wrong file.
                bp_target = os.path.abspath(self.current_session.existing_file_path)
                cmd = f"b {bp_target}:{target_line}"
                if condition:
                    cmd = f"b {bp_target}:{target_line}, {condition}"
                response = self.current_session.send_command(cmd)

                # Validate PDB response to determine success/failure
                import re
                match = re.search(r'Breakpoint (\d+) at', response)
                if match:
                    # Success - track the breakpoint
                    bp_num = int(match.group(1))
                    bp_entry = (self.current_session.filename, target_line)
                    if bp_entry not in self.current_session.breakpoints:
                        self.current_session.breakpoints.append(bp_entry)
                    self.current_session.breakpoint_numbers[bp_num] = bp_entry
                    msg = f"Breakpoint set at line {target_line}"
                    line_text = self._source_line_text(target_line)
                    if line_text:
                        msg += f": {line_text}"
                    if condition:
                        msg += f" (condition: {condition})"
                    msg += adjust_note
                    msg += self._scope_warning(target_line)
                    context = self._breakpoint_context(target_line)
                    if context:
                        msg += f"\n{context}"
                    return msg
                elif "***" in response or "Error" in response:
                    # PDB returned an error
                    return _format_breakpoint_failure(target_line, response, file_hint=self.current_session.filename)
                else:
                    # Unexpected response - return it for debugging
                    return f"Breakpoint response at line {target_line}: {response}"
            except (ValueError, TypeError):
                return "Error: Invalid line number specified"

        # Backward-compatible: complex breakpoints structure
        breakpoints_param = tool_call.parameters.get('breakpoints')
        if not breakpoints_param:
            return "Error: No line number specified"

        try:
            if isinstance(breakpoints_param, str):
                import json
                breakpoints = json.loads(breakpoints_param)
            else:
                breakpoints = breakpoints_param

            results = []
            for bp in breakpoints:
                if isinstance(bp, dict):
                    file_name = bp.get('file', self.current_session.filename)
                    lines = bp.get('lines', [])
                    condition = bp.get('condition')
                    if isinstance(lines, int):
                        lines = [lines]

                    for line_num in lines:
                        try:
                            line_num = int(line_num)
                            bp_file = (
                                None
                                if file_name == self.current_session.filename
                                else self._resolve_breakpoint_file(file_name)
                            )
                            # Reject an unhittable line (blank/comment, def
                            # header, or module/class-body scope) outright —
                            # no auto-adjust; the model must read the code and
                            # pick a real statement. Skip validation when a
                            # cross-file path didn't resolve: we'd be checking
                            # the WRONG file, and pdb's own error (plus the
                            # retry-path hint below) covers that case.
                            if file_name == self.current_session.filename or bp_file is not None:
                                target_line, adjust_note, bp_error = self._resolve_bp_line(line_num, bp_file)
                                if bp_error:
                                    results.append(bp_error)
                                    continue
                            else:
                                target_line, adjust_note = line_num, ""
                            if file_name != self.current_session.filename:
                                # Use the RESOLVED absolute path — a bare
                                # 'subdir/loader.py' isn't found from sys.path
                                # and pdb rejects the breakpoint.
                                bp_target = bp_file or file_name
                            else:
                                # Session file: qualify explicitly — at the
                                # pre-run prompt the current frame is pytest's
                                # bootstrap, so a bare `b {line}` would land
                                # in the wrong file.
                                bp_target = os.path.abspath(
                                    self.current_session.existing_file_path
                                )
                            cmd = f"b {bp_target}:{target_line}"
                            if condition:
                                cmd = f"{cmd}, {condition}"
                            response = self.current_session.send_command(cmd)

                            # Validate PDB response to determine success/failure
                            import re
                            match = re.search(r'Breakpoint (\d+) at', response)
                            if match:
                                # Success - track the breakpoint
                                bp_num = int(match.group(1))
                                bp_entry = (file_name, target_line)
                                if bp_entry not in self.current_session.breakpoints:
                                    self.current_session.breakpoints.append(bp_entry)
                                self.current_session.breakpoint_numbers[bp_num] = bp_entry
                                msg = f"Line {target_line}: Breakpoint set"
                                if (
                                    file_name != self.current_session.filename
                                    and bp_file is None
                                ):
                                    # Cross-file breakpoint whose path didn't
                                    # resolve — don't echo a line from the
                                    # wrong (session) file.
                                    line_text = ""
                                else:
                                    line_text = self._source_line_text(
                                        target_line, bp_file
                                    )
                                if line_text:
                                    msg += f" — {line_text}"
                                if condition:
                                    msg += f" (condition: {condition})"
                                msg += adjust_note
                                msg += self._scope_warning(target_line, bp_file)
                                # Source window with a B marker so the model
                                # sees exactly where the breakpoint sits.
                                # Skip unresolved cross-file paths (we'd read
                                # the wrong file).
                                if file_name == self.current_session.filename or bp_file is not None:
                                    context = self._breakpoint_context(target_line, bp_file)
                                    if context:
                                        msg += f"\n{context}"
                                results.append(msg)
                            elif "***" in response or "Error" in response:
                                # PDB returned an error
                                fail = _format_breakpoint_failure(target_line, response, file_hint=file_name)
                                # If the path didn't resolve but the basename
                                # exists in the workspace, point at the exact
                                # path to retry with.
                                if (
                                    file_name != self.current_session.filename
                                    and bp_file is None
                                ):
                                    located = self._resolve_breakpoint_file(
                                        os.path.basename(str(file_name))
                                    )
                                    if located:
                                        fail += (
                                            f" The file resolves to {located}; "
                                            f"retry add_breakpoint with that path."
                                        )
                                results.append(fail)
                            else:
                                # Unexpected response - include it for debugging
                                results.append(f"Line {target_line}: {response}")
                        except (ValueError, TypeError):
                            results.append(f"Error: Invalid line number '{line_num}'")
                else:
                    results.append(f"Error: Invalid breakpoint format: {bp}")

            return "\n".join(results)

        except Exception as e:
            return f"Error: Invalid breakpoints format: {str(e)}"
    
    def _remove_one_breakpoint(self, file_name: str, line_num: int) -> str:
        """Clear a single tracked breakpoint honestly (see _execute_remove_breakpoint)."""
        session = self.current_session
        base = os.path.basename(str(file_name)) if file_name else session.filename

        def _entry_matches(entry) -> bool:
            try:
                ef, el = entry
            except (TypeError, ValueError):
                return False
            ebase = os.path.basename(str(ef)) if ef else session.filename
            return el == line_num and ebase == base

        # Primary path: clear by the tracked pdb breakpoint number(s).
        matches = [
            (num, entry)
            for num, entry in session.breakpoint_numbers.items()
            if _entry_matches(entry)
        ]
        if matches:
            deleted_nums, deleted_entries, errors = [], [], []
            for num, entry in matches:
                resp = (session.send_command(f"clear {num}") or "").strip()
                if "Deleted" in resp:
                    deleted_nums.append(num)
                    deleted_entries.append(entry)
                else:
                    errors.append(resp)
            for num in deleted_nums:
                session.breakpoint_numbers.pop(num, None)
            for entry in deleted_entries:
                if entry in session.breakpoints:
                    session.breakpoints.remove(entry)
            if deleted_nums:
                return f"Removed breakpoint at {base}:{line_num}."
            return (
                f"Error: Failed to remove breakpoint at {base}:{line_num}. "
                f"(PDB: {'; '.join(e for e in errors if e) or 'no confirmation'})"
            )

        # Fallback: no tracked number — clear by file:line so pdb resolves it.
        bp_file = (
            self._resolve_breakpoint_file(file_name)
            if file_name and base != session.filename
            else None
        )
        target = bp_file or base
        resp = (session.send_command(f"clear {target}:{line_num}") or "").strip()
        if "Deleted" in resp:
            session.breakpoints = [
                b for b in session.breakpoints if not _entry_matches(b)
            ]
            session.breakpoint_numbers = {
                k: v for k, v in session.breakpoint_numbers.items()
                if not _entry_matches(v)
            }
            return f"Removed breakpoint at {base}:{line_num}."
        # A bare 'clear' with no valid arg can trigger the "Clear all breaks?"
        # prompt — decline it so we never wipe unrelated breakpoints.
        if "Clear all breaks" in resp:
            session.send_command("n")
            return f"Error: No breakpoint found at {base}:{line_num} to remove."
        return (
            f"Error: No breakpoint found at {base}:{line_num} to remove. "
            f"(PDB: {resp or 'no matching breakpoint'})"
        )
    
    def _resolve_breakpoint_file(self, file_name: str) -> Optional[str]:
        """Best-effort resolve a breakpoint file reference to an absolute path."""
        if not file_name:
            return None
        if os.path.isabs(file_name) and os.path.exists(file_name):
            return file_name
        candidate = self.working_directory / file_name
        if candidate.exists():
            return str(candidate)
        # Fall back to the directory of the session's target file.
        if self.current_session and self.current_session.existing_file_path:
            sibling = Path(self.current_session.existing_file_path).parent / file_name
            if sibling.exists():
                return str(sibling)
        return None

    def _resolve_bp_line(
        self, requested_line: int, bp_file: Optional[str]
    ) -> Tuple[int, str, Optional[str]]:
        """Validate a requested breakpoint line.

        Returns ``(line_to_use, note, error)``. ``note`` is always ``""``
        (kept for call-site compatibility). ``error`` is non-None when NO
        breakpoint should be set: the line is past end-of-file, or it is not
        an executable statement inside a function (blank/comment line,
        def/decorator header, docstring, or module/class-body line).

        The method does not move a bad request onto the nearest executable
        in-function line. A model would lean on that ("guess a line, the
        harness fixes it") instead of reading the code, and the harness would
        do the localization work for it. A bad line is an explicit error naming
        the offending line's text, with no replacement-line suggestion, so
        choosing a real statement requires reading the code. Callers can key
        on the literal marker ``not an executable statement inside a function``.
        """
        if not self.current_session:
            return requested_line, "", None
        session = self.current_session
        target_file = bp_file or session.existing_file_path
        where = os.path.basename(str(target_file)) if target_file else ""
        in_file = f" in {where}" if where else ""

        # (a) Past-EOF: distinct error, set nothing. Show the end of the file
        # so the model sees what IS there without a follow-up read
        # Failures carry the same source window that successes do.
        n_lines = session.file_line_count(target_file)
        if n_lines is not None and requested_line > n_lines:
            msg = (
                f"Error: line {requested_line} is beyond end of file "
                f"({n_lines} lines){in_file}. No breakpoint was set — pick an "
                f"executable line inside the target function."
            )
            tail_context = self._source_window(target_file, n_lines, "  ")
            if tail_context:
                msg += f"\n{tail_context}"
            return requested_line, "", msg

        try:
            resolved = session.resolve_breakpoint_line(requested_line, bp_file)
        except Exception:
            resolved = None
        if resolved is None or resolved == requested_line:
            return requested_line, "", None

        # (b) Not an executable in-function statement: hard error, set nothing.
        # The surrounding source window (with the rejected line marked) is
        # included so choosing a real statement doesn't require a follow-up
        # read — same context a successful add_breakpoint shows.
        line_text = self._source_line_text(requested_line, bp_file)
        shown = f": `{line_text}`" if line_text else " (blank line)"
        msg = (
            f"Error: line {requested_line}{in_file} is not an executable "
            f"statement inside a function{shown}. No breakpoint was set — pick a "
            f"line with a real statement inside the function."
        )
        context = self._source_window(target_file, requested_line, "! ")
        if context:
            msg += f"\n{context}"
        return requested_line, "", msg

    def _scope_warning(self, line_num: int, file_path: Optional[str] = None) -> str:
        """Return a warning suffix if a breakpoint sits on a line that runs at
        import time (module-level / class body) and therefore won't be hit when
        the test calls into the code. Empty string when the line is inside a
        function or scope can't be determined."""
        if not self.current_session:
            return ""
        try:
            inside = self.current_session.line_is_inside_function(line_num, file_path)
        except Exception:
            return ""
        if inside is False:
            suggestion = ""
            try:
                hint = self.current_session.suggest_executable_line(line_num, file_path)
            except Exception:
                hint = None
            if hint:
                func_name, body_line = hint
                suggestion = (
                    " Try line {bl} (the first executable line inside {fn})."
                ).format(bl=body_line, fn=f"{func_name}()")
            return (
                " — WARNING: line {ln} is at module/class-body scope, which runs "
                "once at import (during test collection) — execution would stop "
                "there during import, NOT during the test's call into the code. "
                "Set the breakpoint on an executable line INSIDE the function "
                "the test exercises (e.g. inside __init__ or the method under "
                "test).{sg}"
            ).format(ln=line_num, sg=suggestion)
        return ""

    # Lines of source shown above/below the cursor after a step.
    STEP_CONTEXT_LINES = 3

    def _resolve_source_path(self, file_path: Optional[str]) -> Optional[str]:
        """Best-effort absolute path for a source file reference."""
        if not file_path:
            return None
        if not os.path.isabs(file_path):
            file_path = str(self.working_directory / file_path)
        return file_path if os.path.exists(file_path) else None

    def _resolve_bp_context_file(self, file_hint: Optional[str]) -> Optional[str]:
        """Resolve a breakpoint file reference (None / session filename / any
        path or basename) to a readable source path for context windows."""
        session = self.current_session
        if not file_hint or (session and str(file_hint) == session.filename):
            return session.existing_file_path if session else None
        hint = str(file_hint)
        return self._resolve_source_path(hint) or self._resolve_breakpoint_file(hint)

    def _source_line_text(
        self, line_num: int, file_path: Optional[str] = None
    ) -> str:
        """Return the stripped source text of ``line_num`` in ``file_path``
        (defaults to the session's target file), or "" when unavailable."""
        target = file_path or (
            self.current_session.existing_file_path if self.current_session else None
        )
        target = self._resolve_source_path(target)
        if not target:
            return ""
        try:
            with open(target, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()
        except OSError:
            return ""
        if not 1 <= line_num <= len(lines):
            return ""
        return lines[line_num - 1].strip()

    def _source_window(
        self, file_path: Optional[str], line_no: Optional[int], marker: str
    ) -> str:
        """Numbered source window around ``line_no`` with ``marker`` on that
        line. Empty string when the position is unknown or unreadable."""
        file_path = self._resolve_source_path(file_path)
        if not file_path or not line_no:
            return ""
        return _numbered_source_window(
            file_path, line_no, marker, self.STEP_CONTEXT_LINES
        )

    def _position_context(self) -> str:
        """Numbered source window around the current execution line, with a
        ``->`` cursor marker. Empty string when the position is unknown or the
        file can't be read.

        pdb's own step output only prints the single pending line
        (``-> code``). Models lose orientation without the surrounding lines,
        so steps show a small window.
        """
        session = self.current_session
        if not session:
            return ""
        pos = session.current_position or {}
        return self._source_window(pos.get("file"), pos.get("line"), "->")

    def _breakpoint_context(
        self, line_no: int, file_path: Optional[str] = None
    ) -> str:
        """Numbered source window around a just-set breakpoint with a ``B``
        marker on the breakpoint line, so the model sees the surrounding code
        without a follow-up read."""
        target = file_path or (
            self.current_session.existing_file_path if self.current_session else None
        )
        return self._source_window(target, line_no, "B ")

    # Matches a pdb position line like `> /path/file.py(12)func()` — present
    # only when the debugger is stopped at a live frame (not after the program
    # finished), so context is never appended from a stale position.
    _POSITION_LINE_RE = re.compile(r">\s*\S+\(\d+\)")

    def _execute_continue_execution(self, tool_call: ToolCall) -> str:
        """Continue execution until next breakpoint or end using real PDB.

        Distinguishes "stopped at a breakpoint" (session stays live for
        inspection) from "ran to completion without hitting a breakpoint".
        Under `-m pdb -m pytest` the child survives a completed run (pdb
        reruns the target in-process, breakpoints intact); we still surface an
        actionable diagnostic, and re-spawn only if the child actually died.
        """
        session = self.current_session
        had_breakpoints = bool(session.breakpoints)
        bp_lines = {ln for (_f, ln) in session.breakpoints}

        response = session.send_command("c", timeout=30.0)

        # Did execution land on one of our breakpoints?
        position = session.current_position or {}
        landed_line = position.get("line")
        hit_breakpoint = landed_line in bp_lines if landed_line is not None else False

        completion_markers = (
            "Program execution completed",
            "Program finished without hitting any breakpoint",
            "The program finished",
            "The program exited via sys.exit",
            "Process terminated",
            "Debugger session ended",
        )
        finished = (not session.is_alive()) or any(
            marker in response for marker in completion_markers
        )

        if had_breakpoints and not hit_breakpoint and finished:
            return self._recover_after_missed_breakpoint(response)

        # Show the source window around the stop position, like step_over and
        # step_into. Without it, the model gets only pdb's single `-> line`
        # after a breakpoint hit.
        return self._with_position_context(response)

    def _recover_after_missed_breakpoint(self, continue_response: str) -> str:
        """Build a diagnostic and keep the session usable after `continue` ran
        the program to completion without stopping at any breakpoint."""
        session = self.current_session
        missed = list(session.breakpoints)

        def _fmt(bp) -> str:
            try:
                file_name, line_num = bp
            except (TypeError, ValueError):
                return str(bp)
            base = os.path.basename(file_name) if file_name else session.filename
            return f"{base}:{line_num}"

        bp_desc = ", ".join(_fmt(bp) for bp in missed) or "(none)"

        restart_note = ""
        if session.is_alive():
            # Under `-m pdb` the child survives a completed run: pdb has
            # already rerun the target in-process and is sitting at the
            # pre-run prompt with every breakpoint still installed. No
            # respawn needed.
            restart_note = (
                " The program was RESTARTED from the beginning with your "
                "breakpoints still applied — move or adjust them if needed, "
                "then call continue_execution to run the test again."
            )
        elif session._auto_restart_count < 3:
            session._auto_restart_count += 1
            outcome = session.restart(reapply_breakpoints=True)
            if outcome.get("alive"):
                restart_note = (
                    " The session was RESTARTED before the test run and your "
                    "breakpoints were re-applied — move or adjust them if "
                    "needed, then call continue_execution to run again."
                )
            else:
                restart_note = (
                    " The session could NOT be restarted automatically; call "
                    "start_debugger on the TEST file to begin a fresh session."
                )
        else:
            restart_note = (
                " The session was NOT restarted again (auto-restart limit reached); "
                "call start_debugger on the TEST file to begin a fresh session."
            )

        # Enumerate the ACTUAL likely causes. Since the session starts BEFORE
        # the test run, breakpoints fire in fixture setup, module import, and
        # library code alike — a miss genuinely means the line never ran.
        causes = [
            "the breakpoint line is on a code path this run does not execute "
            "(a branch the failing test never takes, or a function it never "
            "calls)",
            "the breakpoint has a condition that never evaluated true",
        ]
        # Worker-thread caveat only when we could NOT enable thread tracing
        # (with tracing on, thread targets ARE traced).
        if (
            not session.thread_tracing_enabled
            and self._target_uses_threads()
        ):
            causes.append(
                "the code runs in a worker thread (ThreadPoolExecutor / "
                "threading / concurrent.futures), which is not traced here — "
                "inspect it with print statements instead of a breakpoint"
            )
        cause_text = "; or ".join(causes)

        tail = self._summarize_continue_tail(continue_response)

        msg = (
            f"Program ran to completion WITHOUT stopping at your breakpoint(s) "
            f"({bp_desc}). Likely causes: {cause_text}. Move the breakpoint onto "
            f"a line the failing test actually executes (check the traceback "
            f"below for the path it took)."
        )
        if tail:
            msg += f"\n\nTest result from that run:\n{tail}"
        msg += restart_note
        return msg

    def _target_uses_threads(self) -> bool:
        """True when the debugged file — or any file holding one of the missed
        breakpoints — imports threading / concurrent.futures. Used to tailor
        the missed-breakpoint diagnostic (the thread code usually lives in the
        breakpoint's library file, not the test file)."""
        session = self.current_session
        if not session:
            return False
        candidates = [session.existing_file_path]
        for bp in session.breakpoints:
            try:
                file_name, _line = bp
            except (TypeError, ValueError):
                continue
            resolved = self._resolve_breakpoint_file(file_name)
            if resolved:
                candidates.append(resolved)
        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if re.search(
                r"\b(threading|ThreadPoolExecutor|concurrent\.futures)\b", text
            ):
                return True
        return False

    def _summarize_continue_tail(self, continue_response: str, cap: int = 600) -> str:
        """Trim a `continue` response down to the test outcome: the first
        assertion/traceback block through the summary line, capped at ``cap``
        chars. Returns "" when there's nothing test-shaped to surface."""
        if not continue_response:
            return ""
        text = continue_response.strip()
        # Nothing useful if it's only our own restart/completion banner.
        if text.startswith("Program ran to completion WITHOUT stopping"):
            return ""
        # The in-place restart note is re-stated by the recovery message —
        # don't repeat it inside the test-result tail.
        text = text.replace(RESTARTED_IN_PLACE_NOTE, "")
        lines = [l for l in text.split("\n") if l.strip()]
        if not lines:
            return ""
        start_idx = 0
        for i, l in enumerate(lines):
            if any(
                marker in l
                for marker in ("assert", "Error", "Traceback", "Exception", "FAILED")
            ):
                start_idx = i
                break
        snippet = "\n".join(lines[start_idx:]).strip()
        if not snippet:
            return ""
        if len(snippet) > cap:
            snippet = snippet[:cap].rstrip() + " …[truncated]"
        return snippet
    
    def _with_position_context(self, result: str) -> str:
        """Append the source window around the new position to a step result.
        Only when the step actually reported a live position; completion/EOF
        messages carry no `> file(line)` marker."""
        if result and self._POSITION_LINE_RE.search(result):
            context = self._position_context()
            if context:
                return f"{result}\n{context}"
        return result

    def _execute_step_over(self, tool_call: ToolCall) -> str:
        """Step over the current line using real PDB."""
        return self._with_position_context(self.current_session.send_command("n"))

    def _execute_step_into(self, tool_call: ToolCall) -> str:
        """Step into function calls using real PDB.

        Appends the same source window as step_over, so the two tools give
        equal context."""
        return self._with_position_context(self.current_session.send_command("s"))

    def _execute_step_out(self, tool_call: ToolCall) -> str:
        """Step out of the current function using real PDB."""
        # Use PDB's 'r' (return) command to run until current function returns
        return self._with_position_context(
            self.current_session.send_command("r", timeout=30.0)
        )

    def _execute_print_variable(self, tool_call: ToolCall) -> str:
        """Print the value of a variable using real PDB."""
        # Accept arg-name aliases. The model often emits {"variable_name": ...},
        # a natural key. Rejecting it with "No variable name specified" causes
        # repeated failed inspections and no progress.
        p = tool_call.parameters
        var_name = (
            p.get('name')
            or p.get('variable')
            or p.get('variable_name')
            or p.get('var_name')
            or p.get('var')
            or p.get('expression')
        )
        if not var_name:
            return "Error: No variable name specified"

        # Use PDB's 'p' command to print the variable
        return self._augment_name_error(
            self.current_session.send_command(f"p {var_name}")
        )
    
    def _execute_examine_locals(self, tool_call: ToolCall) -> str:
        """Examine local variables using real PDB.

        Also dumps ``self.__dict__`` when ``self`` is in the locals frame.
        Without it, the model calls ``examine_locals`` several times in a row
        when the bug-relevant state lives on ``self`` as an instance
        attribute, not in locals().
        ``pp locals()`` would only show ``'self': <...obj at 0x...>`` and the
        model had to follow up with ``run_expression`` or ``print_variable``
        to see object state. Appending ``vars(self)`` automatically removes
        that round-trip without changing the existing locals() format
        (the new section follows a clear delimiter).
        """
        locals_output = self.current_session.send_command("pp locals()")
        if self._pdb_reply_errored(locals_output):
            return locals_output
        # Labeled sections, not raw echoed pdb commands ("pp locals()" /
        # "--- self.__dict__ ---" separators).
        result = f"Local variables:\n{locals_output.strip()}"
        # Cheap substring check on the pp-formatted locals output. PDB renders
        # a self-reference as `'self': <ClassName object at 0x...>`. If absent
        # (module-level or static-method frames), skip the second roundtrip.
        if "'self':" in locals_output:
            self_dict = self.current_session.send_command("pp vars(self)")
            # vars() raises TypeError on slotted classes; suppress those.
            if not any(err in self_dict for err in
                       ("NameError", "AttributeError", "TypeError", "Traceback")):
                result += (
                    f"\n\nInstance attributes of self (vars(self)):\n"
                    f"{self_dict.strip()}"
                )
        return result
    
    @staticmethod
    def _pdb_reply_errored(reply: str) -> bool:
        """True when a pdb reply is an error/empty rather than a value — used
        to return the raw error instead of wrapping it in a section header."""
        if not reply or not reply.strip():
            return True
        return (
            "***" in reply
            or reply.lstrip().startswith("Error")
            or "No active debugging session" in reply
        )

    def _execute_examine_globals(self, tool_call: ToolCall) -> str:
        """Examine global variables using real PDB."""
        # Use PDB's 'pp globals()' command to print global variables  
        return self.current_session.send_command("pp globals()")
    
    # Shell programs that models try to run inside the debugger
    # (`pytest -xvs ...`, `import subprocess; ...pytest...`).
    # `(?!\s*\()` keeps genuine Python calls like `find(...)` matchable as
    # expressions.
    _SHELL_EXPRESSION_RE = re.compile(
        r"^\s*(pytest|py\.test|python[0-9.]*|pip[0-9.]*|ls|cd|cat|sed|grep|rg|"
        r"find|echo|head|tail|bash|sh|touch|mkdir|rm)\b(?!\s*[(.=])"
    )
    _IMPORT_STATEMENT_RE = re.compile(r"^\s*(import|from)\s")

    def _current_position_note(self) -> str:
        """One-line ``Current position: file:line — code`` suffix appended to
        failed introspection results so the model can reorient without an
        extra where_am_i round-trip. Empty when
        the position is unknown."""
        session = self.current_session
        if not session:
            return ""
        pos = session.current_position or {}
        file_path, line_no = pos.get("file"), pos.get("line")
        if not file_path or not line_no:
            return ""
        line_text = self._source_line_text(line_no, file_path)
        code = f" — `{line_text}`" if line_text else ""
        return (
            f"\nCurrent position: {os.path.basename(str(file_path))}:{line_no}"
            f"{code}"
        )

    def _augment_name_error(self, result: str, expression: Optional[str] = None) -> str:
        """Append an explanation to pdb NameError output for these failure
        modes: referencing `self` outside a method
        frame, referencing a variable assigned on the CURRENT (not-yet-executed)
        line, and probing a name that simply isn't in scope (for which we list
        the names that ARE available so the model stops guessing). Every branch
        also appends the current position so a failed probe still tells the
        model where it is."""
        if not result or "NameError" not in result:
            return result
        position_note = self._current_position_note()
        _pos = (self.current_session.current_position or {}) if self.current_session else {}
        _frame_func = str(_pos.get("function") or "")
        _in_comprehension = _frame_func in (
            "<genexpr>", "<listcomp>", "<setcomp>", "<dictcomp>",
        )
        if re.search(r"name 'self' is not defined", result):
            # The model can step INTO a generator-expression frame, where
            # CPython's only locals are the loop variable and the
            # pre-evaluated iterator (`.0`). `self` legitimately does not exist
            # there. A "you are not stopped inside a method / program may have
            # finished" note would steer the model away from the two correct
            # moves.
            if _in_comprehension:
                return (
                    f"{result}\n"
                    f"Note: you are stopped inside a {_frame_func} frame — a "
                    "comprehension has its own frame; only the loop variable(s) "
                    "and iterator are local, so `self` isn't visible. Inspect the "
                    "loop variable, or step_over back to the enclosing method."
                    f"{position_note}"
                )
            return (
                f"{result}\n"
                "Note: `self` is not defined in the current frame — you are not "
                "stopped inside a method (program finished, or at module level). "
                "Set a breakpoint inside the method and continue_execution to it, "
                "then inspect."
                f"{position_note}"
            )
        match = re.search(r"name '([^']+)' is not defined", result)
        if match:
            name = match.group(1)
            pos = (self.current_session.current_position or {}) if self.current_session else {}
            current_line = ""
            if pos.get("line"):
                current_line = self._source_line_text(pos["line"], pos.get("file"))
            if current_line and re.search(rf"\b{re.escape(name)}\b", current_line):
                return (
                    f"{result}\n"
                    f"Note: pdb stops BEFORE executing the displayed line — "
                    f"`{name}` appears on the current line ({current_line}) and "
                    f"isn't defined yet. step_over once to execute it, then retry."
                )
            # The name is just not in scope — enumerate what IS, so the model
            # stops probing hallucinated names (a large share of failed calls).
            names = self._available_names()
            if names:
                return (
                    f"{result}\n"
                    f"Available names in the current frame: {', '.join(names)}."
                    f"{position_note}"
                )
        return f"{result}{position_note}"

    def _available_names(self, cap: int = 15) -> List[str]:
        """Non-dunder local+global names visible in the current pdb frame,
        capped at ``cap``. Empty list when it can't be determined."""
        if not self.current_session:
            return []
        try:
            # Print a sentinel-delimited comma list. We can't literal_eval the
            # raw set repr because pexpect echoes the (brace-containing) command
            # back, so we parse the last sentinel line and keep only valid
            # identifiers.
            raw = self.current_session.send_command(
                "!print('__NAMES__=' + ','.join(sorted("
                "k for k in list(locals().keys()) + list(globals().keys()) "
                "if not k.startswith('__'))))",
                timeout=5.0,
            )
        except Exception:
            return []
        matches = re.findall(r"__NAMES__=([^\r\n]*)", raw or "")
        if not matches:
            return []
        names = [t for t in matches[-1].split(",") if t.isidentifier()]
        return names[:cap]

    def _execute_run_expression(self, tool_call: ToolCall) -> str:
        """Run a Python expression in the current context using real PDB."""
        expression = tool_call.parameters.get('expression')

        if not expression:
            return "Error: No expression specified"

        if self._IMPORT_STATEMENT_RE.match(expression):
            # An import STATEMENT can't be evaluated, but the model usually just
            # wants a module-level value — point at the working alternatives so
            # it stops retrying the blocked import.
            return (
                "Error: run_expression evaluates a Python EXPRESSION in the "
                "current debugger frame — it cannot run an import statement. To "
                "read a module value without importing, use "
                "__import__('<module>').<Name>. To run tests/shell commands "
                "(pytest, pip, ...), use the bash tool."
            )

        if self._SHELL_EXPRESSION_RE.match(expression):
            return (
                "Error: run_expression evaluates a Python EXPRESSION in the "
                "current debugger frame — it cannot run shell commands; use the "
                "bash tool for pytest, pip, ls, etc."
            )

        # `p` evaluates a single EXPRESSION, so a semicolon-joined compound
        # ("self._trail.record(...); self._locks._labels") would be a raw
        # SyntaxError. Execute the leading statements via pdb's `!`, then
        # evaluate only the final segment as the value-producing expression.
        segments = self._split_top_level_semicolons(expression)
        setup_output = ""
        if len(segments) > 1:
            stmt = "; ".join(segments[:-1])
            setup = self.current_session.send_command(f"!{stmt}", timeout=5.0)
            setup_clean = (setup or "").strip()
            if setup_clean and any(
                marker in setup_clean
                for marker in ("***", "Traceback", "SyntaxError", "Error")
            ):
                return (
                    f"Error while executing statement(s) `{stmt}`:\n"
                    f"{setup_clean}{self._current_position_note()}"
                )
            if setup_clean:
                setup_output = f"{setup_clean}\n"
            expression = segments[-1]

        result = self._eval_single_expression(expression)
        return f"{setup_output}{result}" if setup_output else result

    @staticmethod
    def _split_top_level_semicolons(text: str) -> List[str]:
        """Split ``text`` on semicolons that sit outside quotes and brackets.
        Returns the non-empty stripped segments (a single-element list when
        there is no top-level semicolon)."""
        parts: List[str] = []
        depth = 0
        quote: Optional[str] = None
        escape = False
        start = 0
        for i, ch in enumerate(text):
            if quote:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    quote = None
                continue
            if ch in "\"'":
                quote = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            elif ch == ";" and depth == 0:
                parts.append(text[start:i])
                start = i + 1
        parts.append(text[start:])
        return [p.strip() for p in parts if p.strip()]

    def _eval_single_expression(self, expression: str) -> str:
        """Evaluate one Python expression via pdb's ``p``, with the retry /
        augmentation layers (empty-echo reprint, comprehension-scope merge,
        NameError/AttributeError context)."""
        # Use PDB's 'p' command to evaluate the expression
        # Use shorter timeout for expressions that might infinite loop
        try:
            result = self.current_session.send_command(f"p {expression}", timeout=5.0)
            if "Timeout" in result:
                return f"Expression '{expression}' timed out (possible infinite loop or long computation)"
            # Always render a value or an explicit error: pdb occasionally
            # returns nothing printable (value filtered out, or only the echoed
            # command survived). Re-issue via an explicit print so the model
            # never sees a value-less echo.
            cleaned = (result or "").strip()
            if not cleaned or cleaned in (expression.strip(), f"p {expression}".strip()):
                retry = self.current_session.send_command(
                    f"!print(repr({expression}))", timeout=5.0
                )
                rc = (retry or "").strip()
                if rc and rc != f"!print(repr({expression}))":
                    result = retry
                else:
                    result = (
                        f"Expression '{expression}' evaluated but produced no "
                        f"displayable value (the result may be None or empty)."
                    )
            # pdb evaluates `p expr` with the frame's locals passed as a
            # SEPARATE dict. On Python before 3.12 (no PEP 709 comprehension
            # inlining), a comprehension/lambda body gets its own scope that
            # only sees globals. Frame locals (`self`, `binding`, ...) then
            # raise NameError even though they ARE in scope. Retry with a
            # merged namespace before surfacing the error.
            if "NameError" in result and re.search(r"\bfor\b|\blambda\b", expression):
                merged_cmd = (
                    f"!print(repr(eval({expression!r}, "
                    f"{{**globals(), **locals()}})))"
                )
                merged = self.current_session.send_command(merged_cmd, timeout=5.0)
                mc = (merged or "").strip()
                if (
                    mc
                    and mc != merged_cmd
                    and not any(
                        marker in mc
                        for marker in (
                            "NameError",
                            "AttributeError",
                            "TypeError",
                            "SyntaxError",
                            "Traceback",
                            "***",
                        )
                    )
                ):
                    return mc
            # Failed attribute probes get the position suffix too, so the
            # model can reorient without a where_am_i round-trip.
            if "AttributeError" in result:
                return f"{result}{self._current_position_note()}"
            return self._augment_name_error(result, expression)
        except Exception as e:
            return f"Error evaluating expression '{expression}': {str(e)}"
    
    def _execute_list_code(self, tool_call: ToolCall) -> str:
        """List code lines, labeled with the file they came from.

        pdb's bare ``l start,end`` is frame-relative and unlabeled, so a model
        that lists while stopped in the test file believes it is reading the
        target module and loops. We prefix an unambiguous ``File:`` header. An
        optional ``file`` argument reads the requested lines FRESH from disk
        (which also fixes stale-linecache reads after an edit).
        """
        p = tool_call.parameters
        start = p.get('start')
        end = p.get('end')
        file_arg = p.get('file') or p.get('file_path')

        if file_arg:
            return self._list_from_disk(file_arg, start, end)

        # While the current frame is OUTSIDE the workspace (the pre-continue
        # pytest/runpy bootstrap, or any interpreter-internal stop), a bare `l`
        # lists Python internals, the same noise start() suppresses from its
        # banner. The harness overwrites the session's "started" state at
        # dispatch, so gate on the frame's file instead. It is authoritative
        # regardless of state bookkeeping.
        session = self.current_session
        pos = (session.current_position or {}) if session else {}
        frame_file = str(pos.get("file") or "")
        wd = str(getattr(self, "working_directory", "") or "")
        if frame_file and wd and not os.path.abspath(frame_file).startswith(
            os.path.abspath(wd) + os.sep
        ):
            return (
                "The current frame is inside Python's bootstrap/interpreter "
                f"internals ({os.path.basename(frame_file)}), not scenario "
                "code — a bare list_code here shows interpreter source. Pass "
                "file='<name>' to read scenario source from disk, or set "
                "breakpoints and call continue_execution to stop in real code "
                "first."
            )

        # Build PDB 'l' (list) command for the current frame's file.
        if start and end:
            raw = self.current_session.send_command(f"l {start}, {end}")
        elif start:
            raw = self.current_session.send_command(f"l {start}")
        else:
            # List around current line (default behavior)
            raw = self.current_session.send_command("l")
        return self._label_list_output(raw, start, end)

    def _label_list_output(self, raw: str, start=None, end=None) -> str:
        """Prefix pdb's ``l`` output with a ``File: <abs path> (lines A-B)``
        header for the current frame's file."""
        if not raw or raw.startswith("Error") or "No active debugging session" in raw:
            return raw
        session = self.current_session
        pos = (session.current_position or {}) if session else {}
        fpath = pos.get("file") or (session.existing_file_path if session else None)
        fpath = self._resolve_source_path(fpath) or (fpath or "")
        nums = [int(m) for m in re.findall(r'^\s*(\d+)\b', raw, re.MULTILINE)]
        if nums:
            span = f" (lines {min(nums)}-{max(nums)})"
        elif start and end:
            span = f" (lines {start}-{end})"
        elif start:
            span = f" (from line {start})"
        else:
            span = ""
        return f"File: {fpath}{span}\n{raw}"

    def _list_from_disk(self, file_arg: str, start=None, end=None) -> str:
        """Read requested lines of ``file_arg`` fresh from disk with a ``File:``
        header — bypasses pdb's frame-relative listing and any stale linecache."""
        path = self._resolve_source_path(file_arg) or self._resolve_breakpoint_file(file_arg)
        if not path or not os.path.exists(path):
            return f"Error: File not found: {file_arg}"
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError as e:
            return f"Error reading {file_arg}: {e}"
        n = len(lines)
        try:
            s = int(start) if start else 1
        except (ValueError, TypeError):
            s = 1
        try:
            e = int(end) if end else min(n, s + 10)
        except (ValueError, TypeError):
            e = min(n, s + 10)
        s = max(1, s)
        if s > n:
            return (
                f"File: {path}\n"
                f"Error: start line {s} is beyond end of file ({n} lines)."
            )
        e = min(n, max(s, e))
        rows = [f"{i:>4}\t{lines[i - 1].rstrip()}" for i in range(s, e + 1)]
        return f"File: {path} (lines {s}-{e})\n" + "\n".join(rows)
    
    def _execute_where_am_i(self, tool_call: ToolCall) -> str:
        """Show current execution location using real PDB."""
        # Use PDB's 'w' (where) command to show call stack and current location
        return self.current_session.send_command("w")

    def _execute_inspect_stack(self, tool_call: ToolCall) -> str:
        """Show the call stack with local variables at each frame."""
        max_frames = tool_call.parameters.get('max_frames')
        try:
            max_frames = int(max_frames) if max_frames is not None else None
        except (ValueError, TypeError):
            max_frames = None

        # Internal frames to skip (not useful for debugging user code)
        _INTERNAL_PATHS = ('bdb.py', 'pdb.py', '<string>')

        def _is_internal(frame_line: str) -> bool:
            return any(p in frame_line for p in _INTERNAL_PATHS)

        # Get the current stack trace
        stack_output = self.current_session.send_command("w")
        if not stack_output:
            return "Error: Could not retrieve stack"

        # Count frame markers in the stack output
        frame_markers = [l for l in stack_output.split('\n') if re.match(r'\s*>?\s*/.+\(\d+\)', l) or re.match(r'\s*>?\s*<.+>\(\d+\)', l)]
        num_frames = max(len(frame_markers), 1)

        # Move to oldest frame
        for _ in range(num_frames + 5):
            resp = self.current_session.send_command("u")
            if "Oldest frame" in resp:
                break

        # Walk down collecting frame info + locals, skipping internal frames
        sections = []
        user_frame_idx = 0
        for _ in range(num_frames + 5):
            if max_frames and user_frame_idx >= max_frames:
                break

            # Get current frame info via 'w' and find the '>' marker
            where = self.current_session.send_command("w")
            current_frame = None
            for line in where.split('\n'):
                if line.strip().startswith('>'):
                    current_frame = line.strip()
                    break

            if current_frame and not _is_internal(current_frame):
                # Get locals using '!' to force Python eval (avoids PDB parse issues)
                locals_output = self.current_session.send_command(
                    "!{k: v for k, v in locals().items() if not k.startswith('__')}"
                )
                # Clean echoed command from output
                cleaned_locals = '\n'.join(
                    l for l in locals_output.split('\n')
                    if not l.strip().startswith('!{') and not l.strip().startswith('{k:')
                ).strip() or locals_output.strip()

                sections.append(f"--- Frame {user_frame_idx} ---\n{current_frame}\nLocals: {cleaned_locals}")
                user_frame_idx += 1

            # Move down one frame
            resp = self.current_session.send_command("d")
            if "Newest frame" in resp:
                # Capture this last frame if it's a user frame
                if not current_frame or _is_internal(current_frame or ""):
                    where = self.current_session.send_command("w")
                    for line in where.split('\n'):
                        if line.strip().startswith('>'):
                            current_frame = line.strip()
                            break
                    if current_frame and not _is_internal(current_frame):
                        locals_output = self.current_session.send_command(
                            "!{k: v for k, v in locals().items() if not k.startswith('__')}"
                        )
                        cleaned_locals = '\n'.join(
                            l for l in locals_output.split('\n')
                            if not l.strip().startswith('!{') and not l.strip().startswith('{k:')
                        ).strip() or locals_output.strip()
                        sections.append(f"--- Frame {user_frame_idx} ---\n{current_frame}\nLocals: {cleaned_locals}")
                break

        if not sections:
            return f"Stack trace:\n{stack_output}"

        return "\n\n".join(sections)

    def _current_file_path(self) -> Optional[str]:
        if self.current_session:
            return self.current_session.existing_file_path
        return None

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session state."""
        if not self.current_session:
            return {"status": "no_session"}
            
        return {
            "status": self.current_session.execution_state,
            "current_position": self.current_session.current_position,
            "breakpoints": list(self.current_session.breakpoints),
            "filename": self.current_session.filename
        }
    
    async def cleanup_async(self):
        """Clean up the harness and any active sessions asynchronously."""
        if self.current_session:
            await self.current_session.cleanup_async()
            self.current_session = None
