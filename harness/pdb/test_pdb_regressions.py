"""Regression tests for the pdb harness.

Covers:
  * remove_breakpoint clears by tracked bp number / file:line, honest errors
  * breakpoint resolver bounds check (past-EOF) + loud vs quiet snap notes
  * worker-thread breakpoints fire under `pytest --trace` (thread tracing)
  * missed-breakpoint recovery carries the test result + says it restarted
  * `--Return--` reported as a frame return, not program completion
  * list_code file header + file arg; cross-file breakpoint resolved path
  * run_expression import hint / available-names / always-render
  * start_debugger restart reloads post-edit code; collection-error honesty

These drive the REAL pdb/pytest child via pexpect (no docker required).
"""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from .pdb import PdbHarness, PdbSession, ToolCall


def _tool(name, parameters=None):
    return ToolCall(name, parameters=parameters or {}, reasoning="t")


class _HarnessCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.dir = self._tmpdir.name
        self.harness = PdbHarness(working_directory=self.dir)

    def tearDown(self):
        asyncio.run(self.harness.cleanup_async())
        self._tmpdir.cleanup()

    def _write(self, name, content):
        path = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path


# --------------------------------------------------------------------------
# remove_breakpoint
# --------------------------------------------------------------------------

COUNTER_LIB = (
    "class Counter:\n"                 # 1
    "    def __init__(self):\n"        # 2
    "        self.values = {}\n"       # 3
    "    def bump(self, key):\n"       # 4
    "        self.values[key] = self.values.get(key, 0) + 1\n"  # 5
    "        return self.values[key]\n"  # 6
)
COUNTER_TEST = (
    "from counter import Counter\n"
    "def test_bump():\n"
    "    c = Counter()\n"
    "    assert c.bump('a') == 1\n"
)


class TestRemoveBreakpoint(_HarnessCase):
    def setUp(self):
        super().setUp()
        self._write("counter.py", COUNTER_LIB)
        self._write("test_counter.py", COUNTER_TEST)
        self.harness.execute_tool_call(_tool("start_debugger", {"file_path": "test_counter.py"}))

    def _add(self, line):
        return self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "counter.py", "lines": [line]}]})
        )

    def test_remove_reports_success_only_on_confirmed_delete(self):
        self._add(5)
        session = self.harness.current_session
        self.assertTrue(any(ln == 5 for _f, ln in session.breakpoints))
        result = self.harness.execute_tool_call(
            _tool("remove_breakpoint", {"breakpoints": [{"file": "counter.py", "lines": [5]}]})
        )
        self.assertIn("Removed breakpoint at counter.py:5", result)
        # No leftover "out of range" wrapped as success, and tracking pruned.
        self.assertNotIn("out of range", result)
        self.assertFalse(any(ln == 5 for _f, ln in session.breakpoints))
        self.assertNotIn(5, [ln for _f, ln in session.breakpoint_numbers.values()])

    def test_remove_actually_clears_so_continue_does_not_stop(self):
        self._add(5)
        self.harness.execute_tool_call(
            _tool("remove_breakpoint", {"breakpoints": [{"file": "counter.py", "lines": [5]}]})
        )
        result = self.harness.execute_tool_call(_tool("continue_execution"))
        # With the only breakpoint gone, the program runs to completion instead
        # of stopping at line 5.
        self.assertNotIn("(5)bump", result)

    def test_remove_nonexistent_is_honest_error_not_success(self):
        # No breakpoint at line 6 — must NOT claim "Removed".
        result = self.harness.execute_tool_call(
            _tool("remove_breakpoint", {"breakpoints": [{"file": "counter.py", "lines": [6]}]})
        )
        self.assertNotIn("Removed breakpoint", result)
        self.assertIn("Error", result)


# --------------------------------------------------------------------------
# resolver bounds check + snap classification
# --------------------------------------------------------------------------

FAR_MODULE_PROGRAM = (
    "X = 1\n"          # 1
    "Y = 2\n"          # 2
    "Z = 3\n"          # 3
    "W = 4\n"          # 4
    "V = 5\n"          # 5
    "U = 6\n"          # 6
    "T = 7\n"          # 7
    "def f():\n"       # 8
    "    a = 1\n"      # 9
    "    return a\n"   # 10
)


class TestResolverBounds(_HarnessCase):
    def test_past_eof_returns_error_and_sets_no_breakpoint(self):
        self._write("counter.py", COUNTER_LIB)
        self._write("test_counter.py", COUNTER_TEST)
        self.harness.execute_tool_call(_tool("start_debugger", {"file_path": "test_counter.py"}))
        result = self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "counter.py", "lines": [99]}]})
        )
        self.assertIn("beyond end of file", result)
        self.assertIn("6 lines", result)
        self.assertNotIn("Breakpoint set", result)
        # Nothing tracked.
        self.assertEqual(self.harness.current_session.breakpoints, [])

    def test_module_level_line_is_rejected(self):
        # A module-level line (import-time execution, never hit during the
        # test's call) is rejected outright, with no replacement-line
        # suggestion.
        self.harness.start_session(FAR_MODULE_PROGRAM, "prog.py")
        result = self.harness.execute_tool_call(_tool("add_breakpoint", {"line": 1}))
        self.assertIn("not an executable statement inside a function", result)
        self.assertIn("No breakpoint was set", result)
        self.assertNotIn("Breakpoint set", result)
        self.assertNotIn("line 9", result)

    def test_blank_line_is_rejected(self):
        # A blank line inside a function is likewise rejected, not nudged.
        prog = (
            "def g():\n"       # 1
            "    a = 1\n"      # 2
            "\n"              # 3 (blank, inside g)
            "    b = 2\n"      # 4
            "    return b\n"   # 5
        )
        self.harness.start_session(prog, "prog.py")
        result = self.harness.execute_tool_call(_tool("add_breakpoint", {"line": 3}))
        self.assertIn("not an executable statement inside a function", result)
        self.assertIn("No breakpoint was set", result)
        self.assertNotIn("Breakpoint set", result)


# --------------------------------------------------------------------------
# worker-thread breakpoints
# --------------------------------------------------------------------------

WORKER_LIB = (
    "from concurrent.futures import ThreadPoolExecutor\n"  # 1
    "\n"                                                    # 2
    "def compute(x):\n"                                     # 3
    "    doubled = x * 2\n"                                 # 4
    "    return doubled + 1\n"                              # 5
    "\n"                                                    # 6
    "def run_all(vals):\n"                                  # 7
    "    with ThreadPoolExecutor(max_workers=2) as ex:\n"   # 8
    "        return list(ex.map(compute, vals))\n"          # 9
)
WORKER_TEST = (
    "from worker_lib import run_all\n"
    "def test_run_all():\n"
    "    assert run_all([1, 2, 3]) == [3, 5, 7]\n"
)


class TestWorkerThreadTracing(_HarnessCase):
    def setUp(self):
        super().setUp()
        self._write("worker_lib.py", WORKER_LIB)
        self._write("test_worker.py", WORKER_TEST)

    def test_worker_thread_breakpoint_fires_by_default(self):
        self.harness.execute_tool_call(_tool("start_debugger", {"file_path": "test_worker.py"}))
        self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "worker_lib.py", "lines": [4]}]})
        )
        result = self.harness.execute_tool_call(_tool("continue_execution"))
        self.assertNotIn("WITHOUT stopping", result)
        self.assertTrue(self.harness.current_session.is_alive())
        self.assertEqual(self.harness.current_session.current_position["line"], 4)

    def test_worker_thread_breakpoint_missed_when_disabled(self):
        # With thread tracing off, the worker breakpoint is missed and the
        # recovery diagnostic names the worker-thread cause.
        self.harness.start_session  # ensure attribute exists
        os.environ["LLM_DEBUGGER_THREAD_TRACE"] = "0"
        try:
            self.harness.execute_tool_call(_tool("start_debugger", {"file_path": "test_worker.py"}))
            self.assertFalse(self.harness.current_session.thread_tracing_enabled)
            self.harness.execute_tool_call(
                _tool("add_breakpoint", {"breakpoints": [{"file": "worker_lib.py", "lines": [4]}]})
            )
            result = self.harness.execute_tool_call(_tool("continue_execution"))
            self.assertIn("WITHOUT stopping", result)
            self.assertIn("worker thread", result)
        finally:
            os.environ.pop("LLM_DEBUGGER_THREAD_TRACE", None)


# --------------------------------------------------------------------------
# missed-breakpoint recovery carries the test result
# --------------------------------------------------------------------------

FAILING_TEST = (
    "def helper():\n"        # 1
    "    return 0\n"         # 2
    "def test_x():\n"        # 3
    "    assert 1 == 2\n"    # 4
)


class TestMissedBreakpointRecoveryDetail(_HarnessCase):
    def test_recovery_includes_test_result_and_restart_note(self):
        self._write("test_fail.py", FAILING_TEST)
        self.harness.execute_tool_call(_tool("start_debugger", {"file_path": "test_fail.py"}))
        # Breakpoint inside helper(), which the failing test never calls.
        self.harness.execute_tool_call(_tool("add_breakpoint", {"line": 2}))
        result = self.harness.execute_tool_call(_tool("continue_execution"))
        self.assertIn("WITHOUT stopping", result)
        # The full PASSED/FAILED + assertion output is surfaced, not discarded.
        self.assertIn("Test result from that run", result)
        self.assertTrue("assert" in result or "FAILED" in result or "1 == 2" in result)
        # And it says the session was restarted.
        self.assertIn("RESTARTED", result)
        self.assertTrue(self.harness.current_session.is_alive())


# --------------------------------------------------------------------------
# --Return-- is a frame return, not program completion
# --------------------------------------------------------------------------

NESTED_PROGRAM = (
    "def inner():\n"         # 1
    "    v = 5\n"            # 2
    "    return v\n"         # 3
    "def outer():\n"         # 4
    "    r = inner()\n"      # 5
    "    return r\n"         # 6
    "print(outer())\n"       # 7
)


class TestFrameReturn(_HarnessCase):
    def test_step_out_reports_frame_return_not_completion(self):
        self.harness.start_session(NESTED_PROGRAM, "prog.py")
        self.harness.execute_tool_call(_tool("add_breakpoint", {"line": 2}))
        self.harness.execute_tool_call(_tool("continue_execution"))
        result = self.harness.execute_tool_call(_tool("step_out"))
        self.assertNotIn("Program execution completed", result)
        self.assertIn("Returned from inner", result)
        # Execution has not ended.
        self.assertTrue(self.harness.current_session.is_alive())

    def test_frame_return_carries_code_context_window(self):
        # The frame-return text drops pdb's `> file(line)` marker. A return
        # event must still carry a numbered source window, like every other
        # step.
        self.harness.start_session(NESTED_PROGRAM, "prog.py")
        self.harness.execute_tool_call(_tool("add_breakpoint", {"line": 2}))
        self.harness.execute_tool_call(_tool("continue_execution"))
        result = self.harness.execute_tool_call(_tool("step_out"))
        self.assertIn("Returned from inner", result)
        self.assertIn("Code context (prog.py):", result)
        self.assertIn("->", result)


# --------------------------------------------------------------------------
# list_code file header / file arg; cross-file breakpoint path
# --------------------------------------------------------------------------

SUB_LIB = (
    "def load(name):\n"          # 1
    "    data = {'k': name}\n"   # 2
    "    return data['k']\n"     # 3
)
SUB_TEST = (
    "from sub.loader import load\n"
    "def test_load():\n"
    "    assert load('x') == 'x'\n"
)


class TestListCodeAndCrossFile(_HarnessCase):
    def test_list_code_has_file_header(self):
        self.harness.start_session(NESTED_PROGRAM, "prog.py")
        result = self.harness.execute_tool_call(_tool("list_code", {"start": 1, "end": 4}))
        self.assertIn("File:", result)
        self.assertIn("prog.py", result)

    def test_list_code_file_arg_reads_fresh_from_disk(self):
        self.harness.start_session(NESTED_PROGRAM, "prog.py")
        other = self._write("other.py", "ALPHA = 1\nBETA = 2\nGAMMA = 3\n")
        result = self.harness.execute_tool_call(
            _tool("list_code", {"file": "other.py", "start": 1, "end": 3})
        )
        self.assertIn("File:", result)
        self.assertIn("other.py", result)
        self.assertIn("ALPHA = 1", result)
        self.assertIn("GAMMA = 3", result)
        # Nothing from the debugged file leaked in.
        self.assertNotIn("def inner", result)

    def test_list_code_file_arg_past_eof_errors(self):
        self.harness.start_session(NESTED_PROGRAM, "prog.py")
        self._write("other.py", "ALPHA = 1\n")
        result = self.harness.execute_tool_call(
            _tool("list_code", {"file": "other.py", "start": 50})
        )
        self.assertIn("beyond end of file", result)

    def test_cross_file_breakpoint_uses_resolved_path(self):
        # 'sub/loader.py' would fail "not found from sys.path" with the raw arg;
        # the resolved absolute path makes it succeed and get hit.
        self._write(os.path.join("sub", "loader.py"), SUB_LIB)
        self._write(os.path.join("sub", "__init__.py"), "")
        self._write("test_load.py", SUB_TEST)
        self.harness.execute_tool_call(_tool("start_debugger", {"file_path": "test_load.py"}))
        result = self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "sub/loader.py", "lines": [2]}]})
        )
        self.assertIn("Breakpoint set", result)
        self.assertNotIn("not found", result)
        cont = self.harness.execute_tool_call(_tool("continue_execution"))
        self.assertNotIn("WITHOUT stopping", cont)
        self.assertEqual(self.harness.current_session.current_position["line"], 2)


# --------------------------------------------------------------------------
# run_expression hints / available names / always render
# --------------------------------------------------------------------------

class TestRunExpression(_HarnessCase):
    SIMPLE = "def add(a, b):\n    total = a + b\n    return total\nprint(add(2, 3))\n"

    def test_import_statement_gets_actionable_alternative(self):
        self.harness.start_session(self.SIMPLE, "prog.py")
        result = self.harness.execute_tool_call(
            _tool("run_expression", {"expression": "import os; os.getcwd()"})
        )
        self.assertIn("__import__", result)
        self.assertIn("bash tool", result)

    def test_shell_command_still_rejected(self):
        self.harness.start_session(self.SIMPLE, "prog.py")
        result = self.harness.execute_tool_call(
            _tool("run_expression", {"expression": "pytest -xvs test_prog.py"})
        )
        self.assertIn("bash tool", result)

    def test_nameerror_lists_available_names(self):
        self.harness.start_session(self.SIMPLE, "prog.py")
        # Break inside add() so a/b/total are in scope, then probe a bad name.
        self.harness.execute_tool_call(_tool("add_breakpoint", {"line": 2}))
        self.harness.execute_tool_call(_tool("continue_execution"))
        result = self.harness.execute_tool_call(
            _tool("run_expression", {"expression": "nonexistent_var"})
        )
        self.assertIn("NameError", result)
        self.assertIn("Available names", result)
        self.assertIn("a", result)

    def test_valid_expression_renders_value(self):
        self.harness.start_session(self.SIMPLE, "prog.py")
        result = self.harness.execute_tool_call(_tool("run_expression", {"expression": "1 + 1"}))
        self.assertIn("2", result)


# --------------------------------------------------------------------------
# restart reloads post-edit code; collection-error honesty
# --------------------------------------------------------------------------

class TestStartDebuggerReloadAndCollection(_HarnessCase):
    def test_restart_reloads_edited_code(self):
        self.harness.start_session("msg = 'old_value'\nprint(msg)\n", "prog.py")
        self.assertTrue(self.harness.current_session.is_alive())
        # Edit the file on disk, then restart via start_debugger.
        self._write("prog.py", "msg = 'new_value'\nprint(msg)\n")
        self.harness.execute_tool_call(_tool("start_debugger", {"file_path": "prog.py"}))
        result = self.harness.execute_tool_call(_tool("list_code", {"start": 1, "end": 2}))
        self.assertIn("new_value", result)
        self.assertNotIn("old_value", result)

    def test_collection_error_returns_honest_error(self):
        # A test file with a syntax error can't be collected; the message must
        # NOT claim the session started.
        self._write("test_broken.py", "def test_x(:\n    pass\n")
        result = self.harness.execute_tool_call(_tool("start_debugger", {"file_path": "test_broken.py"}))
        self.assertIn("Error", result)
        self.assertNotIn("Debugging session started", result)
        self.assertFalse(
            self.harness.current_session is not None
            and self.harness.current_session.is_alive()
        )


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# Fixture setup: sessions start BEFORE the pytest run (`-m pdb -m pytest`), so
# breakpoints in code executed by @pytest.fixture setup fire. Under
# `pytest --trace` the first prompt would appear only after fixture setup, and
# such breakpoints could never hit.
# --------------------------------------------------------------------------

FIXTURE_LIB = (
    "class Engine:\n"                      # 1
    "    def settle(self, vals):\n"        # 2
    "        total = sum(vals)\n"          # 3
    "        if total > 99:\n"             # 4
    "            total = 99\n"             # 5  (dead branch for these inputs)
    "        return total - 1\n"           # 6  (buggy)
)
FIXTURE_TEST = (
    "import pytest\n"                          # 1
    "from fixture_lib import Engine\n"         # 2
    "@pytest.fixture\n"                        # 3
    "def result():\n"                          # 4
    "    return Engine().settle([1, 2, 3])\n"  # 5
    "def test_settle(result):\n"               # 6
    "    assert result == 6\n"                 # 7
)


class TestFixtureBreakpointsFire(_HarnessCase):
    def setUp(self):
        super().setUp()
        self._write("fixture_lib.py", FIXTURE_LIB)
        self._write("test_fixture.py", FIXTURE_TEST)
        self.harness.execute_tool_call(
            _tool("start_debugger", {"file_path": "test_fixture.py"})
        )

    def test_breakpoint_in_fixture_executed_code_is_hit(self):
        # Breakpoint inside Engine.settle — executed ONLY from the fixture
        # (test bodies are bare asserts). Must stop there, with the frame's
        # locals inspectable.
        self.harness.execute_tool_call(
            _tool(
                "add_breakpoint",
                {"breakpoints": [{"file": "fixture_lib.py", "lines": [3]}]},
            )
        )
        result = self.harness.execute_tool_call(_tool("continue_execution"))
        self.assertNotIn("WITHOUT stopping", result)
        self.assertIn("fixture_lib.py", result)
        locals_result = self.harness.execute_tool_call(_tool("examine_locals"))
        self.assertIn("vals", locals_result)

    def test_missed_bp_on_dead_branch_recovers_without_blackout_claim(self):
        # Line 5 never executes for these inputs — a genuine miss. The
        # diagnostic must NOT claim fixture code is unbreakpointable (it no
        # longer is), and the session must survive with breakpoints kept.
        self.harness.execute_tool_call(
            _tool(
                "add_breakpoint",
                {"breakpoints": [{"file": "fixture_lib.py", "lines": [5]}]},
            )
        )
        result = self.harness.execute_tool_call(_tool("continue_execution"))
        self.assertIn("WITHOUT stopping", result)
        self.assertNotIn("NEVER fire", result)
        self.assertIn("RESTARTED", result)
        self.assertIn("breakpoints still applied", result)
        session = self.harness.current_session
        self.assertTrue(session.is_alive())
        self.assertTrue(any(ln == 5 for _f, ln in session.breakpoints))

    def test_startup_message_states_prerun_pause(self):
        # Fresh session (setUp already started one; restart on same target).
        result = self.harness.execute_tool_call(
            _tool("start_debugger", {"file_path": "test_fixture.py"})
        )
        self.assertIn("BEFORE the test run", result)


# --------------------------------------------------------------------------
# Breakpoint B-window, step context parity, introspection position notes,
# comprehension-scope NameError retry
# --------------------------------------------------------------------------


class TestObservationFixes(_HarnessCase):
    def setUp(self):
        super().setUp()
        self._write("counter.py", COUNTER_LIB)
        self._write("test_counter.py", COUNTER_TEST)
        self.harness.execute_tool_call(
            _tool("start_debugger", {"file_path": "test_counter.py"})
        )

    def test_breakpoint_set_shows_b_marked_window(self):
        result = self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "counter.py", "lines": [5]}]})
        )
        self.assertIn("Line 5: Breakpoint set", result)
        self.assertIn("Code context (counter.py):", result)
        self.assertIn("B     5", result)

    def test_step_into_and_step_out_show_code_context(self):
        # step_over already appended a source window; step_into/step_out were
        # bare pdb one-liners for no reason.
        self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "counter.py", "lines": [5]}]})
        )
        self.harness.execute_tool_call(_tool("continue_execution"))
        result = self.harness.execute_tool_call(_tool("step_into"))
        self.assertIn("Code context", result)
        self.assertIn("->", result)

    def test_failed_introspection_reports_current_position(self):
        self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "counter.py", "lines": [5]}]})
        )
        self.harness.execute_tool_call(_tool("continue_execution"))
        # Missing attribute: AttributeError + position suffix.
        result = self.harness.execute_tool_call(
            _tool("run_expression", {"expression": "self._slotz"})
        )
        self.assertIn("AttributeError", result)
        self.assertIn("Current position: counter.py:5", result)
        # Missing name: available-names note + position suffix.
        result = self.harness.execute_tool_call(
            _tool("run_expression", {"expression": "[z for z in nonexistent_name]"})
        )
        self.assertIn("NameError", result)
        self.assertIn("Current position: counter.py:5", result)

    def test_comprehension_over_frame_local_evaluates(self):
        # On pre-PEP-709 Pythons (such as 3.11) pdb's `p` gives a
        # comprehension body its own scope that can't see frame locals like
        # `self` — the harness retries with a merged namespace so the model
        # gets the value either way. On 3.12+ the direct path already works.
        self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "counter.py", "lines": [5]}]})
        )
        self.harness.execute_tool_call(_tool("continue_execution"))
        result = self.harness.execute_tool_call(
            _tool(
                "run_expression",
                {"expression": "[(k, v) for k, v in self.values.items()]"},
            )
        )
        self.assertNotIn("NameError", result)

    def test_comprehension_retry_uses_merged_namespace_on_nameerror(self):
        # Simulate the 3.11 failure regardless of the local interpreter: the
        # first `p` returns the comprehension-scope NameError, the merged-eval
        # retry returns the value.
        calls = []

        class _FakeSession:
            current_position = {"file": "counter.py", "line": 5}

            def send_command(self, cmd, timeout=None):
                calls.append(cmd)
                if cmd.startswith("p "):
                    return "*** NameError: name 'self' is not defined"
                if cmd.startswith("!print(repr(eval("):
                    return "[('a', 1)]"
                return ""

        real_session = self.harness.current_session
        try:
            self.harness.current_session = _FakeSession()
            result = self.harness.execute_tool_call(
                _tool(
                    "run_expression",
                    {"expression": "[(k, self.values[k]) for k in self.values]"},
                )
            )
        finally:
            self.harness.current_session = real_session
        self.assertEqual(result, "[('a', 1)]")
        self.assertTrue(any(c.startswith("!print(repr(eval(") for c in calls))


# --------------------------------------------------------------------------
# Bare list_code before continue_execution must not leak
# Python bootstrap internals (mirrors the start() banner suppression).
# --------------------------------------------------------------------------

PRERUN_LIB = (
    "def f():\n"
    "    return 1\n"
)
PRERUN_TEST = (
    "from prerun_lib import f\n"
    "def test_f():\n"
    "    assert f() == 1\n"
)


class TestListCodePreRunGuard(_HarnessCase):
    def test_bare_list_code_before_continue_returns_guidance(self):
        self._write("prerun_lib.py", PRERUN_LIB)
        self._write("prerun_test.py", PRERUN_TEST)
        res = self.harness.execute_tool_call(
            _tool("start_debugger", {"file_path": "prerun_test.py"})
        )
        self.assertIn("paused BEFORE the test run", res)

        out = self.harness.execute_tool_call(_tool("list_code", {}))
        self.assertIn("interpreter", out)
        self.assertNotIn("The pytest entry point", out)

    def test_list_code_with_file_arg_still_reads_from_disk_pre_run(self):
        self._write("prerun_lib.py", PRERUN_LIB)
        self._write("prerun_test.py", PRERUN_TEST)
        self.harness.execute_tool_call(
            _tool("start_debugger", {"file_path": "prerun_test.py"})
        )
        out = self.harness.execute_tool_call(
            _tool("list_code", {"file": "prerun_lib.py", "start": 1, "end": 2})
        )
        self.assertIn("def f", out)


# --------------------------------------------------------------------------
# NameError('self') inside a comprehension frame gets accurate guidance (not a
# "program may have finished" note).
# --------------------------------------------------------------------------

GENEXPR_LIB = (
    "class Fleet:\n"                                          # 1
    "    def __init__(self):\n"                               # 2
    "        self._summaries = [1, 2, 3]\n"                   # 3
    "    def total(self):\n"                                  # 4
    "        return sum(s for s in self._summaries)\n"        # 5
)
GENEXPR_TEST = (
    "from genexpr_lib import Fleet\n"
    "def test_total():\n"
    "    assert Fleet().total() == 6\n"
)


class TestComprehensionFrameNameError(_HarnessCase):
    def test_self_nameerror_in_genexpr_frame_gets_comprehension_note(self):
        self._write("genexpr_lib.py", GENEXPR_LIB)
        self._write("genexpr_test.py", GENEXPR_TEST)
        self.harness.execute_tool_call(
            _tool("start_debugger", {"file_path": "genexpr_test.py"})
        )
        self.harness.execute_tool_call(
            _tool("add_breakpoint",
                  {"breakpoints": [{"file": "genexpr_lib.py", "lines": [5]}]})
        )
        self.harness.execute_tool_call(_tool("continue_execution", {}))
        # Step from the method line into the <genexpr> frame.
        out = ""
        for _ in range(4):
            out = self.harness.execute_tool_call(_tool("step_into", {}))
            if "<genexpr>" in out:
                break
        self.assertIn("<genexpr>", out)
        res = self.harness.execute_tool_call(
            _tool("run_expression",
                  {"expression": "[s for s in self._summaries]"})
        )
        self.assertIn("NameError", res)
        self.assertIn("comprehension", res)
        self.assertNotIn("program may have finished", res)
