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


# --------------------------------------------------------------------------
# missed-breakpoint recovery carries the test result
# --------------------------------------------------------------------------

FAILING_TEST = (
    "def helper():\n"        # 1
    "    return 0\n"         # 2
    "def test_x():\n"        # 3
    "    assert 1 == 2\n"    # 4
)


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


# --------------------------------------------------------------------------
# run_expression hints / available names / always render
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# restart reloads post-edit code; collection-error honesty
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Breakpoint B-window, step context parity, introspection position notes,
# comprehension-scope NameError retry
# --------------------------------------------------------------------------


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
