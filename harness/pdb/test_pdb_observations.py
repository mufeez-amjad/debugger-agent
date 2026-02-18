"""Tests for pdb harness observations.

Covers:
  * echo-free send_command — the pty is spawned with echo=False and
    _strip_command_echo removes any echo that still leaks through textually
    (including readline's `<tail` horizontal-scroll artifact for over-width
    commands).
  * add_breakpoint failures carry a source window ("Code context") — both the
    non-executable-line rejection (offending line marked `! `) and the past-EOF
    rejection (file tail). The marker string
    "not an executable statement inside a function" is kept.
  * continue_execution appends the same source window as step_over when it
    stops at a live position.
  * examine_locals labeled sections ("Local variables:" /
    "Instance attributes of self (vars(self)):"), errors returned raw.
  * run_expression semicolon compounds — leading statements run via pdb `!`,
    the final segment via `p`; _split_top_level_semicolons is quote/bracket
    aware.

These drive the REAL pdb child via pexpect (no docker required), with the
same fixture pattern as test_pdb_regressions.
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


ACCUM_LIB = (
    "class Accumulator:\n"                          # 1
    "    def __init__(self):\n"                     # 2
    "        self._total = 0\n"                     # 3
    "        self._history = []\n"                  # 4
    "\n"                                            # 5
    "    def add(self, value):\n"                   # 6
    "        marker = 'in-add'\n"                   # 7
    "        self._history.append(value)\n"         # 8
    "        self._total += value\n"                # 9
    "        return self._total\n"                  # 10
)
ACCUM_TEST = (
    "from accum import Accumulator\n"               # 1
    "\n"                                            # 2
    "def test_add():\n"                             # 3
    "    acc = Accumulator()\n"                     # 4
    "    acc.add(2)\n"                              # 5
    "    assert acc.add(3) == 5\n"                  # 6
)


class _AccumSessionCase(_HarnessCase):
    """Spins a real pdb session over the Accumulator scenario."""

    def setUp(self):
        super().setUp()
        self._write("accum.py", ACCUM_LIB)
        self._write("test_accum.py", ACCUM_TEST)
        out = self.harness.execute_tool_call(
            _tool("start_debugger", {"file_path": "test_accum.py"})
        )
        self.assertNotIn("Error", out)


# --------------------------------------------------------------------------
# echo-free send_command
# --------------------------------------------------------------------------

class TestStripCommandEcho(unittest.TestCase):
    """Unit tests for PdbSession._strip_command_echo."""

    def test_plain_echo_first_line_removed(self):
        out = PdbSession._strip_command_echo("pp locals()", "pp locals()\n{'x': 1}")
        self.assertEqual(out, "{'x': 1}")

    def test_readline_scroll_artifact_removed(self):
        # readline redisplays an over-width command as `<tail-of-command`.
        cmd = "p [record for record in self._trail if record.kind == 'edited']"
        out = PdbSession._strip_command_echo(cmd, "<dited']\n[]")
        self.assertEqual(out, "[]")

    def test_genuine_output_not_eaten(self):
        out = PdbSession._strip_command_echo("p x", "42\nmore")
        self.assertEqual(out, "42\nmore")

    def test_first_line_only_is_considered(self):
        # An echo-looking line deeper in the output must survive.
        out = PdbSession._strip_command_echo("p x", "1\np x\n2")
        self.assertEqual(out, "1\np x\n2")

    def test_empty_passthrough(self):
        self.assertEqual(PdbSession._strip_command_echo("p x", ""), "")
        self.assertEqual(PdbSession._strip_command_echo("", "out"), "out")

    def test_short_angle_line_not_treated_as_artifact(self):
        # `<` lines must be >4 chars and a substring of the command.
        out = PdbSession._strip_command_echo("p x", "<x>\nvalue")
        self.assertEqual(out, "<x>\nvalue")


class TestEchoFreeSession(_AccumSessionCase):
    def test_send_command_reply_has_no_echo(self):
        session = self.harness.current_session
        out = session.send_command("p 40 + 2")
        self.assertIn("42", out)
        first_line = out.splitlines()[0].strip() if out.splitlines() else ""
        self.assertNotEqual(first_line, "p 40 + 2")

    def test_examine_locals_reply_has_no_echoed_pp_command(self):
        self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "accum.py", "lines": [8]}]})
        )
        self.harness.execute_tool_call(_tool("continue_execution"))
        result = self.harness.execute_tool_call(_tool("examine_locals"))
        self.assertNotIn("pp locals()", result)


# --------------------------------------------------------------------------
# add_breakpoint failure source windows
# --------------------------------------------------------------------------

class TestAddBreakpointFailureContext(_AccumSessionCase):
    def test_non_executable_line_error_has_marked_context(self):
        # Line 5 of accum.py is blank — not an executable statement.
        result = self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "accum.py", "lines": [5]}]})
        )
        self.assertIn("Error", result)
        # The marker string stays verbatim.
        self.assertIn("not an executable statement inside a function", result)
        # Source window with the rejected line marked `! `.
        self.assertIn("Code context (accum.py):", result)
        self.assertIn("!     5", result)
        self.assertIn("def add(self, value):", result)

    def test_past_eof_error_has_file_tail_context(self):
        result = self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "accum.py", "lines": [999]}]})
        )
        self.assertIn("beyond end of file", result)
        self.assertIn("No breakpoint was set", result)
        # The tail of the file is shown so the model sees what IS there.
        self.assertIn("Code context (accum.py):", result)
        self.assertIn("return self._total", result)


# --------------------------------------------------------------------------
# continue_execution position context
# --------------------------------------------------------------------------

class TestContinuePositionContext(_AccumSessionCase):
    def test_continue_to_breakpoint_appends_code_context(self):
        self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "accum.py", "lines": [9]}]})
        )
        result = self.harness.execute_tool_call(_tool("continue_execution"))
        # Stopped at a live position -> same source window step_over shows.
        self.assertIn("Code context (accum.py):", result)
        self.assertIn("->", result)
        self.assertIn("self._total += value", result)

    def test_step_over_and_continue_share_window_format(self):
        self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "accum.py", "lines": [8]}]})
        )
        cont = self.harness.execute_tool_call(_tool("continue_execution"))
        step = self.harness.execute_tool_call(_tool("step_over"))
        self.assertIn("Code context (accum.py):", cont)
        self.assertIn("Code context (accum.py):", step)


# --------------------------------------------------------------------------
# examine_locals labeled sections
# --------------------------------------------------------------------------

class TestExamineLocalsSections(_AccumSessionCase):
    def test_sections_present_when_self_in_scope(self):
        self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "accum.py", "lines": [8]}]})
        )
        self.harness.execute_tool_call(_tool("continue_execution"))
        result = self.harness.execute_tool_call(_tool("examine_locals"))
        self.assertIn("Local variables:", result)
        self.assertIn("marker", result)
        self.assertIn("Instance attributes of self (vars(self)):", result)
        self.assertIn("_history", result)
        # No raw pdb separator appears.
        self.assertNotIn("--- self.__dict__ ---", result)

    def test_no_self_section_at_module_level_frame(self):
        # The pytest runner's frame at session start has no `self`.
        result = self.harness.execute_tool_call(_tool("examine_locals"))
        self.assertNotIn("Instance attributes of self", result)


class TestPdbReplyErrored(unittest.TestCase):
    """Unit tests for the error-passthrough helper: errors are returned raw
    (never wrapped in a 'Local variables:' section header)."""

    def test_pdb_star_error_is_error(self):
        self.assertTrue(PdbHarness._pdb_reply_errored("*** NameError: name 'q'"))

    def test_harness_error_is_error(self):
        self.assertTrue(PdbHarness._pdb_reply_errored("Error: no session"))

    def test_no_session_is_error(self):
        self.assertTrue(PdbHarness._pdb_reply_errored("No active debugging session"))

    def test_empty_is_error(self):
        self.assertTrue(PdbHarness._pdb_reply_errored(""))
        self.assertTrue(PdbHarness._pdb_reply_errored("   \n"))

    def test_value_output_is_not_error(self):
        self.assertFalse(PdbHarness._pdb_reply_errored("{'x': 1, 'self': <A object>}"))


# --------------------------------------------------------------------------
# run_expression semicolon compounds
# --------------------------------------------------------------------------

class TestSplitTopLevelSemicolons(unittest.TestCase):
    def test_simple_compound_splits(self):
        self.assertEqual(
            PdbHarness._split_top_level_semicolons("x = 1; x + 1"),
            ["x = 1", "x + 1"],
        )

    def test_quoted_semicolon_not_split(self):
        self.assertEqual(
            PdbHarness._split_top_level_semicolons('s = "a;b"; s'),
            ['s = "a;b"', "s"],
        )

    def test_single_quoted_semicolon_not_split(self):
        self.assertEqual(
            PdbHarness._split_top_level_semicolons("s = 'a;b'"),
            ["s = 'a;b'"],
        )

    def test_bracketed_semicolon_not_split(self):
        self.assertEqual(
            PdbHarness._split_top_level_semicolons("f('x;y'); g([1, 2])"),
            ["f('x;y')", "g([1, 2])"],
        )

    def test_escaped_quote_inside_string(self):
        self.assertEqual(
            PdbHarness._split_top_level_semicolons("s = 'a\\';b'; s"),
            ["s = 'a\\';b'", "s"],
        )

    def test_no_semicolon_is_single_segment(self):
        self.assertEqual(
            PdbHarness._split_top_level_semicolons("value + 1"),
            ["value + 1"],
        )

    def test_empty_segments_dropped(self):
        self.assertEqual(
            PdbHarness._split_top_level_semicolons("x = 1;; x ;"),
            ["x = 1", "x"],
        )


class TestRunExpressionCompound(_AccumSessionCase):
    def _stop_in_add(self):
        self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "accum.py", "lines": [9]}]})
        )
        self.harness.execute_tool_call(_tool("continue_execution"))

    def _run(self, expression):
        return self.harness.execute_tool_call(
            _tool("run_expression", {"expression": expression})
        )

    def test_setup_statement_then_expression(self):
        self._stop_in_add()
        result = self._run("tmp_probe = 40; tmp_probe + 2")
        self.assertIn("42", result)
        self.assertNotIn("SyntaxError", result)

    def test_setup_output_prepended(self):
        self._stop_in_add()
        result = self._run("print('setup-mark'); 1 + 1")
        self.assertIn("setup-mark", result)
        self.assertIn("2", result)

    def test_quoted_semicolon_expression_not_split(self):
        self._stop_in_add()
        result = self._run("'a;b'")
        self.assertIn("a;b", result)
        self.assertNotIn("SyntaxError", result)

    def test_setup_error_reported_with_statement(self):
        self._stop_in_add()
        result = self._run("zz_undefined_xyz += 1; value")
        self.assertIn("Error while executing statement(s)", result)
        self.assertIn("zz_undefined_xyz", result)

    def test_single_expression_unchanged(self):
        self._stop_in_add()
        result = self._run("value * 10")
        self.assertIn("20", result)  # first add(2) hits the bp with value=2


if __name__ == "__main__":
    unittest.main()
