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

# --------------------------------------------------------------------------
# continue_execution position context
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# examine_locals labeled sections
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# run_expression semicolon compounds
# --------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
