"""start_debugger can run ONE test instead of the whole file.

The gap this closes: `python -m pdb -m pytest <file>` runs every test in the
file in order, and a breakpoint fires in the FIRST test that reaches it. On a
file whose failing test is not first, the model stops inside a PASSING test,
reads correct behaviour, and concludes the code is fine — and nothing in the
observation says which test it is in.

Without selection, a breakpoint inside the function under test can fire in
every earlier passing test before it reaches the failing one.

Node-id (`file.py::test_name`) and `-k` selection are stock pytest; pdb is
orthogonal (it runs pytest as a module and passes argv through), so this is
about which argv the harness builds.

Drives the REAL pdb child via pexpect (no docker), with the same fixture
pattern as test_pdb_observations.
"""

import asyncio
import os
import tempfile
import unittest

from .pdb import PdbHarness, PdbSession, ToolCall


def _tool(name, parameters=None):
    return ToolCall(name, parameters=parameters or {}, reasoning="t")


# `shared.py` is called by every test; the LAST test is the failing one, so a
# breakpoint in `touch` stops three times before reaching it.
SHARED_LIB = (
    "def touch(tag):\n"          # 1
    "    seen = tag\n"           # 2
    "    return seen\n"          # 3
)
MULTI_TEST = (
    "from shared import touch\n"        # 1
    "\n"                               # 2
    "def test_alpha():\n"              # 3
    "    assert touch('alpha') == 'alpha'\n"   # 4
    "\n"                               # 5
    "def test_beta():\n"               # 6
    "    assert touch('beta') == 'beta'\n"     # 7
    "\n"                               # 8
    "def test_gamma():\n"              # 9
    "    assert touch('gamma') == 'gamma'\n"   # 10
    "\n"                               # 11
    "def test_the_failing_one():\n"    # 12
    "    assert touch('omega') == 'NOPE'\n"    # 13
)


class TargetArgsTest(unittest.TestCase):
    """Selection resolution and argv rendering, without spawning anything.

    The mode is decided against the tests the file actually contains, not from
    the string's shape — a name that merely looks like a node id is not one
    unless it is really there.
    """

    COLLECTED = [
        "test_foo",
        "test_foo_extended",
        "TestThing::test_foo",
        "test_param[3-x]",
        "test_settle_levels_a_lopsided_rota",
    ]

    def _resolved(self, selection):
        s = PdbSession("/w/thing_test.py", test_selection=selection)
        s._collected = list(self.COLLECTED)
        self.err = s._resolve_selection()
        return s

    def test_no_selection_runs_the_whole_file(self):
        self.assertEqual(
            self._resolved(None)._pytest_target_args(), ["/w/thing_test.py"]
        )

    def test_blank_selection_is_treated_as_none(self):
        self.assertEqual(
            self._resolved("   ")._pytest_target_args(), ["/w/thing_test.py"]
        )

    def test_an_exact_name_becomes_a_node_id(self):
        self.assertEqual(
            self._resolved("test_foo")._pytest_target_args(),
            ["/w/thing_test.py::test_foo"],
        )
        self.assertIsNone(self.err)

    def test_an_exact_class_method_becomes_a_node_id(self):
        self.assertEqual(
            self._resolved("TestThing::test_foo")._pytest_target_args(),
            ["/w/thing_test.py::TestThing::test_foo"],
        )

    def test_an_exact_parametrised_case_becomes_a_node_id(self):
        self.assertEqual(
            self._resolved("test_param[3-x]")._pytest_target_args(),
            ["/w/thing_test.py::test_param[3-x]"],
        )

    def test_a_partial_name_becomes_a_dash_k_match(self):
        """"lopsided" is what a model reaches for; it must not hard-fail."""
        self.assertEqual(
            self._resolved("lopsided")._pytest_target_args(),
            ["/w/thing_test.py", "-k", "lopsided"],
        )
        self.assertIsNone(self.err)

    def test_an_expression_goes_to_dash_k(self):
        self.assertEqual(
            self._resolved("settle and lopsided")._pytest_target_args(),
            ["/w/thing_test.py", "-k", "settle and lopsided"],
        )

    def test_an_unmatchable_name_is_an_error_listing_the_real_tests(self):
        self._resolved("test_nope")
        self.assertIsNotNone(self.err)
        self.assertIn("test_nope", self.err)
        self.assertIn("test_foo", self.err)

    def test_an_exact_name_wins_over_its_own_prefix_matches(self):
        """test_foo is exact, so it must not also pull in test_foo_extended."""
        self.assertEqual(
            self._resolved("test_foo")._pytest_target_args(),
            ["/w/thing_test.py::test_foo"],
        )


class TargetingSessionTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.dir = self._tmpdir.name
        for name, body in (("shared.py", SHARED_LIB), ("multi_test.py", MULTI_TEST)):
            with open(os.path.join(self.dir, name), "w", encoding="utf-8") as fh:
                fh.write(body)
        self.harness = PdbHarness(working_directory=self.dir)

    def tearDown(self):
        asyncio.run(self.harness.cleanup_async())
        self._tmpdir.cleanup()

    def _start(self, **params):
        params.setdefault("file_path", "multi_test.py")
        return self.harness.execute_tool_call(_tool("start_debugger", params))

    def _break_and_run(self):
        self.harness.execute_tool_call(
            _tool("add_breakpoint", {"breakpoints": [{"file": "shared.py", "lines": [2]}]})
        )
        self.harness.execute_tool_call(_tool("continue_execution"))
        return self.harness.execute_tool_call(_tool("print_variable", {"name": "tag"}))

    def test_without_targeting_the_first_test_wins(self):
        """The trap: the model lands in test_alpha, three tests early."""
        self._start()
        self.assertIn("alpha", self._break_and_run())

    def test_targeting_lands_in_the_requested_test(self):
        self._start(test="test_the_failing_one")
        self.assertIn("omega", self._break_and_run())

    def test_a_dash_k_expression_also_targets(self):
        self._start(test="failing")
        self.assertIn("omega", self._break_and_run())

    def test_start_message_names_the_selection(self):
        out = self._start(test="test_the_failing_one")
        self.assertIn("running only `test_the_failing_one`", out)

    def test_start_message_warns_when_nothing_is_selected(self):
        """Teaches the parameter at the moment it would have helped."""
        out = self._start()
        self.assertIn("running all 4 tests", out)
        self.assertIn('test="<test_name>"', out)
        self.assertIn("FIRST test that reaches it", out)

    def test_a_missing_test_fails_fast_and_names_the_real_ones(self):
        out = self._start(test="test_does_not_exist")
        self.assertTrue(out.lstrip().startswith("Error"), out)
        self.assertIn("test_does_not_exist", out)
        # The model cannot guess the names; give them rather than make it look.
        self.assertIn("test_alpha", out)
        self.assertIn("test_the_failing_one", out)

    def test_an_exact_name_uses_a_node_id_not_a_substring(self):
        """`test_alpha` must not also drag in `test_alpha_two`."""
        with open(os.path.join(self.dir, "multi_test.py"), "a", encoding="utf-8") as fh:
            fh.write("\n\ndef test_alpha_two():\n    assert touch('a2') == 'a2'\n")
        self.harness.execute_tool_call(_tool("start_debugger", {
            "file_path": "multi_test.py", "test": "test_alpha"}))
        self.assertEqual(self.harness.current_session._selection_mode, "node")
        self.assertIn("alpha", self._break_and_run())

    def test_changing_the_selection_restarts_rather_than_no_ops(self):
        """Same file, new selection: restarting in place would keep the previous run."""
        self._start(test="test_alpha")
        self.assertIn("alpha", self._break_and_run())
        self._start(test="test_the_failing_one")
        self.assertIn("omega", self._break_and_run())

    def test_repeating_the_same_selection_still_restarts_in_place(self):
        self._start(test="test_the_failing_one")
        again = self._start(test="test_the_failing_one")
        self.assertIn("restarted", again.lower())

    def test_non_string_selection_is_rejected(self):
        out = self._start(test=["test_alpha"])
        self.assertTrue(out.lstrip().startswith("Error"), out)


if __name__ == "__main__":
    unittest.main()
