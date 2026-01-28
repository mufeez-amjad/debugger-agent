"""Tests for the FileInterfaceHarness capabilities:

- line-anchored edit (there is no read_file tool; bash owns reads, and the
  tests below pin its absence)
- syntax validation on edits/writes
- closest-match inline no-match error
- snapshot + revert
- small whole-file-rewrite hint
"""
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.file_interface import FileInterfaceHarness
from common.models import ToolCall


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.h = FileInterfaceHarness(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _w(self, name, content):
        p = Path(self.tmp) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def _r(self, name):
        return (Path(self.tmp) / name).read_text()

    def _edit(self, name, old, new, **extra):
        params = {"file_path": name, "old_string": old, "new_string": new}
        params.update(extra)
        return self.h._execute_edit_file(
            ToolCall(name="edit_file", parameters=params, reasoning="")
        )

    def _write(self, name, content):
        return self.h._execute_write_file(
            ToolCall(
                name="write_file",
                parameters={"file_path": name, "content": content},
                reasoning="",
            )
        )


class TestAnchoredEdit(_Base):
    def test_anchor_restricts_to_window(self):
        # Two identical 'a=0' blocks separated by >40 filler lines. block1 is at
        # lines 1-3; block2 at lines 64-66. An anchor of 65 restricts the search
        # to lines 45-85, so ONLY block2 is edited and block1 is left untouched.
        block1 = "a=0\na=0\na=0"
        filler = "\n".join(f"f{i}=0" for i in range(60))  # lines 4-63
        block2 = "a=0\na=0\na=0"                            # lines 64-66
        self._w("dup.py", block1 + "\n" + filler + "\n" + block2 + "\n")
        res = self._edit("dup.py", "a=0", "a=9", anchor_line=65)
        self.assertIn("Successfully replaced", res)
        content = self._r("dup.py")
        lines = content.splitlines()
        self.assertEqual(lines[0:3], ["a=0", "a=0", "a=0"])   # block1 untouched
        self.assertEqual(lines[63:66], ["a=9", "a=9", "a=9"])  # block2 edited

    def test_no_anchor_is_backward_compatible(self):
        self._w("p.py", "x = 1\ny = 2\n")
        res = self._edit("p.py", "x = 1", "x = 3")
        self.assertIn("Successfully replaced", res)
        self.assertIn("x = 3", self._r("p.py"))

    def test_anchor_out_of_window_reports_restriction(self):
        self._w("p.py", "\n".join(f"v{i} = {i}" for i in range(1, 100)) + "\n")
        # value v90 exists but the anchor window is around line 5.
        res = self._edit("p.py", "v90 = 90", "v90 = 0", anchor_line=5)
        self.assertIn("String not found", res)
        self.assertIn("anchor_line window", res)


class TestSyntaxValidation(_Base):
    def test_edit_that_breaks_parse_is_rejected(self):
        self._w("m.py", "def f():\n    return 1\n")
        res = self._edit("m.py", "    return 1", "    return (1")
        self.assertIn("rejected", res)
        self.assertIn("unparseable", res)
        self.assertIn("File unchanged", res)
        # file really unchanged
        self.assertIn("return 1", self._r("m.py"))
        self.assertNotIn("return (1", self._r("m.py"))

    def test_write_that_breaks_parse_is_rejected(self):
        self._w("m.py", "x = 1\n")
        res = self._write("m.py", "x = (1\n")
        self.assertIn("rejected", res)
        self.assertIn("unparseable", res)
        self.assertEqual(self._r("m.py"), "x = 1\n")

    def test_valid_edit_not_rejected(self):
        self._w("m.py", "x = 1\n")
        res = self._edit("m.py", "x = 1", "x = 2")
        self.assertIn("Successfully replaced", res)

    def test_already_broken_file_can_be_repaired(self):
        # A file that does NOT parse today must stay editable so the model can
        # fix it.
        self._w("broken.py", "def f(:\n    pass\n")
        res = self._edit("broken.py", "def f(:", "def f():")
        self.assertIn("Successfully replaced", res)
        self.assertIn("def f():", self._r("broken.py"))

    def test_non_py_never_validated(self):
        self._w("data.txt", "anything")
        res = self._write("data.txt", "((((not python")
        self.assertIn("Successfully wrote", res)

    def test_flag_disables_validation(self):
        h = FileInterfaceHarness(self.tmp, validate_python_syntax=False)
        self._w("m.py", "x = 1\n")
        res = h._execute_edit_file(
            ToolCall(
                name="edit_file",
                parameters={"file_path": "m.py", "old_string": "x = 1", "new_string": "x = (1"},
                reasoning="",
            )
        )
        self.assertIn("Successfully replaced", res)


class TestNoMatchError(_Base):
    def test_small_file_shows_whole_file_numbered(self):
        self._w("a.py", "alpha = 1\nbeta = 2\ngamma = 3\n")
        res = self._edit("a.py", "delta = 4", "delta = 5")
        self.assertIn("String not found", res)
        self.assertIn("   1| alpha = 1", res)
        self.assertIn("   3| gamma = 3", res)
        self.assertIn("shortest unique old_string copied from the ACTUAL", res)
        # must NOT suggest rg/grep
        self.assertNotIn("rg ", res)
        self.assertNotIn("grep", res)

    def test_large_file_shows_closest_region_with_context(self):
        lines = [f"filler {i}" for i in range(1, 200)]
        lines[120] = "    special_marker_line = 1"
        self._w("big.py", "\n".join(lines) + "\n")
        res = self._edit(
            "big.py", "    special_marker_line = 999", "    special_marker_line = 2"
        )
        self.assertIn("String not found", res)
        self.assertIn("Closest match", res)
        self.assertIn("special_marker_line", res)
        # +/-8 context around line 121 (1-based)
        self.assertIn(" 121| ", res)

    def test_mismatch_names_missing_argument_without_editing(self):
        lines = [f"filler {i}" for i in range(1, 140)]
        lines[130:134] = [
            "    def walk(self, references, href):",
            "        for i, ref in enumerate(references):",
            '            resolver.push(f"step[{i}]", ref)',
            "        return resolver.resolve(href)",
        ]
        source = "\n".join(lines) + "\n"
        self._w("resolver.py", source)
        old = "\n".join(lines[130:132] + [
            '            resolver.push(f"step[{i}]")',
            lines[133],
        ])
        res = self._edit("resolver.py", old, "replacement")
        self.assertIn("old_string line 3, resolver.py line 133", res)
        self.assertIn('requested: \'            resolver.push(f"step[{i}]")\'', res)
        self.assertIn('actual:    \'            resolver.push(f"step[{i}]", ref)\'', res)
        self.assertEqual(self._r("resolver.py"), source)


class TestSnapshotRevert(_Base):
    def test_revert_restores_snapshot(self):
        p = self._w("m.py", "x = 1\ny = 2\n")
        self.h.snapshot_workspace()
        p.write_text("TOTALLY DIFFERENT\n")
        res = self.h.execute_revert_file("m.py")
        self.assertIn("Restored m.py", res)
        self.assertIn("2 lines", res)
        self.assertEqual(self._r("m.py"), "x = 1\ny = 2\n")

    def test_revert_without_snapshot_errors(self):
        self._w("m.py", "x = 1\n")
        res = self.h.execute_revert_file("m.py")
        self.assertIn("no snapshot", res)

    def test_revert_unknown_file_errors(self):
        self._w("m.py", "x = 1\n")
        self.h.snapshot_workspace()
        res = self.h.execute_revert_file("never_existed.py")
        self.assertIn("no original snapshot", res)

    def test_dispatch_wrapper(self):
        p = self._w("m.py", "orig\n")
        self.h.snapshot_workspace()
        p.write_text("changed\n")
        res = self.h._execute_revert_file(
            ToolCall(name="revert_file", parameters={"file_path": "m.py"}, reasoning="")
        )
        self.assertIn("Restored", res)
        self.assertEqual(self._r("m.py"), "orig\n")


class TestWriteHint(_Base):
    def test_small_change_gets_edit_tip(self):
        self._w("m.py", "a = 1\nb = 2\nc = 3\n")
        res = self._write("m.py", "a = 1\nb = 99\nc = 3\n")
        self.assertIn("prefer edit_file", res)

    def test_large_change_no_tip(self):
        self._w("m.py", "a = 1\nb = 2\nc = 3\n")
        res = self._write("m.py", "\n".join(f"line{i} = {i}" for i in range(20)) + "\n")
        self.assertNotIn("prefer edit_file", res)

    def test_new_file_no_tip(self):
        res = self._write("fresh.py", "a = 1\n")
        self.assertIn("Successfully wrote", res)
        self.assertNotIn("prefer edit_file", res)


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# Ambiguous edit_file hard-errors. A silent 2-occurrence replace would corrupt
# the healthy sibling method. The taught contract has one rule: old_string must
# appear exactly once; to recover, widen it. anchor_line and replace_all work
# but the tool descriptions do not advertise them.
# --------------------------------------------------------------------------

AMBIG = (
    "class R:\n"
    "    def total_within(self):\n"
    "        return sum(s.within for s in self._summaries)\n"
    "    def total_breached(self):\n"
    "        return sum(s.within for s in self._summaries)\n"
)


class TestAmbiguousEditRefusal(_Base):
    def test_ambiguous_match_refused_file_unchanged(self):
        self._w("reporting.py", AMBIG)
        res = self._edit(
            "reporting.py",
            "        return sum(s.within for s in self._summaries)",
            "        return sum(s.breached for s in self._summaries)",
        )
        self.assertIn("appears 2 times", res)
        self.assertIn("surrounding lines", res)
        self.assertEqual(self._r("reporting.py"), AMBIG)  # untouched

    def test_widened_old_string_disambiguates(self):
        self._w("reporting.py", AMBIG)
        res = self._edit(
            "reporting.py",
            "    def total_breached(self):\n"
            "        return sum(s.within for s in self._summaries)",
            "    def total_breached(self):\n"
            "        return sum(s.breached for s in self._summaries)",
        )
        self.assertIn("replaced 1 occurrence(s)", res)
        content = self._r("reporting.py")
        self.assertIn("def total_within(self):\n        return sum(s.within", content)
        self.assertIn("def total_breached(self):\n        return sum(s.breached", content)

    def test_unadvertised_replace_all_still_works(self):
        self._w("reporting.py", AMBIG)
        res = self._edit("reporting.py", "s.within", "s.breached", replace_all=True)
        self.assertIn("replaced 2 occurrence(s)", res)
        self.assertEqual(self._r("reporting.py").count("s.breached"), 2)

    def test_unadvertised_anchor_line_keeps_legacy_window_semantics(self):
        # Explicit anchor = explicit scope: replace-within-window applies
        # (both occurrences here are inside the +/-20-line window, so both
        # are replaced).
        self._w("reporting.py", AMBIG)
        res = self._edit(
            "reporting.py",
            "        return sum(s.within for s in self._summaries)",
            "        return sum(s.breached for s in self._summaries)",
            anchor_line=5,
        )
        self.assertIn("replaced 2 occurrence(s)", res)

    def test_unique_match_unaffected(self):
        self._w("reporting.py", AMBIG)
        res = self._edit("reporting.py", "def total_breached", "def total_exceeded")
        self.assertIn("replaced 1 occurrence(s)", res)
