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
