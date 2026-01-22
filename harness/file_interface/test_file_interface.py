#!/usr/bin/env python3
"""
Comprehensive tests for FileInterfaceHarness class using unittest framework.

Tests all major methods including execute_tool_call and individual file operations.

There is no read_file tool (bash owns reads), so the read_file tests pin its
absence. edit_file tests use the old_string / new_string / anchor_line
contract.
"""
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from file_interface import FileInterfaceHarness
from common.models import ToolCall

class TestFileInterfaceHarness(unittest.TestCase):
    """Test suite for FileInterfaceHarness using unittest."""

    def setUp(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.harness = FileInterfaceHarness(self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_execute_tool_call_unknown_tool(self):
        """Test execute_tool_call with unknown tool name."""
        tool_call = ToolCall(
            name="unknown_tool",
            parameters={},
            reasoning="test"
        )
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("Unknown tool", result)

    def test_execute_tool_call_read_file_is_unknown(self):
        """read_file is not a registered tool."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Hello\nWorld\nTest")

        tool_call = ToolCall(
            name="read_file",
            parameters={"file_path": "test.txt"},
            reasoning="test reading"
        )

        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("Unknown tool", result)
        # And the capability methods are gone from the harness entirely.
        self.assertFalse(hasattr(self.harness, "_execute_read_file"))
        self.assertFalse(hasattr(self.harness, "execute_read_file"))

    def test_execute_tool_call_non_file_operation(self):
        """Test execute_tool_call with a declared tool that is not a file operation."""
        tool_call = ToolCall(
            name="add_breakpoint",  # This is a PDB tool, not file operation
            parameters={},
            reasoning="test"
        )
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("not implemented in FileInterfaceHarness", result)

    def test_execute_tool_call_write_file_success(self):
        """Test successful file writing through execute_tool_call."""
        tool_call = ToolCall(
            name="write_file",
            parameters={
                "file_path": "new_file.txt",
                "content": "Test content\nLine 2"
            },
            reasoning="test writing"
        )

        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("Successfully wrote", result)
        self.assertIn("2 lines", result)

        # Verify file was actually created
        test_file = Path(self.temp_dir) / "new_file.txt"
        self.assertTrue(test_file.exists())
        self.assertEqual(test_file.read_text(), "Test content\nLine 2")

    def test_execute_tool_call_edit_file_success(self):
        """Test successful file editing through execute_tool_call."""
        # Create test file
        test_file = Path(self.temp_dir) / "edit_test.txt"
        test_file.write_text("Hello World\nThis is a test\nGoodbye")

        tool_call = ToolCall(
            name="edit_file",
            parameters={
                "file_path": "edit_test.txt",
                "old_string": "This is a test",
                "new_string": "This is modified"
            },
            reasoning="test editing"
        )

        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("Successfully replaced", result)

        # Verify file was actually modified
        content = test_file.read_text()
        self.assertIn("This is modified", content)
        self.assertNotIn("This is a test", content)

    def test_execute_tool_call_edit_file_not_found(self):
        """Test editing non-existent file through execute_tool_call."""
        tool_call = ToolCall(
            name="edit_file",
            parameters={
                "file_path": "nonexistent.txt",
                "old_string": "test",
                "new_string": "replacement"
            },
            reasoning="test"
        )

        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("does not exist", result)

    def test_execute_tool_call_missing_file_path(self):
        """Missing file_path is reported as an error, not an exception."""
        tool_call = ToolCall(
            name="write_file",
            parameters={"content": "orphan content"},  # Missing file_path
            reasoning="test error handling"
        )

        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("No file path", result)

    def test_get_set_working_directory(self):
        """Test get_working_directory and set_working_directory methods."""
        # Test getting current directory
        current_dir = self.harness.get_working_directory()
        self.assertEqual(current_dir, str(Path(self.temp_dir).resolve()))

        # Create new test directory
        new_dir = Path(self.temp_dir) / "newdir"
        new_dir.mkdir()

        # Test setting valid directory
        result = self.harness.set_working_directory(str(new_dir))
        self.assertIn("Working directory changed", result)
        self.assertEqual(self.harness.get_working_directory(), str(new_dir.resolve()))

        # Test setting invalid directory
        result = self.harness.set_working_directory("nonexistent_dir")
        self.assertIn("does not exist", result)

        # Test setting file as directory
        test_file = new_dir / "test.txt"
        test_file.write_text("content")
        result = self.harness.set_working_directory(str(test_file))
        self.assertIn("not a directory", result)

    def test_operation_history(self):
        """Test operation history tracking."""
        # Initially empty
        history = self.harness.get_operation_history()
        self.assertEqual(len(history), 0)

        # Execute some operations
        self.harness.execute_tool_call(ToolCall(
            name="unknown_tool", parameters={}, reasoning=""
        ))

        self.harness.execute_tool_call(ToolCall(
            name="write_file",
            parameters={"file_path": "test.txt", "content": "content"},
            reasoning="",
        ))

        # Check history
        history = self.harness.get_operation_history()
        self.assertEqual(len(history), 2)
        self.assertIn("[ERROR]", history[0])
        self.assertIn("[WRITE_FILE]", history[1])

    def test_resolve_path(self):
        """Test _resolve_path method with absolute and relative paths."""
        # Test relative path
        relative_result = self.harness._resolve_path("test.txt")
        expected_relative = Path(self.temp_dir) / "test.txt"
        self.assertEqual(str(relative_result), str(expected_relative.resolve()))

        # Test absolute path
        absolute_path = "/tmp/absolute_test.txt"
        absolute_result = self.harness._resolve_path(absolute_path)
        self.assertEqual(str(absolute_result), absolute_path)

    def test_write_file_edge_cases(self):
        """Test write_file with edge cases."""
        # Test missing file_path parameter
        result = self.harness._execute_write_file(ToolCall(
            name="write_file", parameters={"content": "test"}, reasoning=""
        ))
        self.assertIn("No file path", result)

        # Test writing to nested directory that doesn't exist
        result = self.harness._execute_write_file(ToolCall(
            name="write_file", parameters={
                "file_path": "nested/deep/file.txt",
                "content": "test content"
            }, reasoning=""
        ))
        self.assertIn("Successfully wrote", result)

        # Verify nested directory was created
        nested_file = Path(self.temp_dir) / "nested" / "deep" / "file.txt"
        self.assertTrue(nested_file.exists())

    def test_edit_file_anchor_line_disambiguates(self):
        """edit_file with anchor_line restricts the match to a ±20-line window
        so an otherwise-ambiguous old_string applies unambiguously."""
        # Two identical lines >40 lines apart so only one falls in the window.
        filler = "\n".join(f"# filler {i}" for i in range(50))
        content = "    return value\n" + filler + "\n    return value\n"
        test_file = Path(self.temp_dir) / "context_test.txt"
        test_file.write_text(content)

        result = self.harness._execute_edit_file(ToolCall(
            name="edit_file", parameters={
                "file_path": "context_test.txt",
                "old_string": "    return value",
                "new_string": "    return result",
                "anchor_line": 52,
            }, reasoning=""
        ))

        self.assertIn("Successfully replaced", result)

        # Verify only the second occurrence was changed
        lines = test_file.read_text().split('\n')
        self.assertEqual(lines[0].strip(), "return value")   # first untouched
        self.assertEqual(lines[51].strip(), "return result")  # second changed

    def test_edit_file_missing_parameters(self):
        """Test edit_file with missing required parameters."""
        # Test missing file_path
        result = self.harness._execute_edit_file(ToolCall(
            name="edit_file", parameters={"old_string": "test"}, reasoning=""
        ))
        self.assertIn("No file path", result)

        # Test missing old_string
        result = self.harness._execute_edit_file(ToolCall(
            name="edit_file", parameters={"file_path": "test.txt"}, reasoning=""
        ))
        self.assertIn("No old_string", result)

        # Test missing new_string
        result = self.harness._execute_edit_file(ToolCall(
            name="edit_file", parameters={
                "file_path": "test.txt",
                "old_string": "test"
            }, reasoning=""
        ))
        self.assertIn("No new_string", result)


class TestIndentTolerantEdit(unittest.TestCase):
    """Tier-3 edit_file matching: auto-apply unique leading-indent-drift matches.

    Most verbatim failed-edit retries happen after the model sees a ~99%-similar
    closest-match hint and ignores it. The difference is pure whitespace or
    indent drift. Tier 3 auto-applies the unambiguous cases (unique stripped match + uniform indent delta), re-indenting
    new_string to the file's indentation so the strict-indent safety still holds.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.harness = FileInterfaceHarness(self.temp_dir)

    def tearDown(self):
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def _w(self, name, content):
        (Path(self.temp_dir) / name).write_text(content)

    def _r(self, name):
        return (Path(self.temp_dir) / name).read_text()

    def _edit(self, name, old, new):
        return self.harness.execute_tool_call(ToolCall(
            name="edit_file",
            parameters={"file_path": name, "old_string": old, "new_string": new},
            reasoning="",
        ))

    def test_indent_drift_autoapplies_at_file_indent(self):
        self._w("a.py", "class C:\n    def f(self):\n        x = foo()\n        return x\n")
        r = self._edit("a.py", "    x = foo()\n    return x", "    x = bar()\n    return x")
        self.assertIn("re-indented", r)
        self.assertIn("        x = bar()\n        return x\n", self._r("a.py"))

    def test_content_mismatch_still_bounces(self):
        self._w("b.py", "        x = foo()\n")
        r = self._edit("b.py", "    x = qux()", "    x = baz()")
        self.assertIn("String not found", r)

    def test_non_unique_stripped_match_bounces(self):
        self._w("c.py", "    a = 1\n    b = 2\n        a = 1\n        b = 2\n")
        r = self._edit("c.py", "  a = 1\n  b = 2", "  a = 9\n  b = 9")
        self.assertIn("String not found", r)

    def test_exact_match_unaffected(self):
        self._w("d.py", "    y = 1\n")
        r = self._edit("d.py", "    y = 1", "    y = 2")
        self.assertIn("Successfully replaced", r)
        self.assertIn("    y = 2", self._r("d.py"))


if __name__ == "__main__":
    unittest.main()
