"""Pdb commands, cross-file breakpoints, and session recovery."""

import asyncio
import tempfile
import unittest
from pathlib import Path

from .pdb import PdbHarness, ToolCall

class TestPdbHarness(unittest.TestCase):
    """Test suite for PdbHarness functionality."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.harness = PdbHarness(working_directory=self._tmpdir.name)

    def tearDown(self):
        asyncio.run(self.harness.cleanup_async())
        self._tmpdir.cleanup()

    def _tool(self, name: str, parameters=None, reasoning: str = "Auto-generated reasoning") -> ToolCall:
        """Create a ToolCall with required fields filled."""
        return ToolCall(name, parameters=parameters or {}, reasoning=reasoning)
    
    def test_simple_factorial_program(self):
        """Test debugging a simple factorial function with a bug."""
        buggy_code = """
def factorial(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n):  # Bug: should be range(2, n+1)
        result *= i
    return result

# Test the function
print(f"factorial(5) = {factorial(5)}")
"""
        
        # Start session
        result = self.harness.start_session(buggy_code, "factorial.py")
        self.assertIn("Debugging session started", result)
        
        # Run program to see the bug
        tool_call = ToolCall("run_program", parameters={}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("factorial(5) = 24", result)  # Should be 120, bug confirmed
        
        # Start debugger
        tool_call = ToolCall("start_debugger", parameters={}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("Debugger", result)
        
        # Set breakpoint at the loop
        tool_call = ToolCall("add_breakpoint", parameters={"line": 6}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("Breakpoint set at", result)
        self.assertIn("6", result)
        
        # Continue to breakpoint
        tool_call = ToolCall("continue_execution", parameters={}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertTrue("breakpoint" in result.lower() or "->" in result)
        
        # Examine local variables
        tool_call = ToolCall("examine_locals", parameters={}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIsInstance(result, str)
        self.assertNotIn("Error:", result)
        
        # Print specific variable
        tool_call = ToolCall("print_variable", parameters={"name": "n"}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIsInstance(result, str)
        self.assertNotIn("Error:", result)
        self.assertNotIn("NameError", result)
        
        # Step over a few lines
        tool_call = ToolCall("step_over", parameters={}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIsInstance(result, str)
        self.assertNotIn("Error: No active debugging session", result)
        
        # List code around current line
        tool_call = ToolCall("list_code", parameters={}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("factorial", result)
        
        # Propose a fix
        fixed_code = """
def factorial(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n+1):  # Fixed: added +1
        result *= i
    return result

# Test the function
print(f"factorial(5) = {factorial(5)}")
"""
        tool_call = ToolCall("propose_fix", parameters={"code": fixed_code}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("Proposed fix", result)
        
        # Verify the fix
        tool_call = ToolCall("verify_output", parameters={}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("factorial(5) = 120", result)  # Now should be correct
    
    def test_fibonacci_with_logic_error(self):
        """Test debugging a Fibonacci function with a logic error."""
        buggy_code = """
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    a, b = 0, 1
    for i in range(2, n):  # Bug: should go to n+1
        a, b = b, a + b
    return b

# Test cases
for i in range(6):
    print(f"fib({i}) = {fibonacci(i)}")
"""
        
        # Start session
        self.harness.start_session(buggy_code, "fibonacci.py")
        
        # Run program to see output
        tool_call = ToolCall("run_program", parameters={}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("fib(", result)
        
        # Start debugging
        tool_call = ToolCall("start_debugger", parameters={}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("Debugger", result)
        
        # Set breakpoint in the loop
        tool_call = ToolCall("add_breakpoint", parameters={"line": 9}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("Breakpoint set", result)
        
        # Run expression to test hypothesis
        tool_call = ToolCall("run_expression", parameters={"expression": "2 + 2"}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIsInstance(result, str)
        self.assertNotIn("Error: No expression specified", result)
        
        # Step into function call
        tool_call = ToolCall("step_into", parameters={}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIsInstance(result, str)
        self.assertNotIn("Error: No active debugging session", result)
        
        # Where am I?
        tool_call = ToolCall("where_am_i", parameters={}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIsInstance(result, str)
    
    def test_array_processing_with_index_error(self):
        """Test debugging an array processing function with potential index errors."""
        buggy_code = """
def process_array(arr):
    result = []
    for i in range(len(arr)):
        # Bug: accessing arr[i+1] without bounds check
        if i < len(arr) - 1 and arr[i] < arr[i+1]:
            result.append(arr[i])
    return result

def find_increasing_elements(data):
    processed = process_array(data)
    return processed

# Test with different arrays
test_data = [1, 3, 2, 5, 4, 8, 7, 9]
result = find_increasing_elements(test_data)
print(f"Increasing elements: {result}")
"""
        
        # Start session
        self.harness.start_session(buggy_code, "array_proc.py")
        
        # Test all major debugging operations
        operations = [
            ("run_program", {}),
            ("start_debugger", {}),
            ("add_breakpoint", {"line": 4}),
            ("continue_execution", {}),
            ("examine_locals", {}),
            ("print_variable", {"name": "arr"}),
            ("step_over", {}),
            ("list_code", {"start": 1, "end": 10}),
            ("run_expression", {"expression": "len(arr)"}),
            ("examine_globals", {}),
        ]
        
        for op_name, params in operations:
            with self.subTest(operation=op_name):
                tool_call = self._tool(op_name, params)
                result = self.harness.execute_tool_call(tool_call)
                self.assertIsInstance(result, str)
                self.assertNotIn("Error executing", result)
    
    def test_recursive_function_debugging(self):
        """Test debugging a recursive function with a bug."""
        buggy_code = """
def binary_search(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)

# Test the function
sorted_array = [1, 3, 5, 7, 9, 11, 13, 15]
target = 7
index = binary_search(sorted_array, target)
print(f"Found {target} at index: {index}")
"""
        
        # Start session and test recursive debugging
        self.harness.start_session(buggy_code, "binary_search.py")
        
        # Run and debug
        tool_call = ToolCall("run_program", parameters={}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("Found 7 at index:", result)
        
        # Test file operations
        if not getattr(self.harness, "file_interface", None):
            self.skipTest("FileInterfaceHarness not available (harness/file-interface not installed)")

        # Only edit/write tools are supported; verify they execute.
        tool_call = self._tool("write_file", {"file_path": "tmp_note.txt", "content": "hello"})
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("Successfully wrote", result)

        tool_call = self._tool("edit_file", {"file_path": "tmp_note.txt", "old_string": "hello", "new_string": "hello world"})
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("Successfully replaced", result)
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        buggy_code = """
def divide_numbers(a, b):
    return a / b  # Potential division by zero

result = divide_numbers(10, 0)
print(result)
"""
        
        self.harness.start_session(buggy_code, "divide.py")
        
        # Test invalid operations
        test_cases = [
            ("print_variable", {"name": ""}, "Error: No variable name specified"),
            ("add_breakpoint", {}, "Error: No line number specified"),
            ("run_expression", {}, "Error: No expression specified"),
            ("add_breakpoint", {"line": "invalid"}, "Error: Invalid line number"),
            ("propose_fix", {}, "Error: No fixed code provided"),
        ]
        
        for op_name, params, expected_error in test_cases:
            with self.subTest(operation=op_name):
                tool_call = self._tool(op_name, params)
                result = self.harness.execute_tool_call(tool_call)
                self.assertIn(expected_error, result)
    
    def test_complex_program_with_multiple_functions(self):
        """Test debugging a more complex program with multiple functions."""
        complex_code = """
def calculate_statistics(numbers):
    if not numbers:
        return None
    
    total = sum(numbers)
    count = len(numbers)
    mean = total / count
    
    # Bug: incorrect variance calculation
    variance = sum((x - mean) ** 2 for x in numbers) / count  # Should be (count-1) for sample variance
    std_dev = variance ** 0.5
    
    return {
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev,
        'min': min(numbers),
        'max': max(numbers)
    }

def analyze_data(data_sets):
    results = []
    for i, dataset in enumerate(data_sets):
        print(f"Analyzing dataset {i+1}: {dataset}")
        stats = calculate_statistics(dataset)
        if stats:
            results.append(stats)
        else:
            print(f"Warning: Dataset {i+1} is empty")
    return results

# Test data
test_datasets = [
    [1, 2, 3, 4, 5],
    [10, 20, 30, 40, 50],
    [],
    [100, 200, 150, 175, 125]
]

analysis_results = analyze_data(test_datasets)
print(f"Analysis complete. Processed {len(analysis_results)} datasets.")
"""
        
        # Start session
        self.harness.start_session(complex_code, "statistics.py")
        
        # Comprehensive debugging session
        debugging_session = [
            ("run_program", {}),
            ("start_debugger", {}),
            ("add_breakpoint", {"line": 8}),  # In calculate_statistics
            ("add_breakpoint", {"line": 20}), # In analyze_data
            ("continue_execution", {}),
            ("examine_locals", {}),
            ("print_variable", {"name": "numbers"}),
            ("step_over", {}),
            ("step_over", {}),
            ("print_variable", {"name": "mean"}),
            ("run_expression", {"expression": "len(numbers)"}),
            ("list_code", {"start": 5, "end": 15}),
            ("step_into", {}),
            ("where_am_i", {}),
        ]
        
        for op_name, params in debugging_session:
            with self.subTest(operation=op_name):
                tool_call = self._tool(op_name, params)
                result = self.harness.execute_tool_call(tool_call)
                self.assertIsInstance(result, str)
                # Should not have critical errors
                self.assertNotIn("Error executing", result)
    
    def test_session_management(self):
        """Test session creation, management, and cleanup."""
        # Test starting multiple sessions
        code1 = "print('Session 1')"
        code2 = "print('Session 2')"
        
        # Start first session
        result1 = self.harness.start_session(code1, "session1.py")
        self.assertIn("session1.py", result1)
        
        # Start second session (should cleanup first)
        result2 = self.harness.start_session(code2, "session2.py")
        self.assertIn("session2.py", result2)
        
        # Test session summary
        summary = self.harness.get_session_summary()
        self.assertIn("status", summary)
        self.assertEqual(summary["filename"], "session2.py")
        
        # Test operations without active session
        asyncio.run(self.harness.cleanup_async())
        tool_call = ToolCall("run_program", parameters={}, reasoning="Auto-generated reasoning")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("No debugging session active", result)
    
    def test_tool_call_parameter_variations(self):
        """Test different ways of passing parameters to tool calls."""
        code = """
x = 10
y = 20
z = x + y
print(f"Result: {z}")
"""
        
        self.harness.start_session(code, "params_demo.py")

        # Execute the first two assignments so variables exist in the current frame.
        result = self.harness.execute_tool_call(self._tool("step_over"))
        self.assertNotIn("Error:", result)
        result = self.harness.execute_tool_call(self._tool("step_over"))
        self.assertNotIn("Error:", result)
        
        # Test named parameters
        tool_call = ToolCall("print_variable", parameters={"name": "x"}, reasoning="Test print variable with name x")
        result = self.harness.execute_tool_call(tool_call)
        self.assertNotIn("Error:", result)
        self.assertNotIn("NameError", result)
        
        # Test named parameters
        tool_call = ToolCall("print_variable", parameters={"name": "y"}, reasoning="Test print variable with name y")
        result = self.harness.execute_tool_call(tool_call)
        self.assertNotIn("Error:", result)
        self.assertNotIn("NameError", result)
        
        # Test mixed parameters (unnamed takes precedence for some operations)
        tool_call = ToolCall("add_breakpoint", 
                           parameters={"line": 5},
                           reasoning="Test breakpoint with mixed parameters")
        result = self.harness.execute_tool_call(tool_call)
        self.assertIn("Breakpoint", result)
    
    def test_inspect_stack(self):
        """Test inspect_stack shows call stack with locals at each frame."""
        code = """
def inner(x):
    y = x * 2
    return y

def outer(n):
    result = inner(n + 1)
    return result

val = outer(3)
print(val)
"""
        self.harness.start_session(code, "stack_demo.py")

        # Start debugger and set breakpoint inside inner()
        self.harness.execute_tool_call(self._tool("start_debugger"))
        result = self.harness.execute_tool_call(self._tool("add_breakpoint", {"line": 3}))
        self.assertIn("Breakpoint set", result)

        # Continue to hit breakpoint
        self.harness.execute_tool_call(self._tool("continue_execution"))

        # Inspect the stack
        result = self.harness.execute_tool_call(self._tool("inspect_stack"))
        self.assertIsInstance(result, str)
        self.assertNotIn("Error:", result)
        # Should contain frame markers and locals
        self.assertIn("Frame", result)
        self.assertIn("Locals:", result)

    def test_inspect_stack_with_max_frames(self):
        """Test inspect_stack respects max_frames parameter."""
        code = """
def a():
    return b()

def b():
    return c()

def c():
    x = 42
    return x

result = a()
print(result)
"""
        self.harness.start_session(code, "deep_stack.py")
        self.harness.execute_tool_call(self._tool("start_debugger"))
        self.harness.execute_tool_call(self._tool("add_breakpoint", {"line": 9}))
        self.harness.execute_tool_call(self._tool("continue_execution"))

        result = self.harness.execute_tool_call(self._tool("inspect_stack", {"max_frames": 2}))
        self.assertIsInstance(result, str)
        self.assertNotIn("Error:", result)
        self.assertIn("Frame", result)


class TestCrossFileBreakpoints(unittest.TestCase):
    """Tests for breakpoints set in a different file than the debugged file."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.harness = PdbHarness(working_directory=self._tmpdir.name)

    def tearDown(self):
        asyncio.run(self.harness.cleanup_async())
        self._tmpdir.cleanup()

    def _tool(self, name, parameters=None):
        return ToolCall(name, parameters=parameters or {}, reasoning="test")

    def test_breakpoint_in_imported_file_is_hit(self):
        """Setting a breakpoint in an imported module via a pytest test file should
        stop inside that module (exercises the pytest --trace path)."""
        import os

        # Write the library file
        lib_code = """\
def add_tag(tag, tags=None):
    if tags is None:
        tags = []
    tags.append(tag)
    return tags
"""
        lib_path = os.path.join(self._tmpdir.name, "tags.py")
        with open(lib_path, "w") as f:
            f.write(lib_code)

        # Write a proper pytest test file (test_ prefix triggers pytest --trace)
        test_code = """\
from tags import add_tag

def test_add_tag():
    result = add_tag("x")
    assert result == ["x"]
"""
        test_path = os.path.join(self._tmpdir.name, "test_tags.py")
        with open(test_path, "w") as f:
            f.write(test_code)

        # Start debugger on the test file
        result = self.harness.execute_tool_call(self._tool(
            "start_debugger", {"file_path": "test_tags.py"}
        ))
        self.assertIn("Debugging session started", result)

        # Set breakpoint in tags.py (a different file) at line 4 (tags.append)
        result = self.harness.execute_tool_call(self._tool(
            "add_breakpoint",
            {"breakpoints": [{"file": "tags.py", "lines": [4]}]}
        ))
        self.assertIn("Breakpoint set", result)

        # Continue — should hit the breakpoint inside tags.py
        result = self.harness.execute_tool_call(self._tool("continue_execution"))
        self.assertNotIn("Program finished without hitting", result,
                         f"Breakpoint was not hit: {result}")

        # Examine locals — we should see 'tag' and 'tags' from add_tag's frame
        locals_result = self.harness.execute_tool_call(self._tool("examine_locals"))
        self.assertIn("tag", locals_result, f"Expected 'tag' in locals but got: {locals_result}")

    def test_breakpoint_same_file_no_qualifier(self):
        """Breakpoints in the session file should use bare line numbers (no file: prefix)."""
        code = """\
x = 1
y = 2
z = x + y
print(z)
"""
        self.harness.start_session(code, "simple.py")
        self.harness.execute_tool_call(self._tool("start_debugger"))

        # Set breakpoint in same file using the complex format
        result = self.harness.execute_tool_call(self._tool(
            "add_breakpoint",
            {"breakpoints": [{"file": "simple.py", "lines": [3]}]}
        ))
        self.assertIn("Breakpoint set", result)

        # Continue to breakpoint
        result = self.harness.execute_tool_call(self._tool("continue_execution"))

        # Check locals — x and y should be visible
        locals_result = self.harness.execute_tool_call(self._tool("examine_locals"))
        self.assertIn("x", locals_result)

    def test_examine_locals_dumps_self_dict(self):
        """When self is in the current frame, examine_locals should also dump self.__dict__.

        Without this, the model wastes debugger steps because pp locals() only shows `'self': <obj at 0x...>`
        and the bug-relevant state lives on self's instance attributes.
        """
        code = """\
class Counter:
    def __init__(self, start):
        self._n = start
        self._history = []

    def bump(self):
        marker = "before-bump"
        self._history.append(self._n)
        self._n += 1
        return self._n

c = Counter(10)
c.bump()
"""
        self.harness.start_session(code, "counter.py")
        self.harness.execute_tool_call(self._tool("start_debugger"))
        # Break inside bump() so `self` and a local (`marker`) are both in scope.
        self.harness.execute_tool_call(self._tool(
            "add_breakpoint",
            {"breakpoints": [{"file": "counter.py", "lines": [8]}]}
        ))
        self.harness.execute_tool_call(self._tool("continue_execution"))

        result = self.harness.execute_tool_call(self._tool("examine_locals"))
        # Locals are visible (marker, self).
        self.assertIn("marker", result, f"Expected local 'marker' in output: {result}")
        # Labeled sections, not raw pdb separators.
        self.assertIn("Local variables:", result,
                      f"Expected 'Local variables:' section in output: {result}")
        self.assertIn("Instance attributes of self (vars(self)):", result,
                      f"Expected vars(self) section in output: {result}")
        self.assertIn("_n", result, f"Expected attribute '_n' in output: {result}")
        self.assertIn("_history", result, f"Expected attribute '_history' in output: {result}")

    def test_examine_locals_no_self_section_when_no_self(self):
        """At module-level frames (no self), examine_locals should not append the section."""
        code = """\
a = 1
b = 2
c = a + b
"""
        self.harness.start_session(code, "noself.py")
        self.harness.execute_tool_call(self._tool("start_debugger"))
        self.harness.execute_tool_call(self._tool(
            "add_breakpoint",
            {"breakpoints": [{"file": "noself.py", "lines": [3]}]}
        ))
        self.harness.execute_tool_call(self._tool("continue_execution"))

        result = self.harness.execute_tool_call(self._tool("examine_locals"))
        self.assertNotIn("Instance attributes of self", result,
                         f"Did not expect vars(self) section: {result}")


class TestMissedBreakpointRecovery(unittest.TestCase):
    """Scope-warning on import-time breakpoints, and keeping
    the session alive (with an actionable diagnostic) when `continue` runs the
    program to completion without hitting a breakpoint."""

    LIB = (
        "class Counter:\n"
        "    _shared = []\n"            # line 2: class-body, runs at import
        "    def __init__(self):\n"
        "        self.values = {}\n"     # line 4: inside __init__
        "    def bump(self, k):\n"
        "        self.values[k] = self.values.get(k, 0) + 1\n"
        "        return self.values[k]\n"
        "    def never_called(self):\n"  # line 8: def header
        "        return 42\n"            # line 9: in-function, but never called
    )
    TEST = (
        "from counter import Counter\n"
        "def test_bump():\n"
        "    c = Counter()\n"
        "    assert c.bump('a') == 1\n"
    )

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        import os
        with open(os.path.join(self._tmpdir.name, "counter.py"), "w") as fh:
            fh.write(self.LIB)
        with open(os.path.join(self._tmpdir.name, "test_counter.py"), "w") as fh:
            fh.write(self.TEST)
        self.harness = PdbHarness(working_directory=self._tmpdir.name)

    def tearDown(self):
        asyncio.run(self.harness.cleanup_async())
        self._tmpdir.cleanup()

    def _tool(self, name, parameters=None):
        return ToolCall(name, parameters=parameters or {}, reasoning="t")

    def test_class_body_breakpoint_rejected(self):
        # A breakpoint on the class-body line 2 (runs once at import, never
        # hit) is REJECTED outright. The harness does not move it to another
        # line, because the model would lean on that instead of reading the
        # code. No breakpoint is set and no replacement line is suggested.
        self.harness.execute_tool_call(self._tool("start_debugger", {"file_path": "test_counter.py"}))
        result = self.harness.execute_tool_call(
            self._tool("add_breakpoint", {"breakpoints": [{"file": "counter.py", "lines": [2]}]})
        )
        self.assertIn("not an executable statement inside a function", result)
        self.assertIn("No breakpoint was set", result)
        self.assertNotIn("Breakpoint set", result)
        self.assertEqual(self.harness.current_session.breakpoints, [])

    def test_executable_line_breakpoint_actually_hit(self):
        # A breakpoint on a real in-function line fires instead of the program
        # running to completion.
        self.harness.execute_tool_call(self._tool("start_debugger", {"file_path": "test_counter.py"}))
        self.harness.execute_tool_call(
            self._tool("add_breakpoint", {"breakpoints": [{"file": "counter.py", "lines": [4]}]})
        )
        result = self.harness.execute_tool_call(self._tool("continue_execution"))
        self.assertNotIn("WITHOUT stopping", result)
        self.assertTrue(self.harness.current_session.is_alive())
        self.assertEqual(self.harness.current_session.current_position["line"], 4)

    def test_inside_function_breakpoint_has_no_warning(self):
        self.harness.execute_tool_call(self._tool("start_debugger", {"file_path": "test_counter.py"}))
        result = self.harness.execute_tool_call(
            self._tool("add_breakpoint", {"breakpoints": [{"file": "counter.py", "lines": [4]}]})
        )
        self.assertIn("Breakpoint set", result)
        self.assertNotIn("adjusted from", result)
        self.assertNotIn("WARNING", result)

    def test_continue_past_missed_breakpoint_recovers_and_stays_alive(self):
        # Line 9 is inside never_called() — executable, so it is NOT adjusted,
        # but the test never calls it, so the program still runs to completion.
        # The recovery path must still produce an actionable diagnostic.
        self.harness.execute_tool_call(self._tool("start_debugger", {"file_path": "test_counter.py"}))
        self.harness.execute_tool_call(
            self._tool("add_breakpoint", {"breakpoints": [{"file": "counter.py", "lines": [9]}]})
        )
        result = self.harness.execute_tool_call(self._tool("continue_execution"))
        # Actionable diagnostic instead of a silent dead session.
        self.assertIn("WITHOUT stopping at your breakpoint", result)
        # Session was restarted and remains usable.
        self.assertTrue(self.harness.current_session.is_alive())

    def test_good_breakpoint_stops_and_keeps_session_live(self):
        self.harness.execute_tool_call(self._tool("start_debugger", {"file_path": "test_counter.py"}))
        self.harness.execute_tool_call(
            self._tool("add_breakpoint", {"breakpoints": [{"file": "counter.py", "lines": [4]}]})
        )
        result = self.harness.execute_tool_call(self._tool("continue_execution"))
        self.assertNotIn("WITHOUT stopping", result)
        self.assertTrue(self.harness.current_session.is_alive())
        self.assertEqual(self.harness.current_session.current_position["line"], 4)
