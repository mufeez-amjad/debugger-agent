"""Route tool calls to a workspace’s pdb, shell, and file handlers."""

import sys
from pathlib import Path
from typing import Optional

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from common.models import ToolCall
from .bash import BashSession
from .pdb import PdbHarness
from .file_interface import FileInterfaceHarness

class DebugSession:
    """A debugging session rooted at a codebase directory."""
    
    def __init__(self, codebase_directory: str, protected_files: list = None):
        """
        Initialize debug session at a codebase directory.

        Args:
            codebase_directory: Directory containing the codebase to debug
            protected_files: Optional list of file names that cannot be edited
        """
        self.codebase_directory = Path(codebase_directory).resolve()

        self.pdb_harness = PdbHarness(str(self.codebase_directory))
        self.file_interface = FileInterfaceHarness(
            str(self.codebase_directory),
            protected_files=protected_files
        )
        self._bash_session: Optional[BashSession] = None
    
    def execute_tool_call(self, tool_call: ToolCall) -> str:
        """Dispatch a tool call and return its observation."""
        pdb_tools = {
            'start_debugger', 'add_breakpoint', 'remove_breakpoint', 'clear_breakpoints',
            'continue_execution', 'step_over', 'step_into', 'step_out',
            'print_variable', 'examine_locals', 'examine_globals',
            'run_expression', 'list_code', 'where_am_i', 'inspect_stack'
        }
        bash_tools = {'bash'}
        # snapshot_workspace is an internal call used once at session start.
        file_tools = {'edit_file', 'write_file', 'revert_file'}
        control_tools = {'done'}

        if tool_call.name in control_tools:
            return "Task marked as complete."

        elif tool_call.name == 'snapshot_workspace':
            if self.file_interface:
                self.file_interface.snapshot_workspace()
                return "Workspace snapshot captured for revert_file."
            return "Error: File interface not available for snapshot_workspace"

        elif tool_call.name in pdb_tools:
            if self.pdb_harness:
                return self.pdb_harness.execute_tool_call(tool_call)
            else:
                return f"Error: PDB harness not available for {tool_call.name}"

        elif tool_call.name in bash_tools:
            return self._execute_bash(tool_call)

        elif tool_call.name in file_tools:
            if self.file_interface:
                return self.file_interface.execute_tool_call(tool_call)
            else:
                return f"Error: File interface not available for {tool_call.name}"

        else:
            return f"Error: Unknown tool '{tool_call.name}'"
    
    def start_debugging(self, target_code: str, filename: str = "debug_target.py") -> str:
        """Write target_code to filename and start a pdb session."""
        if not self.pdb_harness:
            return "Error: PDB harness not available"
            
        return self.pdb_harness.start_session(target_code, filename)
    
    def get_codebase_directory(self) -> str:
        """Get the codebase directory path."""
        return str(self.codebase_directory)
    
    # Point failed git/backup undo attempts to the session snapshot.
    _REVERT_HINT = (
        "\n\nNote: this workspace has no git history and no backup copies. "
        "To restore a file to its exact state from the start of the session, "
        "use the revert_file tool with its file_path."
    )
    _BACKUP_TOKENS = (".bak", ".orig", "backup", "_original")

    def _execute_bash(self, tool_call: ToolCall) -> str:
        """Execute a bash command with output truncation."""
        raw_command = tool_call.parameters.get("command", "")
        session = self._ensure_bash_session()
        result = session.execute(raw_command)
        if self._is_failed_undo_attempt(raw_command, result):
            result += self._REVERT_HINT
        return result

    def _is_failed_undo_attempt(self, command: str, result: str) -> bool:
        if "revert_file" in result:
            return False
        if "not a git repository" in result:
            return True
        lowered = command.lower()
        if "cannot stat" in result and any(
            tok in lowered for tok in self._BACKUP_TOKENS
        ):
            return True
        return False

    def _ensure_bash_session(self) -> BashSession:
        """Lazily initialize the persistent bash session."""
        if not self._bash_session:
            self._bash_session = BashSession(self.codebase_directory)
        return self._bash_session

    async def cleanup_async(self):
        """Clean up the debug session asynchronously."""
        if self.pdb_harness:
            await self.pdb_harness.cleanup_async()
        if self._bash_session:
            self._bash_session.close()
