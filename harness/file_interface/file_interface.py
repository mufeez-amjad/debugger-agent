"""Write, edit, and revert files in a debugging workspace."""

import ast
import difflib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.models import ToolCall, TOOLS_DESCRIPTIONS


class FileInterfaceHarness:
    """File operations with protected-file checks and optional Python syntax validation."""
    
    ENCODING = 'utf-8'
    
    def __init__(
        self,
        working_directory: str = ".",
        protected_files: Optional[List[str]] = None,
        validate_python_syntax: bool = True,
    ):
        """
        Initialize file interface harness.

        Args:
            working_directory: Base directory for file operations
            protected_files: Optional list of file names that cannot be edited or written
            validate_python_syntax: When True (default), edits/writes to .py files are
                ast.parse-checked before writing; a change that would make the file
                unparseable is rejected and the file left unchanged.
        """
        self.working_directory = Path(working_directory).resolve()
        self.operation_history: List[str] = []
        self.protected_files = set(protected_files) if protected_files else set()
        self.validate_python_syntax = validate_python_syntax
        # Content of every .py file captured at reset by snapshot_workspace();
        # None until snapshotted (revert then errors informatively).
        self._snapshot: Optional[Dict[str, str]] = None
        
    def execute_tool_call(self, tool_call: ToolCall) -> str:
        """Execute a file operation tool call and return the result."""
        if tool_call.name not in TOOLS_DESCRIPTIONS:
            return self._log_and_return_error(
                f"Unknown tool '{tool_call.name}'. Available tools: {list(TOOLS_DESCRIPTIONS.keys())[:5]}..."
            )
        
        method_name = f"_execute_{tool_call.name}"
        if not hasattr(self, method_name):
            return self._log_and_return_error(
                f"Tool {tool_call.name} not implemented in FileInterfaceHarness"
            )
            
        try:
            result = getattr(self, method_name)(tool_call)
            self.operation_history.append(f"[{tool_call.name.upper()}] {result[:100]}...")
            return result
        except Exception as e:
            return self._log_and_return_error(f"Error executing {tool_call.name}: {str(e)}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the working directory."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        else:
            return self.working_directory / path_obj

    def _is_file_protected(self, file_path: str) -> bool:
        """Match protected files by basename."""
        if not self.protected_files:
            return False

        file_name = Path(file_path).name

        return file_name in self.protected_files

    def _log_and_return_error(self, error_msg: str) -> str:
        """Log error message to history and return it."""
        self.operation_history.append(f"[ERROR] {error_msg}")
        return error_msg
        
    def _validate_file_path(self, file_path: str, must_exist: bool = False, 
                          must_be_file: bool = False, must_be_writable: bool = False) -> Tuple[Optional[Path], Optional[str]]:
        """Validate file path and return resolved path or error message."""
        full_path = self._resolve_path(file_path)
        
        if must_exist and not full_path.exists():
            return None, f"Error: File {file_path} does not exist"
            
        if must_be_file and full_path.exists() and not full_path.is_file():
            return None, f"Error: {file_path} is not a file"
            
        if must_be_writable and full_path.exists() and not os.access(full_path, os.W_OK):
            return None, f"Error: File {file_path} is read-only and cannot be modified"
            
        return full_path, None
        
    def get_working_directory(self) -> str:
        """Get current working directory."""
        return str(self.working_directory)
    
    def set_working_directory(self, directory: str) -> str:
        """Set working directory."""
        try:
            new_dir = Path(directory).resolve()
            if not new_dir.exists():
                return f"Error: Directory {directory} does not exist"
            if not new_dir.is_dir():
                return f"Error: {directory} is not a directory"
            
            old_dir = self.working_directory
            self.working_directory = new_dir
            return f"Working directory changed from {old_dir} to {new_dir}"
            
        except Exception as e:
            return f"Error changing working directory: {str(e)}"
    
    def get_operation_history(self) -> List[str]:
        """Get history of file operations."""
        return self.operation_history.copy()

    def _execute_write_file(self, tool_call: ToolCall) -> str:
        """Write content to a file (creates or overwrites)."""
        file_path = tool_call.parameters.get('file_path')
        content = tool_call.parameters.get('content')

        if not file_path:
            return "Error: No file path specified"
        if content is None:  # Allow empty content
            return "Error: No content specified"

        # Check if file is protected
        if self._is_file_protected(file_path):
            return f"Error: Cannot write to protected file {file_path}"

        full_path = self._resolve_path(file_path)

        # Capture existing content (for the small-change hint AND the syntax
        # guard, which only fires when the file currently parses) before overwrite.
        existing: Optional[str] = None
        if full_path.exists() and full_path.is_file():
            try:
                existing = full_path.read_text(encoding=self.ENCODING)
            except Exception:
                existing = None

        # Reject writes that would newly break a parseable .py file.
        rejection = self._syntax_rejection(
            file_path, content, verb="Write", old_content=existing
        )
        if rejection is not None:
            return rejection

        try:
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            full_path.write_text(content, encoding=self.ENCODING)

            # Calculate stats
            lines = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
            bytes_written = len(content.encode(self.ENCODING))

            message = f"Successfully wrote to {file_path} ({lines} lines, {bytes_written} bytes)"

            # A whole-file rewrite that changes only a handful of lines
            # can never fit the generation cap — nudge toward edit_file. Only when
            # overwriting an existing file with a genuinely small delta.
            if existing is not None and existing != content:
                changed = self._changed_line_count(existing, content)
                if 0 < changed < 5:
                    message += (
                        " (tip: for small changes prefer edit_file with a minimal "
                        "old_string)"
                    )

            return message
        except Exception as e:
            return f"Error writing to {file_path}: {str(e)}"

    @staticmethod
    def _changed_line_count(old: str, new: str) -> int:
        """Number of lines that differ between old and new (a 1-line edit == 1)."""
        a = old.splitlines()
        b = new.splitlines()
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        changed = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                changed += max(i2 - i1, j2 - j1)
        return changed

    def _syntax_rejection(
        self,
        file_path: str,
        new_content: str,
        verb: str = "Edit",
        old_content: Optional[str] = None,
    ) -> Optional[str]:
        """If a .py file that CURRENTLY parses would become unparseable, return a
        rejection message (with the offending region); otherwise None.

        The check is guarded on the file having been parseable before the change:
        an already-broken file is left editable so the model can repair it.
        ``old_content`` is
        the pre-change content (None for a brand-new file, which is not gated).
        """
        if not self.validate_python_syntax:
            return None
        if Path(file_path).suffix != ".py":
            return None
        # Only guard files that parse today — never block repairs of broken ones.
        if old_content is None:
            return None
        try:
            ast.parse(old_content)
        except SyntaxError:
            return None
        try:
            ast.parse(new_content)
            return None
        except SyntaxError as err:
            lineno = err.lineno or 0
            new_lines = new_content.splitlines()
            lo = max(0, lineno - 4)
            hi = min(len(new_lines), lineno + 3)
            region = "\n".join(
                f"{i + 1:>4}| {new_lines[i]}" for i in range(lo, hi)
            )
            return (
                f"{verb} rejected: it would make {file_path} unparseable: "
                f"line {lineno}: {err.msg}. File unchanged.\n{region}"
            )

    def _execute_edit_file(self, tool_call: ToolCall) -> str:
        """Edit a file by replacing exact string match with new content.

        Optional parameter ``anchor_line`` (1-based int): restrict the match
        search to a +/-20-line window around that line, so a match that is
        ambiguous file-wide can still be applied unambiguously. Backward
        compatible when absent (search spans the whole file).
        """
        file_path = tool_call.parameters.get('file_path')
        old_string = tool_call.parameters.get('old_string')
        new_string = tool_call.parameters.get('new_string')
        anchor_line = tool_call.parameters.get('anchor_line')
        replace_all = bool(tool_call.parameters.get('replace_all', False))

        if not file_path:
            return "Error: No file path specified"
        if old_string is None:
            return "Error: No old_string specified"
        if new_string is None:
            return "Error: No new_string specified"
        if old_string == new_string:
            # A terse form of this message gets misread as a
            # whitespace/verbatim-matching problem ("copy the exact text
            # including trailing spaces") or as "the file is already correct",
            # which causes identical-noop retry loops. Spell out the contract
            # and the way forward. Detectors key on the
            # "old_string == new_string" / "would make no changes" substrings —
            # keep both.
            return (
                "Error: NO-OP — the file was NOT modified. old_string == "
                "new_string (edit_file would make no changes). new_string "
                "must differ from old_string. Do NOT repeat this edit: "
                "re-read the current source lines first; if the fix is "
                "already in the file, run the tests instead of re-editing."
            )

        # Check if file is protected
        if self._is_file_protected(file_path):
            return f"Error: Cannot edit protected file {file_path}"

        full_path, error = self._validate_file_path(file_path, must_exist=True, must_be_file=True)
        if error:
            return error

        try:
            # Read current content
            content = full_path.read_text(encoding=self.ENCODING)

            # Resolve the search window (whole file when anchor_line is absent).
            lo_off, hi_off, window = self._window_offsets(content, anchor_line)
            region = content[lo_off:hi_off]

            # Uniqueness is enforced only on the whole-file path: an explicit
            # anchor_line is an explicit scope (replace-within-window),
            # and replace_all is an explicit opt-in.
            matched = self._match_in_region(
                region, old_string, new_string, file_path,
                replace_all=replace_all or window is not None,
            )
            if matched is not None and matched[0] is None:
                # Ambiguous match — hard error, do not fall through to the
                # tolerant tiers (they would re-detect the same ambiguity).
                return matched[1]
            if matched is None:
                # No exact or tolerant match — return the CLOSEST matching region
                # so the retry is informed, not another guess.
                return self._format_no_match_error(
                    file_path, content, old_string, window=window
                )

            new_region, message = matched
            new_content = content[:lo_off] + new_region + content[hi_off:]

            # Reject edits that would newly break a parseable .py file.
            rejection = self._syntax_rejection(
                file_path, new_content, verb="Edit", old_content=content
            )
            if rejection is not None:
                return rejection

            full_path.write_text(new_content, encoding=self.ENCODING)
            return message
        except Exception as e:
            return f"Error editing {file_path}: {str(e)}"

    def _window_offsets(
        self, content: str, anchor_line, radius: int = 20
    ) -> Tuple[int, int, Optional[Tuple[int, int]]]:
        """Return (start_offset, end_offset, (lo_line, hi_line)) for a +/-radius
        line window around a 1-based anchor_line, or (0, len, None) when no
        (valid) anchor is given. lo_line/hi_line are 0-based, hi exclusive."""
        if anchor_line is None:
            return 0, len(content), None
        try:
            anchor = int(anchor_line)
        except (TypeError, ValueError):
            return 0, len(content), None

        lines = content.splitlines(keepends=True)
        total = len(lines)
        if total == 0:
            return 0, len(content), None
        anchor = max(1, min(anchor, total))  # clamp into range
        lo_line = max(0, anchor - 1 - radius)
        hi_line = min(total, anchor - 1 + radius + 1)
        lo_off = sum(len(x) for x in lines[:lo_line])
        hi_off = sum(len(x) for x in lines[:hi_line])
        return lo_off, hi_off, (lo_line, hi_line)

    def _match_in_region(
        self, region: str, old_string: str, new_string: str, display_name: str,
        replace_all: bool = False,
    ) -> Optional[Tuple[Optional[str], str]]:
        """Try exact then whitespace- then indent-tolerant matching within region.

        Returns (new_region, success_message), (None, error_message) for an
        AMBIGUOUS match (>1 occurrence without replace_all), or
        None if nothing matched. Does NOT write — the caller centralizes
        writing (and syntax validation).

        Replacing all occurrences would hide which match the model meant. The
        old_string can match both the buggy method and a healthy sibling
        (`return sum(s.within ...)` in total_within and, as the bug, in
        total_breached), and replacing both breaks the healthy one. A
        multi-match is a hard error unless replace_all=true; anchor_line gives
        positional disambiguation.
        """
        if old_string in region:
            count = region.count(old_string)
            if count > 1 and not replace_all:
                # Single-rule contract: the corrective message teaches exactly
                # ONE recovery, which is to widen old_string with verbatim
                # neighboring lines. anchor_line and replace_all work but are
                # not advertised; extra modes add load for the model.
                return (
                    None,
                    f"Error: old_string appears {count} times in "
                    f"{display_name} (ambiguous, nothing changed). Copy MORE "
                    "surrounding lines into old_string (verbatim) until it "
                    "matches exactly one place, then retry.",
                )
            return (
                region.replace(old_string, new_string),
                f"Successfully replaced {count} occurrence(s) in {display_name}",
            )

        # Tier 2: tolerate trailing-whitespace / line-ending drift. Leading
        # indentation is kept STRICT (rstrip is right-side only) so we never
        # apply an edit at the wrong indentation level — the model frequently
        # gets trailing whitespace or CRLF vs LF wrong, and a 1-char mismatch
        # would otherwise force a doomed retry loop.
        applied = self._apply_whitespace_tolerant_edit(
            region, old_string, new_string, display_name
        )
        if applied is not None:
            return applied

        # Tier 3: tolerate LEADING-indentation drift. The model often reproduces
        # the right lines at the wrong indent level, and it often ignores a
        # 99%-similar closest-match hint and resends the same text. We
        # auto-apply only when the
        # stripped content matches a UNIQUE block and every line shares the SAME
        # indent delta, then re-indent new_string by that delta so the edit still
        # lands at the file's (correct) indentation.
        applied = self._apply_indent_tolerant_edit(
            region, old_string, new_string, display_name
        )
        if applied is not None:
            return applied

        return None

    def _apply_whitespace_tolerant_edit(
        self, content: str, old_string: str, new_string: str, display_name: str
    ) -> Optional[Tuple[str, str]]:
        """Match old_string against a UNIQUE block of lines ignoring only trailing
        whitespace and line-ending style. Returns (new_content, message) or None.

        Leading indentation is matched strictly (we rstrip the right side only),
        so the edit is never applied at the wrong indentation.
        """
        content_lines = content.splitlines(keepends=True)
        old_lines = old_string.splitlines()  # drops line endings, keeps indent
        n = len(old_lines)
        if n == 0:
            return None
        old_trimmed = [ln.rstrip() for ln in old_lines]

        hits = [
            i
            for i in range(len(content_lines) - n + 1)
            if [content_lines[j].rstrip() for j in range(i, i + n)] == old_trimmed
        ]
        if len(hits) != 1:
            return None

        i = hits[0]
        real_old = "".join(content_lines[i : i + n])
        # Preserve the trailing newline of the matched block if new_string omits
        # one (old_lines was split without keepends).
        last = content_lines[i + n - 1]
        trailing_nl = last[len(last.rstrip("\r\n")) :]
        repl = new_string
        if trailing_nl and not repl.endswith(("\n", "\r")):
            repl = repl + trailing_nl

        off = sum(len(x) for x in content_lines[:i])
        new_content = content[:off] + repl + content[off + len(real_old) :]
        return (
            new_content,
            "Successfully replaced 1 occurrence(s) in "
            f"{display_name} (matched ignoring trailing whitespace / line "
            "endings).",
        )

    def _apply_indent_tolerant_edit(
        self, content: str, old_string: str, new_string: str, display_name: str
    ) -> Optional[Tuple[str, str]]:
        """Apply an edit when old_string matches a UNIQUE block of file lines
        ignoring per-line leading/trailing whitespace, PROVIDED every matched
        line shares the same leading-indent delta (a uniform shift).

        This catches the dominant doomed-retry case: the model copies the right
        lines but at the wrong indentation. We never apply when the stripped
        content differs (a genuine content mismatch still bounces to the model),
        when the match is non-unique, or when the indent delta is inconsistent
        across lines (which would signal a structural mismatch, not a shift).
        new_string is re-indented by the same delta so the result lands at the
        file's indentation, not the model's. Returns the success message or None.
        """
        content_lines = content.splitlines(keepends=True)
        old_lines = old_string.splitlines()
        n = len(old_lines)
        if n == 0:
            return None

        old_stripped = [ln.strip() for ln in old_lines]
        # Skip if old_string is entirely blank — too ambiguous to place.
        if not any(old_stripped):
            return None

        def _leading_ws(s: str) -> str:
            return s[: len(s) - len(s.lstrip())]

        hits = [
            i
            for i in range(len(content_lines) - n + 1)
            if [content_lines[j].strip() for j in range(i, i + n)] == old_stripped
        ]
        if len(hits) != 1:
            return None

        i = hits[0]
        block = [content_lines[i + k] for k in range(n)]

        # Require a single, consistent indent delta across all non-blank lines.
        # Collect (file_indent, old_indent) pairs; the file indent must be the
        # old indent plus a constant prefix delta (positive = add, "-" = strip).
        pairs = [
            (_leading_ws(f), _leading_ws(o))
            for o, f in zip(old_lines, block)
            if o.strip()
        ]
        prefixes = set()
        for f_ind, o_ind in pairs:
            if not f_ind.endswith(o_ind) and not o_ind.endswith(f_ind):
                # indentation isn't a clean prefix relationship — bail.
                return None
            prefixes.add(f_ind[: len(f_ind) - len(o_ind)] if len(f_ind) >= len(o_ind)
                         else "-" + o_ind[: len(o_ind) - len(f_ind)])
        if len(prefixes) != 1:
            return None
        delta = next(iter(prefixes))

        # Re-indent new_string lines by the same delta.
        repl_lines = new_string.splitlines(keepends=True)
        rebuilt = []
        for ln in repl_lines:
            if not ln.strip():
                rebuilt.append(ln)
                continue
            if delta.startswith("-"):
                strip_n = len(delta) - 1
                lead = _leading_ws(ln)
                cut = min(strip_n, len(lead))
                rebuilt.append(ln[cut:])
            else:
                rebuilt.append(delta + ln)
        repl = "".join(rebuilt)

        # Preserve the trailing newline of the matched block if new_string omits one.
        last = block[-1]
        trailing_nl = last[len(last.rstrip("\r\n")):]
        if trailing_nl and not repl.endswith(("\n", "\r")):
            repl = repl + trailing_nl

        real_old = "".join(block)
        off = sum(len(x) for x in content_lines[:i])
        new_content = content[:off] + repl + content[off + len(real_old):]
        return (
            new_content,
            "Successfully replaced 1 occurrence(s) in "
            f"{display_name} (matched ignoring indentation; new_string "
            "re-indented to the file's indentation).",
        )

    def _format_no_match_error(
        self,
        file_path: str,
        content: str,
        old_string: str,
        window: Optional[Tuple[int, int]] = None,
    ) -> str:
        """Build a 'String not found' error that includes numbered file lines the
        model can copy old_string from verbatim.

        Shows the whole file when it is <=60 lines, otherwise the +/-8 lines
        around the closest matching region. When ``window`` (a 0-based
        [lo, hi) line range from an anchor_line) is given, the closest-match
        search is confined to it and the restriction is noted. No rg/grep hint
        is emitted.
        """
        lines = content.splitlines(keepends=True)
        total = len(lines)
        old_lines = old_string.splitlines() or [old_string]
        n = max(1, len(old_lines))

        anchor_note = ""
        if window is not None:
            anchor_note = (
                f" Search was restricted to the anchor_line window "
                f"(lines {window[0] + 1}-{window[1]})."
            )

        # Find the closest block for both small and large files. A short,
        # explicit comparison makes a missing argument visible before the
        # surrounding source excerpt, which the model may skim or truncate.
        search_lo = window[0] if window is not None else 0
        search_hi = (window[1] if window is not None else total) - n + 1
        search_lo = max(0, min(search_lo, max(0, total - n)))
        search_hi = max(search_lo + 1, min(search_hi, total - n + 1))
        best_i, best_ratio = search_lo, -1.0
        sm = difflib.SequenceMatcher()
        sm.set_seq2(old_string)
        for i in range(search_lo, search_hi):
            sm.set_seq1("".join(lines[i : i + n]))
            ratio = sm.ratio()
            if ratio > best_ratio:
                best_ratio, best_i = ratio, i

        mismatch_note = ""
        actual_lines = lines[best_i : best_i + n]
        if best_ratio >= 0.75 and len(actual_lines) == n:
            for offset, (requested, actual) in enumerate(zip(old_lines, actual_lines)):
                actual = actual.rstrip("\r\n")
                if requested != actual:
                    mismatch_note = (
                        f"\nFirst differing line near the closest match "
                        f"(old_string line {offset + 1}, "
                        f"{file_path} line {best_i + offset + 1}):"
                        f"\n  requested: {requested[:200]!r}"
                        f"\n  actual:    {actual[:200]!r}"
                    )
                    break

        if total <= 60:
            numbered = "".join(f"{i + 1:>4}| {lines[i]}" for i in range(total))
            region_desc = f"Current content of {file_path} ({total} lines)"
        else:
            ctx_start = max(0, best_i - 8)
            ctx_end = min(total, best_i + n + 8)
            numbered = "".join(
                f"{j + 1:>4}| {lines[j]}" for j in range(ctx_start, ctx_end)
            )
            region_desc = (
                f"Closest match in {file_path} (lines {ctx_start + 1}-{ctx_end}, "
                f"~{best_ratio:.0%} similar)"
            )

        return (
            f"Error: String not found in {file_path} ({total} lines) — the "
            f"file was NOT modified."
            f"{anchor_note} old_string must match exactly, including whitespace "
            f"and indentation.{mismatch_note}\n{region_desc}:\n{numbered}"
            f"\nRetry with the shortest unique old_string copied from the ACTUAL "
            f"file lines above (often one line). Put the intended code only "
            f"in new_string; do not reuse the requested text that failed."
        )

    # ------------------------------------------------------------------ #
    # snapshot + revert capability
    # ------------------------------------------------------------------ #
