import hashlib
import os
import pty
import re
import select
import shlex
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple


# Range-read commands whose empty output usually means "start line is past EOF"
# rather than "the command genuinely produced nothing". Each pattern captures
# (start_line, file) so we can report the file's true length.
_RANGE_READ_PATTERNS = (
    # sed -n '71,74p' FILE   /   sed -n '71p' FILE
    re.compile(r"\bsed\b[^|;]*?-n\b[^|;]*?['\"]?(\d+)(?:,\d*)?p['\"]?\s+(\S+)"),
    # tail -n +71 FILE  (print starting AT line 71)
    re.compile(r"\btail\b[^|;]*?-n\s*\+(\d+)\s+(\S+)"),
    # awk 'NR>=71 ...' FILE  /  awk 'NR==71' FILE
    re.compile(r"\bawk\b[^|;]*?['\"][^'\"]*?NR\s*(?:==|>=|>)\s*(\d+)[^'\"]*['\"]\s+(\S+)"),
)

# Pure single-file read commands (no pipes/compounds/redirects). For these,
# oversized output is truncated to a contiguous HEAD with a continue-from-here
# hint instead of the default middle omission: a middle window hides exactly
# the lines the model asked to read (the buggy function can sit inside the
# omitted middle of a `sed -n '1,150p'` read). Middle omission remains
# correct for command output like pytest, where the head (first traceback) and
# tail (summary) are the high-value parts.
_FILE_READ_PATTERNS = (
    # (start_line_group_present, pattern) — group "start" is the first file
    # line printed when the command addresses a range; absent means line 1.
    re.compile(r"^\s*cat\s+(?:-[A-Za-z]+\s+)*(?P<file>[^-\s]\S*)\s*$"),
    re.compile(r"^\s*head\s+(?:-\S+\s+)*(?P<file>[^-\s]\S*)\s*$"),
    re.compile(
        r"^\s*tail\s+(?:-n\s*\+(?P<start>\d+)\s+|-\S+\s+)*(?P<file>[^-\s]\S*)\s*$"
    ),
    re.compile(
        r"^\s*sed\s+-n\s+['\"]?(?P<start>\d+)(?:,\d+)?p['\"]?\s+(?P<file>[^-\s]\S*)\s*$"
    ),
    re.compile(
        r"^\s*awk\s+'[^']*NR\s*(?:==|>=|>)\s*(?P<start>\d+)[^']*'\s+(?P<file>[^-\s]\S*)\s*$"
    ),
)

_COMPOUND_TOKENS = ("|", ";", "&&", "||", ">", "<", "`", "$(")


def _file_read_target(command: str) -> Optional[Tuple[str, int]]:
    """Return ``(file, first_file_line_printed)`` for pure single-file reads."""
    if any(tok in command for tok in _COMPOUND_TOKENS):
        return None
    for pattern in _FILE_READ_PATTERNS:
        match = pattern.match(command)
        if match:
            start = 1
            try:
                if match.group("start"):
                    start = int(match.group("start"))
            except (IndexError, ValueError):
                start = 1
            return match.group("file").strip("'\""), start
    return None


_ANSI_ESCAPE_RE = re.compile(
    r"""
    (?:\x1B\[ [0-?]* [ -/]* [@-~])   # CSI ... Cmd
    |(?:\x1B\] .*? (?:\x07|\x1B\\))  # OSC ... (BEL or ST)
    |(?:\x1B[@-_])                  # 2-char escape sequences
    """,
    re.VERBOSE | re.DOTALL,
)


def _strip_ansi_and_control(text: str) -> str:
    """
    Remove ANSI escape sequences and most control characters.

    Keeps newlines and tabs to preserve readability of command output.
    """
    if not text:
        return text

    text = _ANSI_ESCAPE_RE.sub("", text)
    # Drop remaining control chars (incl. ESC) except \n and \t
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or (ord(ch) >= 32 and ch != "\x7f"))


class BashSession:
    """Maintain a persistent bash session over a PTY to keep shell state."""

    def __init__(
        self,
        root_directory: Path,
        *,
        timeout: float = 30.0,
        max_output_lines: int = 100,
        max_output_bytes: int = 8192
    ):
        self.root_directory = root_directory
        self.timeout = timeout
        self.max_output_lines = max_output_lines
        self.max_output_bytes = max_output_bytes
        self._prompt_token = f"__BASH_PROMPT_{uuid.uuid4().hex}__"
        self._prompt = f"{self._prompt_token} "
        self._exit_code_token = f"__BASH_EXIT_CODE_{uuid.uuid4().hex}__:"
        self._master_fd: Optional[int] = None
        self._process: Optional[subprocess.Popen] = None
        self._closed = False
        # Duplicate-output ledger: command -> (first_seen_exec_index, sha1(body
        # + exit code), repeat_count). Re-running a command whose output is
        # byte-identical to its previous run compresses the observation to a
        # short "nothing changed" note. This stops the model from spending
        # many turns on the same `sed` re-read.
        self._dup_history: Dict[str, Tuple[int, str, int]] = {}
        self._exec_count = 0
        self._spawn()

    def _spawn(self) -> None:
        """Start (or restart) the PTY-backed bash process and initialize it.

        Factored out of __init__ so a bricked shell (killed by the model's own
        `killall`/`kill $$`, or a hung command we had to tear down) can be
        respawned in place; see the self-heal in execute()."""
        self._master_fd, slave_fd = pty.openpty()
        env = os.environ.copy()
        env["PS1"] = self._prompt
        env["PS2"] = self._prompt
        self._process = subprocess.Popen(
            ["/bin/bash", "--noprofile", "--norc"],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=str(self.root_directory),
            bufsize=0,
            close_fds=True,
            # Own session/process-group so a runaway command's whole tree can be
            # killed on timeout without touching the daemon/host process.
            start_new_session=True,
            env=env
        )
        os.close(slave_fd)
        self._closed = False
        self._initialize_shell()

    def _kill_foreground_job(self) -> None:
        """SIGKILL the PTY's foreground process group if it differs from the
        shell's own.

        Interactive bash with job control puts each foreground pipeline in its
        OWN process group, so killpg(shell_pgid) on timeout misses the running
        command, and timed-out pytest runs survive shell restarts as orphans
        (they can use a lot of memory). tcgetpgrp() names the foreground group
        directly; kill it before tearing down the shell.
        """
        if self._master_fd is None or self._process is None:
            return
        try:
            fg = os.tcgetpgrp(self._master_fd)
            shell_pgid = os.getpgid(self._process.pid)
            if fg > 0 and fg != shell_pgid:
                os.killpg(fg, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def _restart_shell(self) -> None:
        """Tear down the current bash (and its process group) and spawn a fresh
        one, so the NEXT command runs cleanly even after a brick."""
        self._kill_foreground_job()
        proc = self._process
        if proc is not None:
            try:
                if proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    try:
                        proc.wait(timeout=3)
                    except Exception:
                        pass
            except Exception:
                pass
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None
        self._spawn()

    def _initialize_shell(self) -> None:
        """Configure the shell prompt and disable local echo."""
        # Drain the initial bash prompt
        self._read_until_prompt()
        prompt_setup = f"PS1=\"{self._prompt}\"; PS2=\"{self._prompt}\"; export PS1 PS2\n"
        self._send(prompt_setup)
        self._read_until_prompt()
        # Make pipeline exit codes reflect upstream failures when possible.
        self._send("set -o pipefail\n")
        self._read_until_prompt()
        # Disable history expansion so commands containing '!' (e.g. `[ ! -f x ]`,
        # `grep '!important'`) don't fail with "event not found".
        self._send("set +H\n")
        self._read_until_prompt()
        self._send("stty -echo\n")
        self._read_until_prompt()

    def _send(self, payload: str) -> None:
        """Send a string payload to the bash shell."""
        if self._closed or self._process.poll() is not None:
            raise RuntimeError("Bash session has already terminated")
        os.write(self._master_fd, payload.encode("utf-8"))

    def _read_until_prompt(self, timeout: Optional[float] = None) -> str:
        """Read output from the PTY until the custom prompt is encountered."""
        deadline = time.time() + (timeout or self.timeout)
        buffer = bytearray()
        prompt_bytes = self._prompt.encode("utf-8")

        while time.time() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError("Bash session terminated unexpectedly")

            ready, _, _ = select.select([self._master_fd], [], [], 0.1)
            if not ready:
                continue

            chunk = os.read(self._master_fd, 4096)
            if not chunk:
                break

            buffer.extend(chunk)
            if buffer.endswith(prompt_bytes):
                break
        else:
            raise TimeoutError(
                f"Command execution timed out (>{int(self.timeout)} seconds)"
            )

        if buffer.endswith(prompt_bytes):
            buffer = buffer[:-len(prompt_bytes)]

        decoded = buffer.decode("utf-8", errors="ignore")
        return decoded.replace("\r", "")

    def execute(self, raw_command: Optional[str]) -> str:
        """Execute a bash command in the persistent shell session."""
        if not raw_command:
            return "Error: No command specified for bash"

        command = raw_command.strip()
        if not command:
            return "Error: No command specified for bash"

        # Self-heal: if a prior command bricked the shell (e.g. `killall python`,
        # `kill $$`, or a hung command we tore down), respawn it so THIS command
        # runs cleanly instead of erroring for the rest of the session.
        if self._closed or self._process is None or self._process.poll() is not None:
            self._restart_shell()

        timeout_msg = f"Command execution timed out (>{int(self.timeout)} seconds)"
        try:
            # Send one physical line to avoid intermediate PTY prompts. ANSI-C
            # quoting transports the command unchanged (including quoted newlines,
            # indentation, comments and heredocs); eval runs it in this shell so
            # variables and functions still persist between calls.
            escaped_command = (
                command.replace("\\", "\\\\")
                .replace("'", "\\'")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )
            # Reset cwd to the workspace root at the START of every command so a
            # stray `cd` in a prior call can't poison later file/debugger tools
            # (which resolve against a fixed root). `cd` is still usable WITHIN
            # this command via a compound (e.g. `cd sub && pytest`). The reset's
            # own exit status is discarded — __BASH_EC reflects the user command.
            reset_cd = f"cd {shlex.quote(str(self.root_directory))} 2>/dev/null ; "
            wrapped = (
                f"{reset_cd}"
                f"eval -- $'{escaped_command}'; "
                "__BASH_EC=$?; "
                f"echo \"{self._exit_code_token}$__BASH_EC\""
            )
            self._send(wrapped + "\n")
            output = self._read_until_prompt()
        except TimeoutError:
            # The command is still running and holding the PTY — a fresh command
            # would never see a clean prompt. Kill the whole tree and respawn so
            # the NEXT tool call works.
            self._restart_shell()
            return timeout_msg
        except RuntimeError:
            # The shell died mid-command (e.g. the command killed it). Respawn so
            # the next call is clean; report the reset rather than a raw error.
            self._restart_shell()
            return (
                "Error: the previous command terminated the shell; it has been "
                "reset. Re-run your command."
            )
        except Exception as exc:
            return f"Error running command: {str(exc)}"

        exit_code: Optional[int] = None
        cleaned = output.lstrip("\n")
        cleaned = _strip_ansi_and_control(cleaned)
        # ANSI stripping can turn a leading control-only line (e.g. bracketed
        # paste toggles some PTYs emit) into an empty first line — re-strip so
        # observations don't start with a blank line (varies by node).
        cleaned = cleaned.lstrip("\n")

        # Extract the exit-code marker emitted by the harness wrapper.
        if cleaned:
            lines = cleaned.split("\n")
            marker_idx = None
            marker_val = None
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if line.startswith(self._exit_code_token):
                    marker_idx = i
                    marker_val = line[len(self._exit_code_token):].strip()
                    break
            if marker_idx is not None:
                try:
                    exit_code = int(marker_val) if marker_val is not None else None
                except Exception:
                    exit_code = None
                del lines[marker_idx]
                cleaned = "\n".join(lines).rstrip("\n")

        exit_code_line = f"Exit code: {exit_code}" if exit_code is not None else "Exit code: unknown"
        if not cleaned.strip():
            note = self._eof_range_note(command)
            suffix = f"\n{note}" if note else ""
            return f"Command executed with no output{suffix}\n\n{exit_code_line}"

        duplicate = self._register_and_check_duplicate(command, cleaned, exit_code_line)
        if duplicate is not None:
            return duplicate

        return self._apply_truncation(f"{cleaned}\n\n{exit_code_line}", command)

    # Outputs smaller than this pass through even when repeated — compressing
    # a 3-line `ls` buys nothing and the full text is the clearer observation.
    _DUP_MIN_BYTES = 400
    _DUP_MIN_LINES = 8
    _DUP_HISTORY_MAX = 200

    def _register_and_check_duplicate(
        self, command: str, body: str, exit_code_line: str
    ) -> Optional[str]:
        """Compress the observation when a re-run reproduces identical output.

        The command still EXECUTES normally every time (side effects included);
        only the returned text is compressed, and only when the output is
        byte-identical to this exact command's previous run. The final lines
        (a pytest summary, the tail of a file) are kept so the observation
        stays grounded even if the earlier copy was truncated out of context.
        """
        body_hash = hashlib.sha1(
            (body + "\n" + exit_code_line).encode("utf-8")
        ).hexdigest()
        self._exec_count += 1
        prev = self._dup_history.get(command)
        is_big = (
            len(body.encode("utf-8")) > self._DUP_MIN_BYTES
            or body.count("\n") + 1 > self._DUP_MIN_LINES
        )
        if prev is not None and prev[1] == body_hash and is_big:
            repeat_n = prev[2] + 1
            self._dup_history[command] = (prev[0], body_hash, repeat_n)
            tail_lines = body.rstrip("\n").split("\n")[-5:]
            tail_txt = "\n".join(tail_lines)
            return (
                f"[Output is byte-identical to your previous run of this exact "
                f"command (repeat #{repeat_n}) — nothing this command reads has "
                f"changed. Re-running it again will not produce new "
                f"information; try a different command, a different line "
                f"range, or make an edit first. Final {len(tail_lines)} lines:]\n"
                f"{tail_txt}\n\n{exit_code_line}"
            )
        if prev is not None and prev[1] == body_hash:
            return None  # small output: pass through, but keep the ledger entry
        if len(self._dup_history) >= self._DUP_HISTORY_MAX:
            self._dup_history.pop(next(iter(self._dup_history)))
        self._dup_history[command] = (self._exec_count, body_hash, 0)
        return None

    def _eof_range_note(self, command: str) -> Optional[str]:
        """For range-read commands (sed -n 'A,Bp', tail -n +A, awk NR>=A) that
        produced no output, return a note when the target file has fewer than A
        lines — i.e. the requested range starts beyond EOF. Best
        effort: any parse/IO issue simply yields no note (never raises)."""
        for pattern in _RANGE_READ_PATTERNS:
            match = pattern.search(command)
            if not match:
                continue
            try:
                start = int(match.group(1))
                raw_name = match.group(2).strip().strip("'\"")
            except (ValueError, IndexError):
                continue
            if not raw_name or raw_name.startswith("-"):
                continue
            candidate = Path(raw_name)
            if not candidate.is_absolute():
                candidate = self.root_directory / candidate
            try:
                if not candidate.is_file():
                    continue
                with candidate.open("r", encoding="utf-8", errors="ignore") as fh:
                    line_count = sum(1 for _ in fh)
            except OSError:
                continue
            if line_count < start:
                return (
                    f"note: {raw_name} has {line_count} lines — the requested "
                    f"range starts beyond EOF"
                )
        return None

    # Middle-truncation budget: keep this many lines from the START and END of
    # oversized output so the FIRST traceback and the FINAL summary (pytest's
    # short-summary / pass-fail line) both survive.
    _TRUNCATE_HEAD_LINES = 60
    _TRUNCATE_TAIL_LINES = 40

    def _apply_truncation(self, output: str, command: str) -> str:
        encoded = output.encode("utf-8")
        lines = output.split("\n")

        # Preserve trailing exit-code line under truncation.
        exit_code_line = None
        if lines and lines[-1].startswith("Exit code:"):
            exit_code_line = lines.pop()
            body = "\n".join(lines)
        else:
            body = output

        body_lines = body.split("\n")
        needs_truncation = (
            len(body_lines) > self.max_output_lines
            or len(encoded) > self.max_output_bytes
        )

        if not needs_truncation:
            if exit_code_line is None:
                return output
            return body + "\n" + exit_code_line

        total_body = len(body_lines)
        total_bytes = len(encoded)
        output_hash = hashlib.sha1(encoded).hexdigest()[:16]

        # Pure file reads keep a contiguous HEAD + a continue-from-here hint;
        # middle omission would hide exactly the lines the model asked for.
        read_target = _file_read_target(command)
        if read_target is not None:
            target_file, first_file_line = read_target
            byte_budget = self.max_output_bytes
            if exit_code_line is not None:
                byte_budget = max(
                    0, byte_budget - len(("\n" + exit_code_line).encode("utf-8"))
                )
            # Drop PTY-echo blank lines at the top so the kept window maps 1:1
            # onto file lines and the continue-hint arithmetic stays honest.
            read_lines = list(body_lines)
            while read_lines and not read_lines[0].strip():
                read_lines.pop(0)
            total_body = len(read_lines)
            kept = read_lines[: self.max_output_lines]

            def _read_marker(n_kept: int) -> str:
                omitted = total_body - n_kept
                next_from = first_file_line + n_kept
                next_to = next_from + self.max_output_lines - 1
                return (
                    f"<TOOL_OUTPUT_TRUNCATED: {omitted} trailing lines omitted "
                    f"(shown: lines {first_file_line}-"
                    f"{first_file_line + n_kept - 1} of {target_file}); "
                    f"continue reading with e.g. "
                    f"sed -n '{next_from},{next_to}p' {target_file}; "
                    f"bytes={total_bytes} sha1={output_hash}>"
                )

            combined = "\n".join(kept + [_read_marker(len(kept))])
            while len(combined.encode("utf-8")) > byte_budget and kept:
                kept = kept[:-1]
                combined = "\n".join(kept + [_read_marker(len(kept))])
            if exit_code_line is not None:
                combined = combined.rstrip("\n") + "\n" + exit_code_line
            return combined

        head_n = self._TRUNCATE_HEAD_LINES
        tail_n = self._TRUNCATE_TAIL_LINES
        # Scale the head/tail budget down if the line cap is smaller than 60+40,
        # keeping the ~60/40 split (at least one line each).
        if head_n + tail_n > self.max_output_lines:
            head_n = max(1, int(self.max_output_lines * 0.6))
            tail_n = max(1, self.max_output_lines - head_n)

        if total_body > head_n + tail_n:
            head_part = body_lines[:head_n]
            tail_part = body_lines[-tail_n:]
        else:
            # Byte overflow only (too few lines to drop a middle by count). Split
            # into a non-overlapping head and tail so the byte-trim loop below can
            # still preserve the FINAL summary line, not just the head.
            split = total_body // 2
            head_part = body_lines[:split]
            tail_part: list = body_lines[split:]

        cmd_disp = command[:100] + ("..." if len(command) > 100 else "")

        def _marker(omitted: int) -> str:
            # Leads with the middle-omission note and keeps the
            # command=/lines=/bytes=/sha1= fields that callers and tests read.
            return (
                f"<TOOL_OUTPUT_TRUNCATED: {omitted} middle lines omitted; "
                f"command=\"{cmd_disp}\" lines={total_body} "
                f"bytes={total_bytes} sha1={output_hash}>"
            )

        def _assemble(head: list, tail: list) -> str:
            omitted = total_body - len(head) - len(tail)
            return "\n".join(head + [_marker(omitted)] + tail)

        # Reserve room for the exit-code line so it is never dropped.
        budget = self.max_output_bytes
        if exit_code_line is not None:
            budget = max(0, budget - len(("\n" + exit_code_line).encode("utf-8")))

        combined = _assemble(head_part, tail_part)
        # Trim the HEAD first (the tail summary is the higher-value part), then
        # the tail from the front, until the body fits the byte budget.
        while len(combined.encode("utf-8")) > budget and head_part:
            head_part = head_part[:-1]
            combined = _assemble(head_part, tail_part)
        while len(combined.encode("utf-8")) > budget and tail_part:
            tail_part = tail_part[1:]
            combined = _assemble(head_part, tail_part)
        if len(combined.encode("utf-8")) > budget:
            combined = combined.encode("utf-8")[:budget].decode(
                "utf-8", errors="ignore"
            )

        if exit_code_line is not None:
            combined = combined.rstrip("\n") + "\n" + exit_code_line
        return combined

    def close(self) -> None:
        """Terminate the bash session and clean up resources.

        Interactive bash (it owns a PTY) IGNORES SIGTERM, so we SIGKILL the whole
        process group — this also reaps any child the shell was running and keeps
        teardown fast instead of blocking on a doomed terminate()+wait.

        NOTE: the shell's group does NOT include the current foreground job
        (job control gives it its own pgid) — kill that first, same as
        _restart_shell (fix #6)."""
        if self._closed:
            return

        self._kill_foreground_job()
        proc = self._process
        if proc is not None:
            try:
                if proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    try:
                        proc.wait(timeout=3)
                    except Exception:
                        pass
            except Exception:
                pass

        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None

        self._closed = True
