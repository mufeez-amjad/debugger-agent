"""Run the model against a debugging workspace and yield events for the UI.

The server uses common/chat_template_json.jinja and the Hermes tool-call parser.
Servers without the parser are supported by parsing <tool_call> blocks locally.
Assistant history stores structured tool calls separately from reasoning so the
chat template renders each call once."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urljoin

# Repo root on path so common/* and harness/* import when run as a script.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.context_utils import truncate_messages
from common.models import ToolCall as HarnessToolCall
from common.native_tools import build_assistant_message, build_tool_message, strip_tool_call_block
from common.output_markers import FILE_EDIT_TOOLS, file_edit_applied
from common.observation_hints import (
    ObservationHinter,
    pytest_collected_nothing,
    pytest_collection_errored,
)
from common.prompts import (
    SIMPLE_DEBUG_SYSTEM_PROMPT,
    WORKSPACE_LINE,
    build_initial_observation,
)
from common.tools import get_openai_tools
from harness.debug_session import DebugSession

from . import events
from .tokenization import load_tokenizer

# The model emits one tool call per turn; this stop string ends the turn there.
_STOP = ["</tool_call>"]

DEFAULT_MODEL_URL = "http://localhost:8000"
# The --served-model-name in inference_server/modal_app.py.
DEFAULT_MODEL_NAME = "llm-debugger"

# Modal flash endpoints return 503 while a container cold-boots instead of
# holding the request, so the client owns the retry. Snapshot restores are
# ~40 s; a recording boot (first after each deploy) can take several minutes.
_COLD_START_RETRY_SECONDS = 15.0
_COLD_START_DEADLINE_SECONDS = 900.0

# Server context window: MAX_MODEL_LEN in inference_server/modal_app.py. vLLM
# rejects prompt+completion over this with a hard 400. A verbatim retry would
# fail the same way, so once a session crossed the window EVERY remaining turn
# would fail. Overridable for a local server with a different window. This value
# must never EXCEED the server's, or the 400 returns; below is merely wasteful.
_CONTEXT_LIMIT_TOKENS = 32768
# Chat-template scaffolding (role headers, tool-call wrappers, the structural
# tag) that the char estimator below never sees.
_TEMPLATE_OVERHEAD_TOKENS = 256
# The same reserve when counting exactly (`tokenization.load_tokenizer`). Most
# of the 256 above is scaffolding the estimator is blind to but the rendered
# template already contains, so what's left to reserve is only what the count
# still can't see: the generation prompt (the contract renders with
# add_generation_prompt=False) and the structural tag prepended under guided
# decoding. Both are tens of tokens; 64 keeps a margin without spending 3.5% of
# the window on scaffolding we already counted.
_EXACT_TEMPLATE_OVERHEAD_TOKENS = 64
# Chars per token before the server has told us the real ratio. Debug
# transcripts are traceback- and source-heavy and go on the wire as escaped
# JSON, which tokenizes far denser than the 6.0 chars/token
# common.context_utils assumes for prose. 3.0 is the pessimistic end of the
# plausible range for this tokenizer rather than a measurement — it only has to
# be low enough that the first request of a session fits, since any overflow
# replaces it with a number derived from the server's own count.
_INITIAL_CHARS_PER_TOKEN = 3.0
_CONTEXT_RETRIES = 3

class _ContextOverflow(RuntimeError):
    """A 400 from the server saying the prompt exceeded the context window.

    The body carries both numbers we need: how many tokens the server counted
    in our prompt, and the window it is actually serving. The second one is the
    ground truth about the deployment — a client constant can only ever be a
    guess about what someone last deployed.
    """

    def __init__(
        self,
        message: str,
        input_tokens: Optional[int],
        max_context: Optional[int] = None,
    ):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.max_context = max_context


class _WireSizeEstimator:
    """Duck-typed tokenizer for ``common.context_utils.truncate_messages``.

    The CLI talks to a remote vLLM and deliberately carries no local tokenizer:
    loading one would add a transformers dependency and seconds of startup to
    every `debug` invocation. truncate_messages only calls
    ``apply_chat_template`` then ``encode`` and takes ``len()`` of the result,
    so a char-ratio estimator satisfies the contract.

    Its own chars//6 fallback is not usable here because it counts only
    ``content`` — blind to both the ``tool_calls`` arguments (whole file edits)
    and the 20-tool schema we send as ``tools=`` (~2k tokens, a third of the
    window). Serializing exactly what goes on the wire is what makes the
    ratio recalibrated in `_call_model` meaningful.
    """

    def __init__(self, chars_per_token: float, tools: Optional[List[Dict]] = None):
        self.chars_per_token = chars_per_token
        self._tools = tools

    @staticmethod
    def wire_chars(messages: List[Dict[str, Any]], tools: Optional[List[Dict]]) -> int:
        return len(json.dumps([messages, tools], default=str))

    def apply_chat_template(self, messages, tools=None, **_kwargs) -> str:
        return json.dumps([messages, tools if tools is not None else self._tools],
                          default=str)

    def encode(self, text: str, **_kwargs) -> List[int]:
        return [0] * int(len(text) / self.chars_per_token)


@dataclass
class AgentConfig:
    model_url: str = DEFAULT_MODEL_URL
    model_name: str = DEFAULT_MODEL_NAME
    # Bearer token for the endpoint. The Modal deployment is transport-public
    # but vLLM enforces --api-key, so this is required there; a local
    # server has no key and leaves it None.
    api_key: Optional[str] = None
    # The budget is never shown to the model, so a low cap silently cuts off
    # sessions that the model would finish.
    max_turns: int = 50
    temperature: float = 0.1
    max_tokens: int = 1024
    request_timeout: float = 300.0
    context_limit: int = _CONTEXT_LIMIT_TOKENS
    # Where to find tokenizer.json for exact prompt-token counting: a file, a
    # directory, or an HF repo id. None searches the local cache only. Falling
    # back to the char estimator is not an error — see `tokenization`.
    tokenizer: Optional[str] = None
    # Append the observation hints (common.observation_hints). ON by default:
    # a mistyped path or an empty pytest run gets a one-line correction appended
    # to the observation. Without it the model sees a bare error and has no
    # signal to change tactics, so a session can run to max_turns.
    hints: bool = True


class DebugAgent:
    """Runs a debugging session against a real directory and streams events."""

    def __init__(self, codebase_dir: str, prompt: str, config: AgentConfig):
        self.codebase_dir = str(Path(codebase_dir).resolve())
        self.prompt = prompt
        self.config = config
        self.debug_session: Optional[DebugSession] = None
        self._messages: List[Dict[str, Any]] = []
        self._files: List[str] = []
        self._test_file: Optional[str] = None
        self._last_test_passed: Optional[bool] = None
        # Recalibrated downward from the server's own token count on any
        # context-overflow 400; see `_call_model`. Unused when the exact
        # tokenizer below loads.
        self._chars_per_token: float = _INITIAL_CHARS_PER_TOKEN
        # Exact prompt-token counting when a tokenizer is reachable, estimation
        # otherwise. Built once: loading is milliseconds, but it happens per
        # `debug` invocation and the template compile is not free.
        self._tokenizer = load_tokenizer(config.tokenizer)
        self._overhead_tokens: int = (
            _EXACT_TEMPLATE_OVERHEAD_TOKENS
            if self._tokenizer is not None
            else _TEMPLATE_OVERHEAD_TOKENS
        )
        # The window we currently believe the server serves. Starts at what the
        # caller configured and shrinks to whatever a context-overflow 400
        # reports, so a stale client constant costs one retry rather than the
        # rest of the session.
        self._context_limit = config.context_limit
        # The hint layer the model saw in training. It also owns the
        # observation cap, so the truncation shape has one implementation.
        self._hinter = ObservationHinter(workspace_path=self.codebase_dir)
        # Turn of the last pytest run that reported real counts, and of the last
        # edit that actually landed. The rerun hint fires only when the model
        # reruns pytest with no edit in between, so the agent tracks both.
        self._turn_last_pytest = 0
        self._turn_last_edit = 0

    # ── public ─────────────────────────────────────────────────────

    def run(self) -> Iterator[events.Event]:
        """Drive the loop, yielding events until the task is done or exhausted."""
        try:
            yield from self._start()
            success = False
            steps = 0
            for turn in range(self.config.max_turns):
                steps = turn + 1
                done = yield from self._run_turn(turn)
                if done:
                    success = self._last_test_passed is True
                    break
            yield events.SessionComplete(step=steps, success=success, steps_taken=steps)
        finally:
            self._cleanup()

    # ── setup ──────────────────────────────────────────────────────

    def _start(self) -> Iterator[events.Event]:
        self.debug_session = DebugSession(
            codebase_directory=self.codebase_dir, protected_files=[]
        )
        self._snapshot_workspace_for_revert()
        self._files = self._list_py_files()
        self._test_file = self._guess_test_file(self._files)
        self._messages = [
            {"role": "system", "content": SIMPLE_DEBUG_SYSTEM_PROMPT},
            {"role": "user", "content": self._initial_user_message()},
        ]
        yield events.SessionStart(
            step=0,
            workspace_dir=self.codebase_dir,
            files=self._files,
            test_file=self._test_file,
        )

    def _snapshot_workspace_for_revert(self) -> None:
        """Capture original Python files for revert_file before the session starts.

        Snapshot failure disables revert_file but does not abort the session."""
        try:
            self.debug_session.execute_tool_call(
                HarnessToolCall(
                    name="snapshot_workspace",
                    parameters={},
                    reasoning="internal: capture original files for revert_file",
                )
            )
        except Exception:  # noqa: BLE001 — best effort, mirrors the env
            pass

    # Directories that never hold the code under debug. Walking them turns a
    # normal project into thousands of files and buries the real modules.
    _SKIP_DIRS = frozenset(
        {
            "__pycache__", ".git", ".venv", "venv", "env", ".tox", ".mypy_cache",
            ".pytest_cache", ".ruff_cache", "node_modules", "site-packages",
            "build", "dist", ".eggs",
        }
    )
    # Manifest cap. This list is display + test-file detection only (it does not
    # enter the prompt when the user passed -p), but an unbounded manifest makes
    # the SessionStart line useless on a large repo.
    _MAX_LISTED_FILES = 200

    def _list_py_files(self) -> List[str]:
        """List up to _MAX_LISTED_FILES repo-relative Python paths, excluding build and cache directories."""
        root = Path(self.codebase_dir)
        found: List[str] = []
        try:
            for p in sorted(root.rglob("*.py")):
                if not p.is_file():
                    continue
                rel = p.relative_to(root)
                if self._SKIP_DIRS.intersection(rel.parts[:-1]):
                    continue
                found.append(rel.as_posix())
                if len(found) >= self._MAX_LISTED_FILES:
                    break
        except OSError:
            return []
        return found

    @staticmethod
    def _guess_test_file(files: List[str]) -> Optional[str]:
        """Pick the first test module by basename, falling back to conftest.py."""
        conftest: Optional[str] = None
        for f in files:
            base = f.rsplit("/", 1)[-1]
            if base.startswith("test_") or base.endswith("_test.py"):
                return f
            if base == "conftest.py" and conftest is None:
                conftest = f
        return conftest

    def _initial_user_message(self) -> str:
        """Prepend the workspace path to the user’s prompt, matching the training format."""
        prompt = (self.prompt or "").strip()
        ws = WORKSPACE_LINE.format(path=str(Path(self.codebase_dir).resolve()))
        if prompt:
            return f"{ws}\n\n{prompt}"
        return build_initial_observation(
            test_filename=self._test_file,
            workspace_path=str(Path(self.codebase_dir).resolve()),
        )

    # ── one turn ───────────────────────────────────────────────────

    def _run_turn(self, step: int) -> Iterator[events.Event]:
        """Run one model turn. Generator returns True when the session is done."""
        try:
            message = self._call_model(self._messages)
        except Exception as e:  # noqa: BLE001 — surface as a non-fatal event
            yield events.AgentError(step=step, message=str(e), code="model_request_failed")
            return False

        # Hermes path: vLLM has already parsed the <tool_call> block out of the
        # text, so `content` is reasoning-only and `tool_calls` is structured.
        raw_content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []

        # Fallback for a server without the hermes parser: the model's raw
        # emission (including the <tool_call> block) lands in `content`. Re-parse
        # it so the CLI still works; build_assistant_message strips the block.
        if not tool_calls and raw_content:
            _fb_msg, fb_calls = build_assistant_message(raw_content)
            if fb_calls:
                tool_calls = fb_calls

        # Reasoning shown/stored NEVER contains the <tool_call> block: on the
        # hermes path it is already absent; on the fallback path we strip it.
        # This avoids a double render and string-args errors on later turns.
        content = strip_tool_call_block(raw_content)
        if content:
            yield events.Reasoning(step=step, text=content)

        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        self._messages.append(assistant_msg)

        # Termination is the `done` tool (what the model is trained to emit).
        # The DEBUGGING_COMPLETE sentinel is also accepted, so a model that
        # emits it does not run to max_turns. The prompt does not ask for it.
        if not tool_calls:
            return "DEBUGGING_COMPLETE" in raw_content

        done = False
        for tc in tool_calls:
            name = tc["function"]["name"]
            params = _decode_arguments(tc["function"].get("arguments"))
            observation = self._execute(name, params, step)

            yield events.ToolCall(step=step, name=name, parameters=params)
            yield from self._emit_state(step, name, params, observation)
            yield events.ToolResult(step=step, name=name, result=observation)

            self._messages.append(
                build_tool_message(
                    observation,
                    tool_call_id=tc.get("id") or f"call_{step}",
                    name=name,
                )
            )
            if name == "done":
                done = True

        if "DEBUGGING_COMPLETE" in raw_content:
            done = True
        return done

    def _execute(self, name: str, params: Dict[str, Any], step: int) -> str:
        call = HarnessToolCall(name=name, parameters=params, reasoning="")
        try:
            raw = str(self.debug_session.execute_tool_call(call))
        except Exception as e:  # noqa: BLE001
            raw = f"Error executing {name}: {e}"

        # Hint first, then advance the counters — the rerun hint compares the
        # CURRENT run against the previous one, so it has to read the state as
        # it was before this turn.
        self._hinter.set_progress(
            turn_last_pytest=self._turn_last_pytest,
            turn_last_edit=self._turn_last_edit,
        )
        observation = self._hinter.format_observation(
            name, raw, params, hints_enabled=self.config.hints, wrap=False
        )
        self._record_progress(name, params, raw, step)
        return observation

    def _record_progress(
        self, name: str, params: Dict[str, Any], result: str, step: int
    ) -> None:
        """Advance the turn counters the rerun hint reads.

        A pytest run that collected nothing or died in collection is NOT a test
        observation — the model saw no pass/fail counts — so it must not count
        as "you already ran the tests", or the next correctly-pathed run gets
        scolded for a rerun it never really made.
        """
        turn = step + 1
        if name in FILE_EDIT_TOOLS and file_edit_applied(result):
            self._turn_last_edit = turn
            return
        if name != "bash":
            return
        if "pytest" not in (params.get("command") or "").lower():
            return
        if pytest_collected_nothing(result) or pytest_collection_errored(result):
            return
        self._turn_last_pytest = turn

    # ── derived state / test detection ─────────────────────────────

    def _emit_state(
        self, step: int, name: str, params: Dict[str, Any], observation: str
    ) -> Iterator[events.Event]:
        pdb_tools = {
            "start_debugger", "add_breakpoint", "remove_breakpoint",
            "clear_breakpoints", "continue_execution", "step_over", "step_into",
            "step_out", "print_variable", "examine_locals", "examine_globals",
            "where_am_i", "inspect_stack", "run_expression", "list_code",
        }
        if name in pdb_tools:
            position, breakpoints = self._read_pdb_state()
            locals_map = None
            if name == "print_variable":
                var = params.get("name") or params.get("variable") or "?"
                locals_map = {var: observation.strip()}
            yield events.DebuggerState(
                step=step,
                position=position,
                breakpoints=breakpoints,
                locals=locals_map,
            )

        if name == "bash":
            cmd = params.get("command", "")
            if "pytest" in cmd or re.search(r"python3?\s+\S*test", cmd):
                passed = "passed" in observation and "failed" not in observation
                self._last_test_passed = passed
                m = re.search(r"(\d+ (?:passed|failed)[^\n]*)", observation)
                yield events.TestResult(
                    step=step, passed=passed, summary=(m.group(1) if m else "")
                )

    def _read_pdb_state(self):
        """Read position + breakpoints straight off the live pdb session."""
        sess = getattr(getattr(self.debug_session, "pdb_harness", None),
                       "current_session", None)
        if sess is None:
            return None, None
        position = None
        pos = getattr(sess, "current_position", None)
        if isinstance(pos, dict) and pos.get("file") and pos.get("line") is not None:
            position = {"file": pos["file"], "line": pos["line"]}
        breakpoints = None
        bps = getattr(sess, "breakpoints", None)
        if bps:
            parsed = []
            for bp in bps:
                if isinstance(bp, (tuple, list)) and len(bp) >= 2 and bp[0] is not None:
                    parsed.append({"file": bp[0], "line": bp[1]})
            breakpoints = parsed or None
        return position, breakpoints

    # ── model client (stdlib urllib) ────────────

    def _prompt_budget(self) -> int:
        """Tokens available for messages + tool schemas on one request."""
        return max(
            512,
            self._context_limit
            - self.config.max_tokens
            - self._overhead_tokens,
        )

    def _fit_to_context(
        self, messages: List[Dict[str, Any]], tools: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Drop old history so the request fits the server's window.

        Delegates to `common.context_utils.truncate_messages`, so the CLI keeps
        the pins the model was trained around: the original task description and
        the latest STATE block survive history-dropping (losing them makes the
        model re-explore with bash), and the retained slice never starts on an
        orphaned tool result.
        """
        return truncate_messages(
            messages,
            max_tokens=self._prompt_budget(),
            tokenizer=(
                self._tokenizer
                if self._tokenizer is not None
                else _WireSizeEstimator(self._chars_per_token, tools)
            ),
            tools=tools,
        )

    def _call_model(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """POST the conversation and return the assistant `message` dict.

        Injects the native tool schemas (`tools=`) + `tool_choice:"auto"` so the
        server renders/parses them; returns the raw `message` (``content`` +
        structured ``tool_calls``) for the hermes path. Server must run with
        ``--chat-template common/chat_template_json.jinja --enable-auto-tool-choice
        --tool-call-parser hermes`` (see module docstring).

        History is trimmed to the context window before sending. If the server
        rejects it anyway, its 400 reports the true input-token count — divide
        the bytes we sent by that and we have the model's real chars/token, so
        the retry is sized from a measurement instead of an estimate, and every
        later turn in the session reuses the calibrated ratio.
        """
        tools = get_openai_tools()
        for attempt in range(_CONTEXT_RETRIES):
            sent = self._fit_to_context(messages, tools)
            try:
                return self._post_completion(sent, tools)
            except _ContextOverflow as e:
                # The server serves a SMALLER window than we were configured
                # for: whatever is deployed is older or newer than this
                # constant. Believe the server and re-fit — nothing else in
                # this handler can close a gap of thousands of tokens, because
                # both other paths only nudge the reserve by tens.
                if self._adopt_server_context_limit(e.max_context):
                    continue
                if attempt == _CONTEXT_RETRIES - 1 or not e.input_tokens:
                    raise
                if self._tokenizer is not None:
                    # Exact counting gets the messages right, so an overflow
                    # means the reserve missed something the server adds after
                    # we count — the generation prompt, the structural tag, a
                    # template we didn't render. Measure that blind spot and
                    # reserve it; don't corrupt a count that isn't wrong.
                    counted = len(
                        self._tokenizer.encode(
                            self._tokenizer.apply_chat_template(sent, tools=tools)
                        )
                    )
                    self._overhead_tokens += max(1, e.input_tokens - counted) + 32
                    continue
                measured = _WireSizeEstimator.wire_chars(sent, tools) / e.input_tokens
                # Never revise the ratio upward: a higher chars/token means a
                # bigger budget, and we already know this payload was too big.
                self._chars_per_token = min(self._chars_per_token, measured) * 0.95
        raise RuntimeError("unreachable")  # pragma: no cover

    def _adopt_server_context_limit(self, max_context: Optional[int]) -> bool:
        """Shrink to the window the server says it serves. True when it changed.

        A client constant is a guess about what someone last deployed; the 400
        body is the deployment answering for itself. Without this, pointing the
        CLI at a server with a smaller window kills every remaining turn of the
        session: `_fit_to_context` trims to a budget the server will not accept,
        so the retries re-send a payload of the same size and the third one
        surfaces as a fatal AgentError. Observed against the Modal endpoint,
        which still serves 7168 while this client asked for 32768.

        Only ever shrinks. A server reporting a LARGER window does not mean our
        history is safe to grow into it — max_tokens and the template scaffold
        still have to fit — and taking its word upward would undo a limit the
        caller set deliberately with --context-limit.
        """
        if not max_context or max_context >= self._context_limit:
            return False
        print(
            f"note: server serves {max_context} tokens, not "
            f"{self._context_limit}; refitting history to it.",
            file=sys.stderr, flush=True,
        )
        self._context_limit = max_context
        return True

    def _post_completion(
        self, messages: List[Dict[str, Any]], tools: List[Dict]
    ) -> Dict[str, Any]:
        url = urljoin(self.config.model_url.rstrip("/") + "/", "v1/chat/completions")
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "tools": tools,
            "tool_choice": "auto",
            "stop": _STOP,
            "include_stop_str_in_output": True,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        raw = self._urlopen_with_cold_start_retry(req, url)
        data = json.loads(raw.decode("utf-8"))
        try:
            message = data["choices"][0]["message"]
            return message if isinstance(message, dict) else {}
        except (KeyError, IndexError, TypeError):
            return {}

    def _urlopen_with_cold_start_retry(self, req, url: str) -> bytes:
        """Send the request, absorbing 503s while a serverless container boots.

        A scale-to-zero endpoint answers 503 rather than queueing until vLLM is
        up, so a first call after idle would otherwise fail the whole session on
        a condition that resolves itself in under a minute. Only 503 is retried;
        every other HTTP status is a real error and surfaces immediately.
        """
        deadline = time.monotonic() + _COLD_START_DEADLINE_SECONDS
        warned = False
        while True:
            try:
                with urllib.request.urlopen(
                    req, timeout=self.config.request_timeout
                ) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                if e.code != 503 or time.monotonic() >= deadline:
                    body = e.read().decode("utf-8", errors="replace")
                    if e.code == 400 and "maximum context length" in body:
                        m = re.search(r"contains at least (\d+) input tokens", body)
                        served = re.search(
                            r"maximum context length is (\d+) tokens", body
                        )
                        raise _ContextOverflow(
                            f"model HTTP 400 from {url}: {body[:400]}",
                            int(m.group(1)) if m else None,
                            int(served.group(1)) if served else None,
                        ) from e
                    hint = ""
                    if e.code in (401, 403):
                        hint = (
                            "\nhint: this endpoint requires a key — set VLLM_API_KEY "
                            "or pass --api-key."
                        )
                    raise RuntimeError(
                        f"model HTTP {e.code} from {url}: {body[:400]}{hint}"
                    ) from e
                if not warned:
                    print(
                        "waiting for the inference endpoint to start "
                        "(cold start, up to a few minutes)…",
                        file=sys.stderr, flush=True,
                    )
                    warned = True
                time.sleep(_COLD_START_RETRY_SECONDS)
            except urllib.error.URLError as e:
                raise RuntimeError(f"model request to {url} failed: {e.reason}") from e

    # ── teardown (never deletes the user's dir) ────────────────────

    def _cleanup(self) -> None:
        if self.debug_session is not None:
            try:
                import asyncio
                asyncio.run(self.debug_session.cleanup_async())
            except Exception:  # noqa: BLE001
                pass
            self.debug_session = None


def _decode_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            val = json.loads(arguments)
            return val if isinstance(val, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}
