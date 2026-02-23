"""
System prompts for the debugging model.
"""

import random
from string import Template

# The debugger and no-debugger eval arms differ only in their investigation
# guidance. Tool definitions are injected separately through the chat template.
_SYSTEM_PROMPT_HEADER = "You are an expert Python engineer. ONE tool call per response."

# Tool semantics live in the native <tools> JSON schemas. The system prompt
# only contains behavioral guidance that is not represented by those schemas.
# STATE schema: facts must carry the observation that PROVED them. Unproven
# beliefs go under hypothesis (UNVERIFIED). falsified keeps ruled-out theories
# and the disconfirming result, so the model does not retry them after context
# truncation. The matcher is content-agnostic (common/state_summary.py), so
# blocks with other field names still parse.
_SYSTEM_PROMPT_TAIL = """STATE Summary (create when info accumulates; refresh every ~8 turns while investigating):
<STATE v1>
facts: [only VERIFIED discoveries, each with the observation that proved it, e.g. "X is shared (id check step 7 -> True)"]
hypothesis: [current working theory — UNVERIFIED, do not restate as fact]
falsified: [theories ruled out + the disconfirming result, e.g. "off-by-one in partition — edit changed nothing, same 5 failures"]
decisions: [approach choices]
pointers: [file:path sha1=X lines=[Y-Z]]
todos: [remaining tasks]
</STATE>
Increment version each update (v1→v2→v3). If a test run after your edit shows the SAME failures, your hypothesis is falsified: record it under falsified and form a new one.

If you see <TOOL_OUTPUT_TRUNCATED>, request a smaller window or a different section.

Before starting the debugger, read the failing code path first; if the defect is directly visible in the source, fix it without the debugger.

${investigation}

Re-run tests only after modifying code — never repeat a test run without an intervening change. The failing test is the specification: fix the source module under test, not the test. Do NOT modify test files.

The goal is to FIX THE BUG and PASS TESTS, not just understand it. Be systematic and concise."""

_SYSTEM_PROMPT_TEMPLATE = Template(
    _SYSTEM_PROMPT_HEADER + "\n\n" + _SYSTEM_PROMPT_TAIL
)


# --- Arm-specific fragments interpolated into the template above ---

# Debugger guidance. It carries these directives: ls-first / no-cd /
# multi-testfile, static-vs-dynamic classification, the 5-step debugger loop
# (the harness rejects bad breakpoint lines and does not move them, so the
# guidance says to read first), always-inspect-after-stepping, the object-IDENTITY
# check for aliasing bugs, upstream tracing, the 3-errors fallback, and
# verify-before-done. The wording stays short because this prefix is rendered
# on every step.
_INVESTIGATION_DEBUG = """Start with `ls` — don't assume filenames or layout. Work at or below the current directory; do NOT `cd` to a parent directory. Most workspaces are flat, but if `ls` shows no test files, look one level down before concluding there are none. A defect may have several test files (some named `*_test.py`, some `test_*.py`), so run `pytest -x -q` to find the failing test, then read it and the code it exercises. To read a specific function, locate it first — `grep -n "def <name>" <file.py>` — then read exactly that range with `sed -n 'A,Bp' <file.py>`; don't guess line numbers or rescan ever-larger windows.

Classify the bug before acting:
- STATIC (wrong operator, off-by-one, typo, missing branch): visible in the source → read and fix directly.
- DYNAMIC (unexpected variable state, closure captures, iterator/generator behavior, shared mutable state, async ordering, object lifecycle): visible only at runtime → debugger:
1. start_debugger on the failing TEST file.
2. add_breakpoint on an executable line INSIDE the function the test calls — read the code around the target first; a blank/comment, def header, or module/class-body line is REJECTED and no breakpoint is set.
3. continue_execution to reach it, then step_over/step_into along the failing path.
4. ALWAYS examine_locals/print_variable after stepping — never blind-step. Inspect until you can point to the exact value, frame, or call where the invariant first breaks. For aliasing/shared-state bugs check object IDENTITY (`a is b`, `id(x)`): a shallow copy sharing a nested object shows up only in identity, not value.
5. Make the edit the observed state implies: trace the wrong value UPSTREAM to where it first originates and edit THAT function — the symptom site is often not the bug site.

Default to the debugger when the cause isn't obvious from the source. If it errors 3+ times in a row, stop, read the code, and fix directly. After editing, re-run `pytest -xvs <failing_test_file>` and confirm it passes before calling done."""

# Investigation guidance — no-debugger arm (bash is the confirmation tool).
_INVESTIGATION_NO_DEBUG = """Start by running `ls` to see the files in the working directory — don't assume filenames. The code and tests are in your current working directory or a subdirectory of it; run commands there and do NOT `cd` to a parent directory. Most workspaces are flat, so check the current directory first, but if `ls` shows no test files look one level down before concluding there are none. A defect may have several test files (some named `*_test.py`, some `test_*.py`), so run the suite with `pytest -x -q` to see which test actually fails, then read that failing test and the code it exercises to localize the bug. To read a specific function, locate it first — `grep -n "def <name>" <file.py>` — then read exactly that range with `sed -n 'A,Bp' <file.py>`; don't guess line numbers or rescan ever-larger windows. When the cause isn't clear from the source, reproduce with `pytest -xvs <failing_test_file>` and add prints or run targeted snippets to inspect the real runtime behavior."""

# Investigation guidance for the mandatory-debugger variant. Unlike
# _INVESTIGATION_DEBUG (which lets the model fix static bugs from source), this
# REQUIRES the debugger before any edit. It exists to induce debugger use, so a
# caller can measure whether a model uses the debugger well when it must use it.
# Callers select it with an explicit flag. The harness does not block edits that
# come before debugger use, so a model that edits without debugging anyway is
# visible in the result.
_INVESTIGATION_DEBUG_MANDATORY = """Start by running `ls` to see the files in the working directory — don't assume filenames. The code and tests are in your current working directory or a subdirectory of it; run commands there and do NOT `cd` to a parent directory. Most workspaces are flat, so check the current directory first, but if `ls` shows no test files look one level down before concluding there are none. A defect may have several test files (some named `*_test.py`, some `test_*.py`), so run the suite with `pytest -x -q` to see which test actually fails, then read that failing test and the code it exercises to localize the bug.

You MUST use the debugger to diagnose this bug before changing any code — do not patch from static reading alone. Required workflow:
1. start_debugger on the test file that contains the failing test.
2. add_breakpoint on an executable line at the suspect site — read the code around the target first and pick a line INSIDE the function the test calls (a blank/comment, def header, or module/class-body line is rejected and no breakpoint is set).
3. continue_execution to reach it, then step_over/step_into to follow the failing path.
4. examine_locals/print_variable after stepping to inspect the real state — keep inspecting until you can point to the exact value, frame, or call where the invariant first breaks.
5. Only then edit_file, and make the fix follow from what you observed in the debugger.

Do not call edit_file until you have started the debugger and inspected the runtime state at least once. After editing, re-run the failing test with `pytest -xvs <failing_test_file>` and confirm it passes before calling done."""


def build_system_prompt(use_debugging_tools: bool, mandatory: bool = False) -> str:
    """Assemble the debugging system prompt for the given arm.

    Only the investigation guidance differs between arms. Tool definitions are
    always injected natively through the chat template.

    ``mandatory`` selects the debugger-required investigation guidance
    (``_INVESTIGATION_DEBUG_MANDATORY``). It only applies to the debugger arm and
    is for measuring how well a model uses the debugger when it must use it;
    the default is False.

    """
    if use_debugging_tools:
        investigation = (
            _INVESTIGATION_DEBUG_MANDATORY if mandatory else _INVESTIGATION_DEBUG
        )
    else:
        investigation = _INVESTIGATION_NO_DEBUG
    return _SYSTEM_PROMPT_TEMPLATE.safe_substitute(investigation=investigation)


SIMPLE_DEBUG_SYSTEM_PROMPT = build_system_prompt(use_debugging_tools=True)

# Debugger arm with the mandatory-debugger investigation guidance. Callers
# select it with an explicit flag.
DEBUG_MANDATORY_SYSTEM_PROMPT = build_system_prompt(use_debugging_tools=True, mandatory=True)


# Base text of the first user message ("initial observation") shown to the
# model. Kept minimal and production-shaped. The workspace test keeps its
# natural on-disk name, <module>_test.py, so naming the real test file is safe.
# In real deployment the model is dropped into an
# unknown repo and told a test is failing — sometimes by name, sometimes not —
# so the message mix mirrors that: most variants are scenario-independent
# (the model must run the suite and identify the failure itself), and a
# minority name the actual failing test file. File discovery
# ("ls first") is taught by the investigation guidance in
# SIMPLE_DEBUG_SYSTEM_PROMPT (_INVESTIGATION_DEBUG/_NO_DEBUG), which
# explicitly tells the model to `ls` and match `*_test.py` — not restated here.
# Paraphrases are written in the casual, informal tone a developer actually
# uses when tossing this over (contractions, lowercase, filler). One is chosen
# at random so the model doesn't overfit to a single exact wording. The pool
# also has formal, terse-ticket, and handoff registers, so the model does not
# link "user message" to one chatty lowercase style. No variant carries a
# debugger-preference line.
#
# What is deliberately NOT here: pasted tracebacks, error text, and paths with
# directory components. Those read as the obvious diversification, but the
# opener is chosen without knowing the scenario's real failure, so any concrete
# detail would be FABRICATED — a traceback for a different bug, or a path that
# does not exist in a flat workspace. That trains the model to distrust or
# ignore the one part of the message that is supposed to be ground truth.
# They become available only once scenarios carry real symptom text / nested
# layouts; until then every variant here is true of every task.
INITIAL_OBSERVATIONS = [
    # casual register
    "test's failing — can you figure out what's wrong and fix it?",
    "hey, the tests are broken. mind taking a look?",
    "something's busted and the tests won't pass. can you fix it?",
    "ugh, tests are red again. can you track down the bug?",
    "a test just started failing and idk why. can you dig in and fix it?",
    "tests aren't passing. can you debug it and patch whatever's broken?",
    "this test keeps failing — can you figure out why and fix it?",
    "the test suite's broken, can you sort it out?",
    "something broke the tests. find the bug and fix it?",
    "test's red, can you debug it real quick?",
    # formal / report register
    "The test suite is currently failing. Please identify the root cause and apply a fix.",
    "We have a failing test in this workspace. Investigate and resolve the underlying defect.",
    "Bug report: the tests do not pass. Please diagnose and correct the source.",
    "There is a defect in this codebase causing test failures. Locate it and fix it.",
    # terse / ticket register
    "failing tests. please fix.",
    "repro: run the tests. expected pass, actual fail. fix the cause.",
    "Tests failing — needs a fix.",
    # handoff register
    "picking this up from me — the tests don't pass and I ran out of time. can you finish it?",
    "handing this over: suite is failing, cause unknown. take it from here.",
]

# Variants that name the failing test file (``{test}`` is the workspace
# filename, e.g. ``encoder_test.py``). Chosen for a MINORITY of messages (see
# NAMED_OBSERVATION_PROB): the model must still usually learn to run the suite
# and identify the failure itself, but should also handle the equally-real
# "this specific test is failing" hand-off without re-discovering the target.
NAMED_INITIAL_OBSERVATIONS = [
    # casual register
    "{test} is failing — can you figure out what's wrong and fix it?",
    "hey, {test} just went red. mind taking a look?",
    "{test} keeps failing and idk why. can you dig in and fix it?",
    "can you check {test}? something in there won't pass.",
    "{test} is broken. find the bug and fix it?",
    # formal / terse registers, mirroring INITIAL_OBSERVATIONS above
    "The tests in {test} are failing. Please identify the root cause and apply a fix.",
    "Bug report: {test} does not pass. Diagnose and correct the source it exercises.",
    "{test} failing. please fix.",
    "handing this over: {test} is red and I ran out of time. can you take it?",
]

# Fraction of first messages that name the failing test file (only when the
# caller can supply it). Deliberately a minority so suite-discovery behavior
# stays the common case.
NAMED_OBSERVATION_PROB = 0.3

# Factual working-directory line prepended to the first user message when the
# caller knows the workspace path. Without it, the model `cd`s to invented
# absolute roots (/root, /usr/src/workspace) instead of finding its directory.
# Fixed wording, no paraphrase pool: this is factual context like a header.
WORKSPACE_LINE = "Working directory: {path} (commands run here; use relative paths)."


def build_initial_observation(rng=None, test_filename=None, workspace_path=None) -> str:
    """Build the first user message shown to the model.

    Every caller uses this function, so the first message cannot drift
    between callers.

    ``test_filename`` is the failing test's workspace filename (its natural
    on-disk name, e.g. ``encoder_test.py``). When provided, a minority of
    messages (``NAMED_OBSERVATION_PROB``) name it explicitly; the rest — and
    all messages when it is omitted — stay scenario-independent so the model
    keeps learning to run the suite and find the failure itself. No variant
    carries a debugger-preference line.

    ``rng`` is an optional ``random.Random`` (or the ``random`` module) used to
    pick the variation. Pass a seeded instance for reproducible selection; when
    ``None`` the process-global ``random`` is used.
    """
    chooser = rng if rng is not None else random
    if test_filename and chooser.random() < NAMED_OBSERVATION_PROB:
        template = chooser.choice(NAMED_INITIAL_OBSERVATIONS)
        body = template.format(test=test_filename)
    else:
        body = chooser.choice(INITIAL_OBSERVATIONS)
    if workspace_path:
        return f"{WORKSPACE_LINE.format(path=workspace_path)}\n\n{body}"
    return body


SIMPLE_NO_DEBUG_SYSTEM_PROMPT = build_system_prompt(use_debugging_tools=False)
