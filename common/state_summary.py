"""Helpers for <STATE vN>...</STATE> summaries.

Matching accepts versionless tags, extra spacing, and any letter case."""
import re
from typing import Dict, List, Optional, Tuple

# Whole block (tolerant opening tag). Matches <STATE>, <STATE v3>, <STATE v2 >, …
STATE_BLOCK_RE = re.compile(r"<STATE[^>]*>.*?</STATE>", re.DOTALL | re.IGNORECASE)
# Same, but capturing the inner content.
STATE_BLOCK_CONTENT_RE = re.compile(
    r"<STATE[^>]*>(.*?)</STATE>", re.DOTALL | re.IGNORECASE
)
# Whole block plus any leading whitespace, as a single capture group. Lets a
# caller lift a trailing STATE block out of an assistant turn.
STATE_BLOCK_WITH_LEADING_WS_RE = re.compile(
    r"(\s*<STATE[^>]*>.*?</STATE>)", re.DOTALL | re.IGNORECASE
)
# Version digit inside the opening tag, when present (vN).
_STATE_VERSION_RE = re.compile(r"<STATE[^>]*?\bv(\d+)\b", re.IGNORECASE)
# Presence of an opening STATE tag (even without a closing tag / version).
_STATE_OPEN_RE = re.compile(r"<STATE\b", re.IGNORECASE)


def has_state_block(text: Optional[str]) -> bool:
    """True if an opening ``<STATE`` tag is present (closing tag not required)."""
    return bool(text) and bool(_STATE_OPEN_RE.search(text))


def find_state_blocks(text: Optional[str]) -> List[str]:
    """All complete ``<STATE…>…</STATE>`` blocks, in order."""
    if not text:
        return []
    return STATE_BLOCK_RE.findall(text)


def count_state_blocks(text: Optional[str]) -> int:
    return len(find_state_blocks(text))


def extract_state_version(text: Optional[str]) -> Optional[int]:
    """Version number from the (first) STATE opening tag, or None if versionless."""
    if not text:
        return None
    m = _STATE_VERSION_RE.search(text)
    return int(m.group(1)) if m else None


def latest_state_block(text: Optional[str]) -> Optional[str]:
    """The last complete STATE block in ``text`` (most recently emitted)."""
    blocks = find_state_blocks(text)
    return blocks[-1] if blocks else None


# Literal placeholder fragments from the system prompt's STATE template
# (common.prompts._SYSTEM_PROMPT_TAIL). A block still containing any of them
# is the template copied verbatim, not a summary of anything. The model can
# emit the skeleton with "facts: [key discoveries]" unchanged.
_TEMPLATE_PLACEHOLDERS = (
    "[key discoveries]",
    "[approach choices]",
    "[file:path sha1=x lines=[y-z]]",
    "[remaining tasks]",
)


def is_placeholder_state_block(text: Optional[str]) -> bool:
    """True when ``text`` contains a STATE block that is just the unfilled
    system-prompt template (any literal ``[placeholder]`` fragment survives).
    Checks the LAST complete block; an unclosed opening tag returns False
    (there is no content to judge)."""
    block = latest_state_block(text)
    if block is None:
        return False
    lowered = block.lower()
    return any(ph in lowered for ph in _TEMPLATE_PLACEHOLDERS)


def latest_state_block_in_messages(
    messages: List[Dict], reverse: bool = False
) -> Optional[str]:
    """Most recent STATE block across a message list.

    By default scans forward and keeps the last block seen. ``reverse=True``
    scans newest-first and stops at the first hit.
    """
    seq = reversed(messages) if reverse else messages
    latest: Optional[str] = None
    for msg in seq:
        content = msg.get("content")
        if not isinstance(content, str) or "<STATE" not in content.upper():
            # cheap guard; full match below is case-insensitive
            if not (isinstance(content, str) and _STATE_OPEN_RE.search(content)):
                continue
        blocks = find_state_blocks(content)
        if blocks:
            latest = blocks[-1]
            if reverse:
                return latest
    return latest


def latest_state_content_and_version(
    messages: List[Dict], reverse: bool = True, default_version: int = 0
) -> Tuple[Optional[str], int]:
    """(inner content, version) of the most recent STATE block.

    Versionless blocks get ``1`` (a STATE without a tag is the first summary).
    Returns ``(None, default_version)`` when no block exists.
    """
    block = latest_state_block_in_messages(messages, reverse=reverse)
    if block is None:
        return None, default_version
    cm = STATE_BLOCK_CONTENT_RE.search(block)
    content = cm.group(1).strip() if cm else ""
    version = extract_state_version(block)
    return content, (version if version is not None else 1)
