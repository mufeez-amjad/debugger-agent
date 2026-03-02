"""Trim conversation history to a prompt-token budget."""
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from common.state_summary import (
    is_placeholder_state_block,
    latest_state_block_in_messages,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# Prompt budgets leave room for completion tokens.
LOCAL_CONTEXT_TOKENS = 30000
REMOTE_CONTEXT_TOKENS = 60000

_STATE_PRESERVED_PREFIX = "[Previous context summary]\n"


def _extract_latest_state_summary(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Find the latest non-placeholder STATE block outside the system prompt."""
    for message in reversed(messages):
        # Exclude the blank STATE template in the system prompt.
        if message.get("role") == "system":
            continue
        block = latest_state_block_in_messages([message], reverse=True)
        if block and not is_placeholder_state_block(block):
            return block.strip()
    return None


def truncate_messages(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    tokenizer: Optional["PreTrainedTokenizer"] = None,
    tools: Optional[List[Dict]] = None,
    stability_block: int = 8,
) -> List[Dict[str, Any]]:
    """Keep the system prompt, original task, latest STATE summary, and trailing
    user/tool messages, then include as much recent history as fits.

    The minimum retained set may exceed max_tokens. Token counts use the
    supplied tokenizer and tool schemas, falling back to six characters per
    token if counting fails or no tokenizer is available.

    stability_block rounds the history start up to a multiple of that many
    messages to improve prefix-cache reuse. Set it to 0 or 1 to retain the
    tightest-fitting history."""
    system_msgs = [m for m in messages if m.get('role') == 'system']
    non_system = [m for m in messages if m.get('role') != 'system']

    def estimate_tokens(msg_list: List[Dict[str, Any]]) -> int:
        if tokenizer is not None:
            try:
                prompt = tokenizer.apply_chat_template(
                    msg_list,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False
                )
                return len(tokenizer.encode(prompt, add_special_tokens=False))
            except Exception as e:
                logger.warning(f"Failed to use tokenizer for counting, falling back to estimation: {e}")

        total_chars = sum(len(str(m.get('content', ''))) for m in msg_list)
        return total_chars // 6

    full_tokens = estimate_tokens(messages)
    if full_tokens <= max_tokens:
        return list(messages)

    # Keep trailing observations together, including tool results.
    tail_start = len(non_system)
    while tail_start > 0 and non_system[tail_start - 1].get('role') in ('user', 'tool'):
        tail_start -= 1
    if tail_start < len(non_system):
        history = non_system[:tail_start]
        must_keep_tail = list(non_system[tail_start:])
    else:
        history = non_system
        must_keep_tail = []

    # Preserve the original task when older turns are dropped.
    must_keep_head: List[Dict[str, Any]] = []
    head_indices_to_exclude: set = set()
    for i, msg in enumerate(history):
        if msg.get('role') == 'user':
            must_keep_head = [msg]
            head_indices_to_exclude.add(i)
            break
    if must_keep_head:
        history = [m for i, m in enumerate(history) if i not in head_indices_to_exclude]

    # Recover the latest summary from the full history unless the tail has it.
    state_pin: List[Dict[str, Any]] = []
    latest_state = _extract_latest_state_summary(messages)
    if latest_state:
        tail_has_state = any(
            isinstance(m.get('content'), str) and latest_state in m['content']
            for m in must_keep_tail
        )
        if not tail_has_state:
            state_pin = [{
                'role': 'user',
                'content': _STATE_PRESERVED_PREFIX + latest_state,
            }]
    pinned_prefix = must_keep_head + state_pin

    # Verify the minimum (system + task description + STATE + last_user) fits; if
    # not, we still have to return it — callers expect a non-empty result.
    min_result = system_msgs + pinned_prefix + must_keep_tail
    if estimate_tokens(min_result) > max_tokens:
        logger.warning(
            "truncate_messages: even system+last_user exceeds %d tokens; "
            "returning minimum set anyway", max_tokens
        )
        if len(min_result) < len(messages):
            logger.info(
                f"Truncated conversation from {len(messages)} to {len(min_result)} messages "
                f"(limit={max_tokens})"
            )
        return min_result

    # Find the earliest history entry that fits with the pinned messages.
    lo, hi = 0, len(history)  # include history[start:]; start=hi means include nothing
    best_start = len(history)  # default: include nothing from history
    while lo < hi:
        mid = (lo + hi) // 2
        candidate = system_msgs + pinned_prefix + history[mid:] + must_keep_tail
        if estimate_tokens(candidate) <= max_tokens:
            best_start = mid
            hi = mid
        else:
            lo = mid + 1

    # Drop in blocks to keep the retained prefix stable across nearby turns.
    if stability_block and stability_block > 1 and best_start > 0:
        quantized = ((best_start + stability_block - 1) // stability_block) * stability_block
        best_start = min(quantized, len(history))

    # Do not start history with a tool result whose call was dropped.
    while best_start < len(history) and history[best_start].get('role') == 'tool':
        best_start += 1

    result = system_msgs + pinned_prefix + history[best_start:] + must_keep_tail

    if len(result) < len(messages):
        final_tokens = estimate_tokens(result)
        logger.info(f"Truncated conversation from {len(messages)} to {len(result)} messages "
                   f"({final_tokens} tokens, limit={max_tokens})")

    return result
