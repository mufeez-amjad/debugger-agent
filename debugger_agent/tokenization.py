"""Count prompt tokens using tokenizers and the server’s Jinja chat template.

If the tokenizer or template cannot be loaded, the agent falls back to its
character-count estimator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CHAT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "common" / "chat_template_json.jinja"
)

# Both repos use the same vocabulary. Search the cache in this order;
# only an explicit --tokenizer repo ID triggers a download.
_TOKENIZER_REPOS = (
    "moofeez/qwen3-coder-30b-a3b-debugger",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
)


class ChatTemplateTokenizer:
    """Render and encode prompts through the interface used by truncate_messages."""

    def __init__(self, tokenizer: Any, template: Any, source: str):
        self._tokenizer = tokenizer
        self._template = template
        self.source = source

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        add_generation_prompt: bool = False,
        **_kwargs: Any,
    ) -> str:
        # `tokenize=False` arrives in _kwargs and is implicit: we only ever
        # render. The caller tokenizes separately via encode().
        return self._template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
        )

    def encode(
        self, text: str, add_special_tokens: bool = False, **_kwargs: Any
    ) -> List[int]:
        return self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        ).ids


def _build_template(template_source: str) -> Any:
    """Match the server’s Jinja environment, including its non-HTML-escaping tojson.

    Jinja’s default JSON filter escapes characters such as <, >, &, and quotes,
    changing the rendered prompt and token count. AssistantTracker is omitted
    because token masks are not requested."""
    import jinja2
    import jinja2.ext  # `import jinja2` alone does not bind the submodule
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(
            x,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[jinja2.ext.loopcontrols],
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template_source)


def _resolve_tokenizer_file(spec: Optional[str]) -> Optional[str]:
    """Locate a tokenizer.json, or None.

    With no `spec`, searches the local HF cache only. With one, accepts a file
    path, a directory holding tokenizer.json, or a repo id — a repo id is an
    explicit request, so that path is allowed to hit the network.
    """
    if spec:
        candidate = Path(spec).expanduser()
        if candidate.is_file():
            return str(candidate)
        if candidate.is_dir():
            nested = candidate / "tokenizer.json"
            if nested.is_file():
                return str(nested)
        if candidate.exists():
            logger.debug("tokenizer spec %s holds no tokenizer.json", spec)
            return None
        try:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(spec, "tokenizer.json")
        except Exception as e:
            logger.debug("tokenizer download for %s failed: %s", spec, e)
            return None

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None

    for repo in _TOKENIZER_REPOS:
        try:
            hit = try_to_load_from_cache(repo, "tokenizer.json")
        except Exception:  # malformed cache dir, permissions, ...
            continue
        # Returns a str on hit, or a _CACHED_NO_EXIST sentinel / None on miss.
        if isinstance(hit, str) and Path(hit).is_file():
            return hit
    return None


def load_tokenizer(spec: Optional[str] = None) -> Optional[ChatTemplateTokenizer]:
    """Build an exact tokenizer, or None if one isn't available here.

    None is an ordinary outcome — no `tokenizers` installed (a checkout driven
    by a bare system python3 rather than the repo venv), nothing in the HF
    cache, an unreadable template. The caller falls back to estimating.
    """
    tokenizer_file = _resolve_tokenizer_file(spec)
    if not tokenizer_file:
        logger.debug("no tokenizer.json found (spec=%r); estimating instead", spec)
        return None

    try:
        from tokenizers import Tokenizer

        template_source = _CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
        return ChatTemplateTokenizer(
            tokenizer=Tokenizer.from_file(tokenizer_file),
            template=_build_template(template_source),
            source=tokenizer_file,
        )
    except Exception as e:
        logger.debug("exact tokenizer unavailable (%s); estimating instead", e)
        return None
