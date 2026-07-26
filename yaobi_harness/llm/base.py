"""Provider-neutral LLM interface used by the planning and screening agents.

The harness treats the model as an *advisory* component inside a hard control
plane: an LLM may propose a plan, surface an extra red flag or raise an extra
safety objection, but it can never widen a capability, clear a rule-based risk
signal or emit a dose. Every provider therefore only has to implement
:meth:`LLMClient.chat`.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    """Raised for transport/protocol failures. Always caught by callers."""


@dataclass
class ToolSpec:
    """OpenAI-style function schema, shared by every supported provider."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Provider-assigned id; echoed back on the matching ``role: "tool"`` message.
    id: str = ""


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    provider: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def json(self, default: Any = None) -> Any:
        """Best-effort JSON extraction from the response text."""
        return extract_json(self.text, default)


class LLMClient(Protocol):
    """Minimal chat interface every provider adapter implements."""

    name: str
    model: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format_json: bool = False,
    ) -> LLMResponse:
        ...


class NullLLMClient:
    """Default client: no model configured, so the harness stays deterministic.

    Returning an empty response (rather than raising) is what makes the whole
    system degrade to the rule-based path when no provider is configured.
    """

    name = "null"
    model = "none"
    available = False

    def chat(self, messages, *, tools=None, temperature=0.0, max_tokens=1024, response_format_json=False) -> LLMResponse:
        return LLMResponse(text="", provider="null", model="none")


#: Trailing comma before a closing brace or bracket — the single most common
#: piece of model JSON sloppiness, and strictly a syntax slip: no content is
#: ambiguous, so repairing it changes nothing about what the model said.
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def extract_json(text: str, default: Any = None) -> Any:
    """Parse JSON from a model response, tolerating the ways models really answer.

    Liberal in the *shape* it accepts, because every consumer validates the
    *content* afterwards: schemas check required fields and types, the plan
    validator checks agents and tools, the broker checks capability. Rejecting a
    payload for a trailing comma buys none of that safety and silently drops the
    whole model-driven path to its deterministic fallback.

    Handled: code fences (with or without a language tag), prose before or after
    the object, trailing commas, and Python-dict-literal quoting. Not handled, on
    purpose: anything requiring a guess about meaning.
    """
    if not text:
        return default
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else candidate
        if candidate.lstrip().startswith("json"):
            candidate = candidate.lstrip()[4:]

    for attempt in _json_candidates(candidate):
        parsed = _loads_tolerant(attempt)
        if parsed is not None:
            return parsed
    return default


def _json_candidates(candidate: str) -> list[str]:
    """The whole string, then the widest brace- and bracket-delimited spans."""
    spans = [candidate]
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if 0 <= start < end:
            spans.append(candidate[start : end + 1])
    return spans


def _loads_tolerant(text: str) -> Any:
    """Strict JSON, then trailing-comma repair, then a Python literal.

    ``ast.literal_eval`` is the right last resort for single-quoted output: it
    parses dict/list literals and evaluates nothing, so it cannot execute a model
    response. ``json.loads`` is always tried first, so well-formed JSON never
    takes this path.
    """
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    repaired = _TRAILING_COMMA_RE.sub(r"\1", text)
    if repaired != text:
        try:
            return json.loads(repaired)
        except (ValueError, TypeError):
            pass
    try:
        value = ast.literal_eval(repaired)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, (dict, list)) else None
