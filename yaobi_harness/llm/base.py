"""Provider-neutral LLM interface. Every provider implements one method.

The model makes the clinical judgements — triage, enquiry, differentials, the
wording of every reply. Two things it cannot do, and neither is enforced here:
the capability broker scopes which tools exist for it, and the dose pipeline
requires a physician's signature. Everything else the rules produce is material
handed to the model, not a verdict applied to it.

What *is* enforced here is parsing: :func:`extract_json` gets a usable object out
of however the model chose to format its answer, because a formatting slip must
not silently drop the whole model-driven path to its deterministic fallback.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import jsonrepair


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


def extract_json(text: str, default: Any = None) -> Any:
    """Parse JSON from a model response, tolerating the ways models really answer.

    Liberal in the *shape* it accepts, because every consumer validates the
    *content* afterwards: schemas check required fields and types, the plan
    validator checks agents and tools, the broker checks capability. Rejecting a
    payload for a trailing comma buys none of that safety and silently drops the
    whole model-driven path to its deterministic fallback.

    The repair itself lives in :mod:`yaobi_harness.llm.jsonrepair`; see there for
    what is fixed and what is deliberately left to fail.
    """
    return extract_json_with_repairs(text, default)[0]


def extract_json_with_repairs(text: str, default: Any = None) -> tuple[Any, list[str]]:
    """Like :func:`extract_json`, but also reports which repairs were needed.

    Callers that keep an audit trail use this: "the model's answer parsed only
    after we closed a truncated string" is materially different from "the model's
    answer was well-formed", and a run that repairs every response is a prompt or
    ``max_tokens`` problem rather than a success.
    """
    if not text:
        return default, []
    value, repairs = jsonrepair.loads_with_repairs(text)
    if value is None:
        # ``ast.literal_eval`` as a final resort: it parses Python literals and
        # evaluates nothing, so it cannot execute a model response. Kept because
        # it handles tuple and set literals, which the repairer does not rewrite.
        value = _python_literal(text)
        if value is None:
            return default, repairs
        repairs = [*repairs, "python_literal"]
    return value, repairs


def _python_literal(text: str) -> Any:
    try:
        value = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, (dict, list)) else None
