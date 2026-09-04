"""Build an LLM client from environment variables or an explicit spec.

Configuration is intentionally boring — one variable selects the provider, the
rest are provider-specific:

``YAOBI_LLM_PROVIDER``  ``azure`` | ``poe`` | ``minimax`` | ``litellm`` | ``none``

* azure    — ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_ENDPOINT``,
             ``AZURE_OPENAI_DEPLOYMENT``, optional ``AZURE_OPENAI_API_VERSION``
* poe      — ``POE_API_KEY``, optional ``POE_MODEL``, ``POE_BASE_URL``
* minimax  — ``MINIMAX_API_KEY``, optional ``MINIMAX_MODEL`` (default
             ``MiniMax-M3``), ``MINIMAX_REGION`` (``china`` | ``global``),
             ``MINIMAX_BASE_URL``, ``MINIMAX_GROUP_ID``. The two regional hosts
             are not interchangeable: ``https://api.minimaxi.com/v1`` for China,
             ``https://api.minimax.io/v1`` for everywhere else.
* litellm  — ``LITELLM_API_KEY``, ``LITELLM_MODEL``, optional ``LITELLM_BASE_URL``

If nothing is configured, a :class:`NullLLMClient` is returned and the harness
runs its deterministic rule-based path.
"""

from __future__ import annotations

import os
from typing import Any

from .base import LLMClient, LLMError, NullLLMClient
from .providers import AzureOpenAIClient, LiteLLMClient, MiniMaxClient, PoeClient


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def build_client(provider: str | None = None, **overrides: Any) -> LLMClient:
    """Return a configured client, or :class:`NullLLMClient` when unavailable.

    Raises :class:`LLMError` only when a provider was explicitly requested but
    its credentials are incomplete — silent downgrade in that case would hide a
    misconfiguration behind seemingly-normal deterministic output.
    """
    name = (provider or os.environ.get("YAOBI_LLM_PROVIDER") or "none").strip().lower()
    if name in ("", "none", "null", "off", "disabled"):
        return NullLLMClient()

    common = {
        k: overrides[k] for k in ("timeout", "retries", "extra_headers", "extra_body") if k in overrides
    }

    if name == "azure":
        return AzureOpenAIClient(
            api_key=overrides.get("api_key") or _env("AZURE_OPENAI_API_KEY", "AZURE_API_KEY"),
            deployment=overrides.get("model") or _env("AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_MODEL"),
            endpoint=overrides.get("base_url") or _env("AZURE_OPENAI_ENDPOINT", "AZURE_API_BASE"),
            api_version=overrides.get("api_version") or _env("AZURE_OPENAI_API_VERSION", default="2024-10-21"),
            **common,
        )
    if name == "poe":
        return PoeClient(
            api_key=overrides.get("api_key") or _env("POE_API_KEY"),
            model=overrides.get("model") or _env("POE_MODEL", default="Claude-Sonnet-4.5"),
            base_url=overrides.get("base_url") or _env("POE_BASE_URL", default="https://api.poe.com/v1"),
            **common,
        )
    if name == "minimax":
        return MiniMaxClient(
            api_key=overrides.get("api_key") or _env("MINIMAX_API_KEY"),
            model=overrides.get("model") or _env("MINIMAX_MODEL", default="MiniMax-M3"),
            # No default here: the client resolves the region, so an unset
            # MINIMAX_BASE_URL falls through to MINIMAX_REGION rather than being
            # pinned to one host by the factory.
            base_url=overrides.get("base_url") or _env("MINIMAX_BASE_URL") or None,
            region=overrides.get("region") or _env("MINIMAX_REGION") or None,
            group_id=overrides.get("group_id") or _env("MINIMAX_GROUP_ID") or None,
            **common,
        )
    if name == "litellm":
        return LiteLLMClient(
            api_key=overrides.get("api_key") or _env("LITELLM_API_KEY", "LITELLM_MASTER_KEY", default="sk-noauth"),
            model=overrides.get("model") or _env("LITELLM_MODEL", default=""),
            base_url=overrides.get("base_url") or _env("LITELLM_BASE_URL", default="http://localhost:4000/v1"),
            **common,
        )
    raise LLMError(f"unknown YAOBI_LLM_PROVIDER: {name!r}; expected azure|poe|minimax|litellm|none")


#: Set to reveal which provider and model a run used in the console, the
#: notebook and the startup banner. Off by default.
#:
#: Those three surfaces are the ones other people see — a screen share, a
#: demo, a Colab notebook committed to a repository with its cell outputs still
#: in it. Which vendor is behind a clinical assistant is a procurement and
#: contractual matter, not something a passing screenshot should settle, and it
#: is of no use whatsoever to the clinician reading the answer.
#:
#: What stays unredacted is everything an operator uses to *work*: the CLI, the
#: logs, the journal file on disk. The journal in particular must keep the real
#: model name — it is part of every request's content address, so an offline
#: replay diverges on the name alone without it.
SHOW_MODEL_ENV = "YAOBI_SHOW_MODEL"

#: Keys that name a model or a vendor. Redaction works by key so a payload
#: gains protection by being routed through :func:`redact_model_identity`,
#: rather than by every new caller remembering which fields to strip.
MODEL_IDENTITY_KEYS = frozenset({
    "model", "provider", "llm_model", "llm_provider", "deployment", "base_url",
})


def show_model_identity() -> bool:
    return os.environ.get(SHOW_MODEL_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def describe_client(client: LLMClient) -> dict[str, Any]:
    """Full identity. For the CLI, the logs and the journal — never for a screen."""
    return {
        "provider": getattr(client, "name", "unknown"),
        "model": getattr(client, "model", "unknown"),
        "available": bool(getattr(client, "available", False)),
    }


def redact_model_identity(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Drop vendor and model names from a payload bound for a screen.

    A no-op when :data:`SHOW_MODEL_ENV` is set, so a developer can still see what
    they are talking to without editing code.
    """
    data = dict(payload or {})
    if show_model_identity():
        return data
    return {k: v for k, v in data.items() if k not in MODEL_IDENTITY_KEYS}


def public_client_info(client: LLMClient) -> dict[str, Any]:
    """What a browser or a notebook is told about the configured model.

    Deliberately still answers the question that actually matters at a glance —
    *is a model driving this run, or is it the deterministic path?* — because the
    two produce very different consultations and confusing them is a real error.
    It just does not name the vendor.
    """
    available = bool(getattr(client, "available", False))
    return {**redact_model_identity(describe_client(client)),
            "available": available, "configured": available}
