"""Provider adapters for Azure OpenAI, Poe, MiniMax and LiteLLM.

All four expose an OpenAI-compatible ``chat/completions`` payload, so the
shared :class:`OpenAICompatibleClient` carries the request/response logic and
each subclass only supplies its endpoint, auth header and any payload quirks.

Only the standard library is used for transport; the harness stays
dependency-free and works behind an HTTPS proxy via the usual environment
variables. Endpoints and API versions are configurable so a provider changing
its URL never requires a code change.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from typing import Any

from .base import LLMError, LLMResponse, ToolCall, ToolSpec

DEFAULT_TIMEOUT = float(os.environ.get("YAOBI_LLM_TIMEOUT", "60"))
DEFAULT_RETRIES = int(os.environ.get("YAOBI_LLM_RETRIES", "3"))
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _ssl_context() -> ssl.SSLContext:
    """Default TLS context; honours ``SSL_CERT_FILE``/``YAOBI_LLM_CA_BUNDLE``."""
    ca = os.environ.get("YAOBI_LLM_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    return ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()


class OpenAICompatibleClient:
    """Shared implementation of the OpenAI chat-completions contract."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if not api_key:
            raise LLMError(f"{self.name}: missing API key")
        if not model:
            raise LLMError(f"{self.name}: missing model/deployment name")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = max(1, retries)
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self.available = True

    # ------------------------------------------------------------- overridable
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **self.extra_headers,
        }

    def build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        temperature: float,
        max_tokens: int,
        response_format_json: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]
            payload["tool_choice"] = "auto"
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.extra_body)
        return payload

    # -------------------------------------------------------------- transport
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format_json: bool = False,
    ) -> LLMResponse:
        payload = self.build_payload(messages, tools, temperature, max_tokens, response_format_json)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None

        for attempt in range(self.retries):
            request = urllib.request.Request(self.endpoint(), data=body, headers=self.headers(), method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout, context=_ssl_context()) as response:
                    return self.parse_response(json.loads(response.read().decode("utf-8")))
            except urllib.error.HTTPError as exc:  # pragma: no cover - network path
                detail = exc.read().decode("utf-8", "replace")[:400]
                last_error = LLMError(f"{self.name} HTTP {exc.code}: {detail}")
                if exc.code not in RETRYABLE_STATUS:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, ssl.SSLError, ValueError) as exc:  # pragma: no cover
                last_error = LLMError(f"{self.name} transport error: {exc!r}")
            if attempt < self.retries - 1:  # pragma: no cover - timing path
                time.sleep(2**attempt)
        raise last_error or LLMError(f"{self.name}: request failed")

    def parse_response(self, data: dict[str, Any]) -> LLMResponse:
        choices = data.get("choices") or []
        message = (choices[0].get("message") if choices else {}) or {}
        text = message.get("content") or ""
        if isinstance(text, list):  # some gateways return content parts
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {"_raw": arguments}
            calls.append(ToolCall(str(function.get("name", "")), arguments or {}, str(call.get("id", ""))))
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            tool_calls=calls,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            model=str(data.get("model") or self.model),
            provider=self.name,
            raw=data,
        )


class AzureOpenAIClient(OpenAICompatibleClient):
    """Azure OpenAI Service.

    ``model`` is the *deployment* name. Auth uses the ``api-key`` header rather
    than a bearer token, and the API version is a query parameter.
    """

    name = "azure"

    def __init__(self, *, api_key: str, deployment: str, endpoint: str, api_version: str = "2024-10-21", **kwargs: Any) -> None:
        if not endpoint:
            raise LLMError("azure: missing endpoint (AZURE_OPENAI_ENDPOINT)")
        self.api_version = api_version
        super().__init__(api_key=api_key, model=deployment, base_url=endpoint, **kwargs)

    def endpoint(self) -> str:
        return f"{self.base_url}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}"

    def headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "api-key": self.api_key, **self.extra_headers}


class PoeClient(OpenAICompatibleClient):
    """Poe's OpenAI-compatible endpoint. ``model`` is the bot name."""

    name = "poe"

    def __init__(self, *, api_key: str, model: str = "Claude-Sonnet-4.5", base_url: str = "https://api.poe.com/v1", **kwargs: Any) -> None:
        super().__init__(api_key=api_key, model=model, base_url=base_url, **kwargs)


#: MiniMax's two regional hosts. Picking the wrong one fails to resolve or
#: rejects the key, so both are named rather than left to the caller to guess.
MINIMAX_HOSTS = {
    "china": "https://api.minimaxi.com/v1",
    "global": "https://api.minimax.io/v1",
}
MINIMAX_DEFAULT_BASE_URL = MINIMAX_HOSTS["china"]

#: Legacy hosts and paths, kept only to give a clear error rather than a timeout.
_MINIMAX_LEGACY_HOSTS = ("api.minimax.chat",)


class MiniMaxClient(OpenAICompatibleClient):
    """MiniMax, via its OpenAI-compatible ``/chat/completions`` endpoint.

    Two regions, and they are not interchangeable:

    * China   — ``https://api.minimaxi.com/v1``
    * Global  — ``https://api.minimax.io/v1``

    Set ``MINIMAX_BASE_URL`` explicitly, or ``MINIMAX_REGION=china|global``.
    The old ``api.minimax.chat`` host with its ``/text/chatcompletion_v2`` path is
    no longer the documented surface; the standard OpenAI path is used now, so the
    only MiniMax-specific behaviour left is the ``base_resp`` error envelope, which
    the API still returns on some failures and which would otherwise be read as a
    successful empty completion.
    """

    name = "minimax"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "MiniMax-M3",
        base_url: str | None = None,
        group_id: str | None = None,
        region: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.group_id = group_id
        resolved = base_url or MINIMAX_HOSTS.get((region or "").strip().lower()) or MINIMAX_DEFAULT_BASE_URL
        if any(host in resolved for host in _MINIMAX_LEGACY_HOSTS):
            raise LLMError(
                f"minimax: {resolved} 是已废弃的地址。请改用 "
                f"{MINIMAX_HOSTS['china']}（中国）或 {MINIMAX_HOSTS['global']}（海外），"
                "通过 MINIMAX_BASE_URL 或 MINIMAX_REGION=china|global 设置。"
            )
        super().__init__(api_key=api_key, model=model, base_url=resolved, **kwargs)

    def endpoint(self) -> str:
        # The documented surface is OpenAI-compatible, so the shared endpoint is
        # correct. GroupId is still accepted as a query parameter by accounts that
        # require it.
        url = super().endpoint()
        return f"{url}?GroupId={self.group_id}" if self.group_id else url

    def parse_response(self, data: dict[str, Any]) -> LLMResponse:
        status = (data.get("base_resp") or {}).get("status_code")
        if status not in (None, 0):
            raise LLMError(f"minimax error {status}: {(data.get('base_resp') or {}).get('status_msg')}")
        response = super().parse_response(data)
        usage = data.get("usage") or {}
        if not response.total_tokens and usage.get("total_tokens"):
            response.completion_tokens = int(usage["total_tokens"])
        return response


class LiteLLMClient(OpenAICompatibleClient):
    """A LiteLLM proxy/gateway, which fronts any upstream provider."""

    name = "litellm"

    def __init__(self, *, api_key: str, model: str, base_url: str = "http://localhost:4000/v1", **kwargs: Any) -> None:
        base = base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        super().__init__(api_key=api_key, model=model, base_url=base, **kwargs)


PROVIDERS = {
    "azure": AzureOpenAIClient,
    "poe": PoeClient,
    "minimax": MiniMaxClient,
    "litellm": LiteLLMClient,
}
