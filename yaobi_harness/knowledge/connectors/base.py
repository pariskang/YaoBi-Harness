"""Shared HTTP plumbing for knowledge connectors.

Standard library only, with an on-disk response cache so ingestion is
repeatable offline and so re-running a build does not hammer a public API.
Every connector checks its source's licence before it does anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from ..licensing import LicensePolicy
from ..sources import KnowledgeSource, get_source

DEFAULT_TIMEOUT = float(os.environ.get("YAOBI_KNOWLEDGE_TIMEOUT", "30"))
DEFAULT_RETRIES = int(os.environ.get("YAOBI_KNOWLEDGE_RETRIES", "3"))
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class ConnectorError(RuntimeError):
    """Raised when a source cannot be reached or returns something unusable."""


def _ssl_context() -> ssl.SSLContext:
    ca = os.environ.get("YAOBI_KNOWLEDGE_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    return ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()


class HttpConnector:
    """Base class: licence gate, GET with retry, optional disk cache."""

    source_id = ""

    def __init__(
        self,
        policy: LicensePolicy | None = None,
        *,
        endpoint: str | None = None,
        cache_dir: str | Path | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        api_key: str | None = None,
    ) -> None:
        self.policy = policy or LicensePolicy()
        self.source: KnowledgeSource = get_source(self.source_id)
        self.policy.require(self.source)
        self.endpoint = (endpoint or self.source.default_endpoint).rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.retries = max(1, retries)
        self.api_key = api_key

    # ------------------------------------------------------------------ cache
    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / self.source_id / f"{digest}.json"

    def _read_cache(self, url: str) -> Any | None:
        path = self._cache_path(url)
        if path and path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                return None
        return None

    def _write_cache(self, url: str, payload: Any) -> None:
        path = self._cache_path(url)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # -------------------------------------------------------------- transport
    def headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "yaobi-harness/0.2 (clinical decision support)"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_json(self, path: str, params: dict[str, Any] | None = None, *, use_cache: bool = True) -> Any:
        url = self.build_url(path, params)
        if use_cache:
            cached = self._read_cache(url)
            if cached is not None:
                return cached

        last_error: Exception | None = None
        for attempt in range(self.retries):
            request = urllib.request.Request(url, headers=self.headers(), method="GET")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout, context=_ssl_context()) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if use_cache:
                    self._write_cache(url, payload)
                return payload
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                last_error = ConnectorError(f"{self.source_id} HTTP {exc.code} for {url}: {detail}")
                if exc.code == 404:
                    return None
                if exc.code not in RETRYABLE_STATUS:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, ssl.SSLError, ValueError) as exc:
                last_error = ConnectorError(f"{self.source_id} transport error for {url}: {exc!r}")
            if attempt < self.retries - 1:
                time.sleep(2**attempt)
        raise last_error or ConnectorError(f"{self.source_id}: request failed")

    def build_url(self, path: str, params: dict[str, Any] | None = None) -> str:
        base = self.endpoint if not path else f"{self.endpoint}/{path.lstrip('/')}"
        if params:
            base = f"{base}?{urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})}"
        return base
