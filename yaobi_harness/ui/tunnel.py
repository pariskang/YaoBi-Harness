"""Public tunnelling for the console (ngrok), with authentication forced on.

Exposing a clinical console to the public internet is a meaningfully different
act from binding it to localhost, so this module makes the difference explicit:

* a tunnel **cannot** be opened without an access token — one is generated if
  the operator does not supply it, and the console rejects every request that
  does not carry it;
* the token is printed once, appended to the shareable URL as ``?t=…`` so the
  link works in one click, and never written to disk;
* the URL is a demo/review link, not a deployment. Nothing here provides
  per-user identity, audit of who opened a case, or transport-level controls a
  hospital would require.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Any


class TunnelError(RuntimeError):
    """Raised when a public tunnel cannot be established."""


@dataclass
class Tunnel:
    public_url: str
    provider: str
    token: str
    handle: Any = None

    def shareable_url(self) -> str:
        return f"{self.public_url}/?t={self.token}" if self.token else self.public_url

    def close(self) -> None:  # pragma: no cover - network teardown
        if self.handle is None:
            return
        try:
            from pyngrok import ngrok

            ngrok.disconnect(self.public_url)
        except Exception:  # noqa: BLE001 - teardown must never raise
            pass


def new_token() -> str:
    """A URL-safe token; 32 bytes is well past guessing range."""
    return secrets.token_urlsafe(32)


def open_ngrok(port: int, *, token: str, authtoken: str | None = None, region: str | None = None) -> Tunnel:
    """Open an ngrok HTTP tunnel to ``port``.

    Requires ``pyngrok`` (``pip install pyngrok``) and an ngrok authtoken —
    anonymous tunnels are no longer generally available, so a missing token is
    reported as a configuration error rather than a mysterious failure.
    """
    if not token:
        raise TunnelError("拒绝在无访问令牌的情况下开启公网隧道")
    try:
        from pyngrok import conf, ngrok
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise TunnelError(
            "缺少 pyngrok。安装：pip install pyngrok；并设置 NGROK_AUTHTOKEN（在 "
            "https://dashboard.ngrok.com/get-started/your-authtoken 获取）"
        ) from exc

    auth = authtoken or os.environ.get("NGROK_AUTHTOKEN") or os.environ.get("NGROK_AUTH_TOKEN")
    if not auth:
        raise TunnelError(
            "缺少 ngrok authtoken。设置环境变量 NGROK_AUTHTOKEN 或传入 --ngrok-authtoken；"
            "在 https://dashboard.ngrok.com/get-started/your-authtoken 获取。"
        )
    ngrok.set_auth_token(auth)
    if region:
        conf.get_default().region = region
    try:
        handle = ngrok.connect(port, "http")
    except Exception as exc:  # noqa: BLE001 - surface the provider's message
        raise TunnelError(f"ngrok 隧道建立失败: {exc}") from exc
    return Tunnel(public_url=handle.public_url.rstrip("/"), provider="ngrok", token=token, handle=handle)


def banner(tunnel: Tunnel, *, local_url: str) -> str:
    """The warning an operator should read before sharing the link."""
    return "\n".join([
        "",
        "═" * 68,
        "  ⚠  公网隧道已开启 —— 这是演示/评审链接，不是临床部署",
        "═" * 68,
        f"  本地地址 : {local_url}",
        f"  公开地址 : {tunnel.shareable_url()}",
        f"  访问令牌 : {tunnel.token}",
        "",
        "  · 令牌只在本次进程内有效，不落盘；关闭进程即失效。",
        "  · 任何拿到链接的人都能运行病例；请勿输入真实患者可识别信息。",
        "  · 无逐用户身份、无访问审计、无院内网络边界——正式部署请自行前置",
        "    认证网关，并按本机构的数据合规要求评估。",
        "═" * 68,
        "",
    ])
