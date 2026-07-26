"""Local operator console for the Yaobi harness.

A standard-library HTTP server plus a single self-contained page. No web
framework, no CDN, no build step — the console works in an offline hospital
network and inside a Colab kernel alike.

Scope and safety:

* Binds to ``127.0.0.1`` by default. This is a local operator tool with no
  authentication; do not expose it on a shared network without putting your own
  authenticated proxy in front.
* The role selector shows *what each role would receive* — the delivered answer
  is still produced by :func:`yaobi_harness.render.render`, so the console can
  never show a patient more than the patient path allows. The reasoning audit
  is returned in a separate ``audit`` object that is explicitly operator-only.
* Every run goes through the ordinary :class:`YaobiGraphRunner`; the console has
  no privileged path into the tools.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..graph import YaobiGraphRunner
from ..knowledge import ortho_interactions
from ..llm.base import LLMError, NullLLMClient
from ..llm.factory import build_client, describe_client
from ..render import console_payload
from ..state import Budget, ClinicalRunState
from ..tools import DeidentificationKeyError, ToolRegistry

STATIC_DIR = Path(__file__).parent / "static"
MAX_BODY_BYTES = 256 * 1024

logger = logging.getLogger("yaobi.ui")


class ConsoleService:
    """Holds the configured runner pieces and answers the console's API calls."""

    def __init__(
        self,
        *,
        knowledge_store_path: str | Path | None = None,
        xlsx_path: str | Path | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        checkpoint_dir: str | Path | None = None,
        skill_manifest: str | Path | None = None,
        access_token: str | None = None,
    ) -> None:
        self.knowledge_store_path = str(knowledge_store_path) if knowledge_store_path else None
        self.xlsx_path = str(xlsx_path) if xlsx_path else None
        self.checkpoint_dir = str(checkpoint_dir) if checkpoint_dir else None
        self.skill_manifest = str(skill_manifest) if skill_manifest else None
        #: When set, every request must present it. Required for public tunnels.
        self.access_token = access_token or None
        self.llm_error: str | None = None
        self._lock = threading.Lock()

        try:
            self.llm = build_client(llm_provider, **({"model": llm_model} if llm_model else {}))
        except LLMError as exc:
            self.llm = NullLLMClient()
            self.llm_error = str(exc)

        self.knowledge = self._open_knowledge()
        self.tools = ToolRegistry(self.xlsx_path or None, knowledge=self.knowledge)

    def _open_knowledge(self):
        if not self.knowledge_store_path:
            return None
        from ..knowledge.ingest import open_store

        return open_store(self.knowledge_store_path)

    # ------------------------------------------------------------------ routes
    def bootstrap(self) -> dict[str, Any]:
        """Everything the page needs to render its chrome before any run."""
        from ..knowledge.ingest import list_sources
        from ..knowledge.licensing import LicenseError, LicensePolicy

        try:
            policy = LicensePolicy.from_env()
            sources = list_sources(policy)
            policy_info = policy.to_dict()
            policy_error = None
        except LicenseError as exc:  # pragma: no cover - misconfiguration path
            sources, policy_info, policy_error = [], {}, str(exc)

        skills = []
        try:
            from ..skills.loader import SkillRegistry

            manifest = self.skill_manifest or (Path(__file__).parent.parent / "skills" / "manifest.yaml")
            skills = SkillRegistry.from_file(manifest).catalog()
        except Exception as exc:  # noqa: BLE001 - chrome must render even on a bad manifest
            logger.warning("skill catalog unavailable: %s", exc)

        return {
            "llm": {**describe_client(self.llm), "error": self.llm_error},
            "skills": skills,
            "expert": self.expert_summary(),
            "auth_required": bool(self.access_token),
            "knowledge": {
                "configured": self.knowledge is not None,
                "path": self.knowledge_store_path,
                "stats": self.knowledge.stats() if self.knowledge is not None else None,
            },
            "license": {"policy": policy_info, "sources": sources, "error": policy_error},
            "rules": ortho_interactions.rule_pack_summary(),
            "case_store_records": len(self.tools.case_store.records),
            "conditions": sorted(ortho_interactions.KNOWN_CONDITIONS),
            "examples": EXAMPLE_CASES,
        }

    def run_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        complaint = str(payload.get("complaint") or "").strip()
        if not complaint:
            raise ValueError("请填写主诉")

        role = str(payload.get("role") or "physician")
        if role not in ("patient", "physician", "researcher"):
            raise ValueError(f"未知角色: {role}")

        state = ClinicalRunState(complaint=complaint, role=role)
        facts = payload.get("facts")
        if isinstance(facts, dict):
            state.facts.update(facts)
        state.budget = Budget(
            max_loops=int(payload.get("max_loops", 3)),
            max_tool_calls=int(payload.get("max_tool_calls", 24)),
            max_llm_calls=int(payload.get("max_llm_calls", 40)),
        )

        use_llm = bool(payload.get("use_llm", True))
        runner = YaobiGraphRunner(
            self.tools,
            checkpoint_dir=self.checkpoint_dir,
            skill_manifest=self.skill_manifest,
            llm=self.llm if use_llm else NullLLMClient(),
        )
        # One run at a time: the shared ToolRegistry and circuit breaker are not
        # designed for concurrent mutation from several browser tabs.
        with self._lock:
            out = runner.run(state, allow_prescription=bool(payload.get("allow_prescription")))
        return console_payload(out, role)

    def check_interactions(self, payload: dict[str, Any]) -> dict[str, Any]:
        medications = [str(m) for m in (payload.get("medications") or []) if str(m).strip()]
        conditions = [str(c) for c in (payload.get("conditions") or []) if str(c).strip()]
        if not medications:
            raise ValueError("请至少填写一种药物")
        return self.tools.drug_interaction_check(medications, conditions).data

    def expert_summary(self) -> dict[str, Any]:
        """What the loaded expert corpus actually contains."""
        try:
            profile = self.tools.expert_profile()
        except Exception as exc:  # noqa: BLE001
            return {"total_cases": 0, "error": str(exc)}
        return {
            "total_cases": profile.total_cases,
            "patterns": [
                {"pattern": p.pattern, "n_cases": p.n_cases, "reliable": p.reliable,
                 "core_herbs": [h.herb for h in p.core_herbs[:8]]}
                for p in sorted(profile.patterns.values(), key=lambda x: -x.n_cases)[:8]
            ],
            "followup": profile.followup,
            "limitations": profile.limitations,
        }

    def rules(self) -> dict[str, Any]:
        return {
            "summary": ortho_interactions.rule_pack_summary(),
            "rules": [
                {
                    **rule.to_dict(),
                    "requires_all": ["/".join(group) for group in rule.requires_all],
                    "any_conditions": list(rule.any_conditions),
                }
                for rule in ortho_interactions.ORTHO_RULES
            ],
            "classes": {name: list(members) for name, members in ortho_interactions.DRUG_CLASSES.items()},
        }


EXAMPLE_CASES = [
    {
        "label": "急症 · 马尾综合征",
        "role": "patient",
        "complaint": "去年做过腰椎手术，今天突然不能排尿、会阴麻木，双腿越来越无力",
        "facts": {},
    },
    {
        "label": "急症 · 非腰痛红旗",
        "role": "patient",
        "complaint": "既往体健，现突发胸痛、大汗、呼吸困难",
        "facts": {},
    },
    {
        "label": "用药安全 · NSAIDs + 抗凝",
        "role": "physician",
        "complaint": "腰痛3月，久坐加重，右下肢麻木，无大小便异常",
        "facts": {"medications": ["布洛芬 0.3g bid", "华法林 3mg qd"], "conditions": ["elderly"]},
    },
    {
        "label": "常规 · 医师开方路径",
        "role": "physician",
        "complaint": "腰痛3月，刺痛固定，久坐加重",
        "facts": {
            "special_population": {"pregnancy": False, "age": 63, "renal": "normal", "liver": "normal"},
            "medications_confirmed": True,
            "allergies_confirmed": True,
            "medications": [],
        },
    },
    {
        "label": "否定语境 · 应判为常规",
        "role": "patient",
        "complaint": "腰痛，无发热、无外伤、无大小便失禁、无会阴麻木",
        "facts": {},
    },
]


class ConsoleHandler(BaseHTTPRequestHandler):
    server_version = "YaobiConsole/0.2"
    service: ConsoleService

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------------- auth
    def _authorized(self) -> bool:
        """Check the access token when one is configured.

        Accepted from a bearer header, an ``X-Yaobi-Token`` header, a ``?t=``
        query parameter (so a shared link works in one click) or the cookie the
        page sets from that parameter.
        """
        expected = getattr(self.service, "access_token", None)
        if not expected:
            return True
        import urllib.parse

        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer ") and secrets.compare_digest(header[7:].strip(), expected):
            return True
        if secrets.compare_digest((self.headers.get("X-Yaobi-Token") or "").strip(), expected):
            return True
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if any(secrets.compare_digest(v, expected) for v in query.get("t", [])):
            return True
        cookie = self.headers.get("Cookie") or ""
        for part in cookie.split(";"):
            name, _, value = part.strip().partition("=")
            if name == "yaobi_token" and secrets.compare_digest(value, expected):
                return True
        return False

    # ------------------------------------------------------------------ helpers
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise ValueError("请求体过大")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except ValueError as exc:
            raise ValueError(f"请求体不是合法 JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    # -------------------------------------------------------------------- verbs
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if not self._authorized():
            return self._error(401, "缺少或错误的访问令牌；请使用启动时打印的带 ?t= 的链接")
        if path in ("/", "/index.html"):
            page = (STATIC_DIR / "index.html").read_bytes()
            return self._send(200, page, "text/html; charset=utf-8")
        if path == "/api/bootstrap":
            return self._safely(self.service.bootstrap)
        if path == "/api/rules":
            return self._safely(self.service.rules)
        if path == "/api/health":
            return self._json({"ok": True})
        return self._error(404, f"未知路径 {path}")

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if not self._authorized():
            return self._error(401, "缺少或错误的访问令牌")
        try:
            payload = self._read_json()
        except ValueError as exc:
            return self._error(400, str(exc))
        if path == "/api/run":
            return self._safely(lambda: self.service.run_case(payload))
        if path == "/api/interactions":
            return self._safely(lambda: self.service.check_interactions(payload))
        return self._error(404, f"未知路径 {path}")

    def _safely(self, action) -> None:
        try:
            self._json(action())
        except ValueError as exc:
            self._error(400, str(exc))
        except DeidentificationKeyError as exc:
            self._error(500, str(exc))
        except Exception as exc:  # noqa: BLE001 - the console must not die on one bad run
            logger.exception("console request failed")
            self._error(500, f"{type(exc).__name__}: {exc}")


def create_server(service: ConsoleService, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    handler = type("BoundConsoleHandler", (ConsoleHandler,), {"service": service})
    return ThreadingHTTPServer((host, port), handler)


def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    knowledge_store: str | Path | None = None,
    xlsx: str | Path | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    checkpoint_dir: str | Path | None = None,
    skill_manifest: str | Path | None = None,
    open_browser: bool = False,
    public: bool = False,
    access_token: str | None = None,
    ngrok_authtoken: str | None = None,
    ngrok_region: str | None = None,
) -> None:
    """Run the console until interrupted, optionally behind a public tunnel."""
    from .tunnel import TunnelError, banner, new_token, open_ngrok

    token = access_token or (new_token() if public else None)
    service = ConsoleService(
        knowledge_store_path=knowledge_store,
        xlsx_path=xlsx,
        llm_provider=llm_provider,
        llm_model=llm_model,
        checkpoint_dir=checkpoint_dir,
        skill_manifest=skill_manifest,
        access_token=token,
    )
    httpd = create_server(service, host, port)
    url = f"http://{host}:{port}/"
    print(f"Yaobi 控制台已启动: {url}")
    print(f"  LLM      : {describe_client(service.llm)}")
    print(f"  知识库   : {service.knowledge_store_path or '未配置（指南/药典证据为占位数据）'}")
    print("  按 Ctrl+C 停止")
    tunnel = None
    if public:  # pragma: no cover - network path
        try:
            tunnel = open_ngrok(port, token=token, authtoken=ngrok_authtoken, region=ngrok_region)
            print(banner(tunnel, local_url=url))
        except TunnelError as exc:
            print(f"公网隧道未开启: {exc}")
            print("控制台仍在本地可用。")
    elif token:
        print(f"  访问令牌 : {token}\n  带令牌链接: {url}?t={token}")

    if open_browser:  # pragma: no cover - convenience path
        import webbrowser

        webbrowser.open(f"{url}?t={token}" if token else url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        print("\n已停止")
    finally:
        if tunnel is not None:  # pragma: no cover
            tunnel.close()
        httpd.server_close()
