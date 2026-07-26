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
#: Live conversations kept in memory before the oldest is evicted.
MAX_SESSIONS = 50
#: Recorded journals kept for replay before the oldest is evicted. Journals hold
#: tool payloads and model text — clinical content — so the console keeps a short
#: window in memory and never writes one to disk.
MAX_RECORDINGS = 20

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
        skill_dirs: list[str] | None = None,
        vision: bool = True,
        panel_concurrency: int | None = None,
    ) -> None:
        self.knowledge_store_path = str(knowledge_store_path) if knowledge_store_path else None
        self.panel_concurrency = panel_concurrency
        self.xlsx_path = str(xlsx_path) if xlsx_path else None
        self.checkpoint_dir = str(checkpoint_dir) if checkpoint_dir else None
        self.skill_manifest = str(skill_manifest) if skill_manifest else None
        self.skill_dirs = list(skill_dirs or [])
        #: When set, every request must present it. Required for public tunnels.
        self.access_token = access_token or None
        self.llm_error: str | None = None
        self._lock = threading.Lock()
        #: Separate from ``_lock``: session bookkeeping must not wait behind a
        #: whole clinical run, and a run must not hold the session lock while it
        #: fans out to a consult panel.
        self._sessions_lock = threading.Lock()

        try:
            self.llm = build_client(llm_provider, **({"model": llm_model} if llm_model else {}))
        except LLMError as exc:
            self.llm = NullLLMClient()
            self.llm_error = str(exc)

        self.knowledge = self._open_knowledge()
        self.vision = self._open_vision(vision)
        self.tools = ToolRegistry(self.xlsx_path or None, knowledge=self.knowledge, vision=self.vision)
        #: Live conversations, keyed by session id. In-memory only: transcripts
        #: are clinical content and must not be persisted by a demo console.
        self.sessions: dict[str, Any] = {}
        #: Recorded runs available for offline re-derivation, keyed by run id.
        #: Also in-memory only, and for the same reason.
        self.recordings: dict[str, dict[str, Any]] = {}

    def _open_knowledge(self):
        if not self.knowledge_store_path:
            return None
        from ..knowledge.ingest import open_store

        return open_store(self.knowledge_store_path)

    @staticmethod
    def _open_vision(enabled: bool):
        """Build the vision client, tolerating an unconfigured environment."""
        if not enabled:
            return None
        from ..vision.client import build_vision_client

        try:
            return build_vision_client()
        except Exception as exc:  # noqa: BLE001 - the console must still start
            logger.warning("vision client unavailable: %s", exc)
            return None

    def _runner(
        self,
        use_llm: bool = True,
        interview_loop: Any | None = None,
        journal: Any | None = None,
        panel_concurrency: int | None = None,
    ) -> YaobiGraphRunner:
        return YaobiGraphRunner(
            self.tools,
            checkpoint_dir=self.checkpoint_dir,
            skill_manifest=self.skill_manifest,
            llm=self.llm if use_llm else NullLLMClient(),
            skill_dirs=self.skill_dirs or None,
            interview_loop=interview_loop,
            journal=journal,
            panel_concurrency=panel_concurrency or self.panel_concurrency,
        )

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
            skills = SkillRegistry.discover(manifest, extra_roots=self.skill_dirs or None).catalog()
        except Exception as exc:  # noqa: BLE001 - chrome must render even on a bad manifest
            logger.warning("skill catalog unavailable: %s", exc)

        from ..interview.axes import AXES, TIERS
        from ..vision.client import IMAGE_KINDS, describe_vision

        return {
            "llm": {**describe_client(self.llm), "error": self.llm_error},
            "vision": {**describe_vision(self.vision), "kinds": list(IMAGE_KINDS)},
            "skills": skills,
            "interview": {
                "tiers": list(TIERS),
                "axes": [
                    {"axis_id": a.axis_id, "label": a.label, "tier": a.tier,
                     "tradition": a.tradition, "rationale": a.rationale,
                     "probes": list(a.probes)}
                    for a in AXES
                ],
            },
            "personas": _persona_catalog(),
            "panel": {
                "concurrency_default": self.panel_concurrency or _env_concurrency(),
                "concurrency_max": 8,
            },
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
        role = self._role_of(payload, default="physician")
        state = self._state_from(payload, role)
        journal = None
        if payload.get("record_journal"):
            from ..journal import Journal

            # No path: the console records for replay in this process only.
            journal = Journal(mode="record")

        runner = self._runner(
            bool(payload.get("use_llm", True)),
            journal=journal,
            panel_concurrency=_coerce_concurrency(payload.get("panel_concurrency")),
        )
        # One run at a time: the shared ToolRegistry and circuit breaker are not
        # designed for concurrent mutation from several browser tabs.
        with self._lock:
            out = runner.run(state, allow_prescription=bool(payload.get("allow_prescription")))
        result = console_payload(out, role)
        result["meta"]["panel_concurrency"] = runner.panel_concurrency
        if journal is not None:
            self._keep_recording(out, payload, journal)
            result["journal"] = {
                "run_id": out.run_id,
                **journal.summary(),
                "replay_hint": journal.replay_hint(),
            }
        return result

    def replay_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Re-derive a recorded run from its journal and report whether it matched.

        The point of the exercise is the *comparison*, not the second answer: a
        replay that quietly produces a different release status is worse than no
        replay at all, so the fidelity block is returned first and a divergence is
        reported as a failure rather than folded into the new payload.

        ``complaint``/``facts`` may be supplied to replay the journal against a
        *changed* case. That is the interesting audit question — "would this
        recording still justify the decision if the history had read differently?"
        — and the honest answer is a loud failure, because the journal's entries
        are addressed by request content and no longer match.
        """
        from ..journal import Journal, JournalError

        run_id = str(payload.get("run_id") or "")
        with self._sessions_lock:
            recording = self.recordings.get(run_id)
        if recording is None:
            raise ValueError("没有这次运行的录制日志；请勾选「录制可复核日志」后重新运行")

        source = recording["journal"]
        # A fresh Journal over a copy of the entries: replaying advances a cursor,
        # so reusing the recording's own object would consume it and make the
        # second replay of the same run fail for the wrong reason.
        replay = Journal(mode="replay", entries=list(source.entries), meta=dict(source.meta))
        role = recording["role"]
        case = dict(recording["payload"])
        against = "recording"
        if str(payload.get("complaint") or "").strip():
            case["complaint"] = str(payload["complaint"]).strip()
            against = "modified"
        if isinstance(payload.get("facts"), dict):
            case["facts"] = payload["facts"]
            against = "modified"

        state = self._state_from(case, role)
        runner = self._runner(
            bool(case.get("use_llm", True)),
            journal=replay,
            panel_concurrency=replay.meta.get("panel_concurrency"),
        )
        try:
            with self._lock:
                out = runner.run(state, allow_prescription=bool(case.get("allow_prescription")))
        except JournalError as exc:
            return {
                "fidelity": {"reproduced": False, "against": against, "error": str(exc),
                             "divergences": replay.divergences},
                "journal": replay.summary(),
            }

        after = _fingerprint(out)
        before = recording["fingerprint"]
        differences = [k for k in before if before[k] != after.get(k)]
        return {
            "fidelity": {
                # A journal that ran dry mid-run did not re-derive the decision
                # either: the tail was executed live, so this is not a replay of
                # the recording but a hybrid, and it must not report success.
                "reproduced": (not differences and not replay.diverged
                               and not replay.live_after_exhaustion and against == "recording"),
                "against": against,
                "differences": differences,
                "before": before,
                "after": after,
                "divergences": replay.divergences,
                "live_after_exhaustion": replay.live_after_exhaustion,
            },
            "journal": {**replay.summary(), "replay_hint": replay.replay_hint()},
            **console_payload(out, role),
        }

    # --------------------------------------------------------------- internals
    @staticmethod
    def _role_of(payload: dict[str, Any], *, default: str) -> str:
        role = str(payload.get("role") or default)
        if role not in ("patient", "physician", "researcher"):
            raise ValueError(f"未知角色: {role}")
        return role

    def _state_from(self, payload: dict[str, Any], role: str) -> ClinicalRunState:
        """Build the run state. Shared by a live run and by its replay, so a
        replay cannot accidentally be given different inputs than the recording."""
        complaint = str(payload.get("complaint") or "").strip()
        if not complaint:
            raise ValueError("请填写主诉")
        state = ClinicalRunState(complaint=complaint, role=role)
        facts = payload.get("facts")
        if isinstance(facts, dict):
            state.facts.update(facts)
        state.budget = Budget(
            max_loops=int(payload.get("max_loops", 3)),
            max_tool_calls=int(payload.get("max_tool_calls", 24)),
            max_llm_calls=int(payload.get("max_llm_calls", 40)),
        )
        state.enable_panel = bool(payload.get("enable_panel"))
        state.images = _coerce_images(payload.get("images"))
        return state

    def _keep_recording(self, state: ClinicalRunState, payload: dict[str, Any], journal: Any) -> None:
        with self._sessions_lock:
            self.recordings[state.run_id] = {
                "payload": dict(payload),
                "role": state.role,
                "journal": journal,
                "fingerprint": _fingerprint(state),
            }
            while len(self.recordings) > MAX_RECORDINGS:
                self.recordings.pop(next(iter(self.recordings)))

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One conversation turn. Creates the session on the first message."""
        from ..conversation import ConversationSession

        message = str(payload.get("message") or "").strip()
        if not message:
            raise ValueError("消息不能为空")
        role = self._role_of(payload, default="patient")

        session_id = str(payload.get("session_id") or "")
        # Session creation and eviction under one lock: the console is a threading
        # server, and ``pop(next(iter(...)))`` is a read-modify-write that raises
        # RuntimeError if another request inserts during the iteration.
        with self._sessions_lock:
            session = self.sessions.get(session_id)
            if session is None:
                session = ConversationSession(
                    role=role,
                    runner=self._runner(
                        bool(payload.get("use_llm", True)),
                        panel_concurrency=_coerce_concurrency(payload.get("panel_concurrency")),
                    ),
                    allow_prescription=bool(payload.get("allow_prescription")),
                )
                self.sessions[session.session_id] = session
                while len(self.sessions) > MAX_SESSIONS:  # bound memory on a long-lived console
                    self.sessions.pop(next(iter(self.sessions)))

        for image in _coerce_images(payload.get("images")):
            session.attach_image(image["ref"], kind=image["kind"], deidentified=image["deidentified"])

        with self._lock:
            reply = session.send(message)
        return {
            "session_id": session.session_id,
            "interview": session.interview.summary(session.facts, session.complaint, role=session.role),
            "reply": reply.to_dict(),
            "audit": console_payload(session.state, session.role)["audit"] if session.state else {},
            "meta": console_payload(session.state, session.role)["meta"] if session.state else {},
            "turn_count": len(session.turns),
        }

    def reset_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._sessions_lock:
            self.sessions.pop(str(payload.get("session_id") or ""), None)
        return {"ok": True}

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


def _env_concurrency() -> int:
    from ..agent.panel import _default_concurrency

    return _default_concurrency()


def _coerce_concurrency(raw: Any) -> int | None:
    """Panel threads requested by the browser, or ``None`` to keep the default.

    Clamped rather than rejected: a slider that sends 99 means "as parallel as
    you can", not "fail the run". The upper bound matches the panel's own.
    """
    if raw in (None, "", 0):
        return None
    try:
        return max(1, min(8, int(raw)))
    except (TypeError, ValueError):
        raise ValueError(f"并发数必须是整数: {raw!r}") from None


def _fingerprint(state: ClinicalRunState) -> dict[str, Any]:
    """The parts of a run a replay has to reproduce exactly.

    Deliberately not the whole payload: timestamps and run ids differ by
    construction, and comparing them would report every replay as a divergence.
    What must match is the decision — release status, risk mode, which planner
    produced the graph — and the evidence it rests on, in ledger order.
    """
    return {
        "release_status": state.release_status,
        "risk_mode": state.risk_mode,
        "planner_mode": state.planner_mode,
        "tasks": [f"{t.agent}:{t.status}" for t in state.tasks],
        "evidence": [f"{eid}|{e.source}|{e.level}" for eid, e in state.evidence.items()],
        "safety_issues": list(state.safety_issues),
    }


def _persona_catalog() -> list[dict[str, Any]]:
    """The consult panel's members, for the console to show before a run."""
    from ..agent.panel import DEFAULT_PANEL, PERSONAS

    return [
        {
            "persona": name,
            "label": profile.get("label", name),
            "consult_mode": profile.get("consult_mode", "screening"),
            "default": name in DEFAULT_PANEL,
            "brief": str(profile.get("instructions", ""))[:220],
            "inputs": list(profile.get("inputs") or ()),
            "outputs": list(profile.get("outputs") or ()),
        }
        for name, profile in PERSONAS.items()
    ]


def _coerce_images(raw: Any) -> list[dict[str, Any]]:
    """Validate inbound image attachments from the browser.

    The de-identification attestation must be explicit: a payload that omits it is
    rejected rather than defaulted, because defaulting it to ``true`` would let a
    forgotten checkbox send an identifiable image to a third-party model.
    """
    from ..vision.client import IMAGE_KINDS, MAX_IMAGE_BYTES

    images: list[dict[str, Any]] = []
    for entry in (raw or [])[:6]:
        if not isinstance(entry, dict):
            continue
        ref = str(entry.get("ref") or "").strip()
        if not ref:
            continue
        kind = str(entry.get("kind") or "other")
        if kind not in IMAGE_KINDS:
            raise ValueError(f"未知图片类型: {kind}")
        if not entry.get("deidentified"):
            raise ValueError("上传图片前必须勾选「已去标识化」：请先遮盖姓名、各类编号、日期、条码与人脸")
        # A data URI inflates ~4/3 over the raw bytes; check before decoding so an
        # oversized upload fails fast instead of after a full base64 pass.
        if ref.startswith("data:") and len(ref) > MAX_IMAGE_BYTES * 4 // 3 + 512:
            raise ValueError(f"图片过大，上限约 {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
        images.append({"kind": kind, "ref": ref, "deidentified": True})
    return images


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
        if path == "/api/replay":
            return self._safely(lambda: self.service.replay_case(payload))
        if path == "/api/interactions":
            return self._safely(lambda: self.service.check_interactions(payload))
        if path == "/api/chat":
            return self._safely(lambda: self.service.chat(payload))
        if path == "/api/chat/reset":
            return self._safely(lambda: self.service.reset_chat(payload))
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


class _ConsoleServer(ThreadingHTTPServer):
    """The console's HTTP server, with a queue deep enough for a real burst.

    ``socketserver`` defaults to a listen backlog of 5. Clinical runs are slow
    relative to HTTP, and the service serialises them, so a handful of browser
    tabs — or one page firing several requests on load — overflows the accept
    queue and the client sees a connection reset rather than a queued request.
    A reset looks like the console crashed, which is a much worse diagnosis than
    "your request waited".
    """

    request_queue_size = 128
    #: Threads must not keep the process alive after Ctrl+C, and a Colab kernel
    #: cell that starts the console should be interruptible.
    daemon_threads = True
    allow_reuse_address = True


def create_server(service: ConsoleService, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    handler = type("BoundConsoleHandler", (ConsoleHandler,), {"service": service})
    return _ConsoleServer((host, port), handler)


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
    skill_dirs: list[str] | None = None,
    vision: bool = True,
    panel_concurrency: int | None = None,
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
        skill_dirs=skill_dirs,
        vision=vision,
        panel_concurrency=panel_concurrency,
    )
    httpd = create_server(service, host, port)
    url = f"http://{host}:{port}/"
    # Print the URL that actually works. Printing the bare address when a token is
    # configured hands the operator a link that 401s on every request, which reads
    # as a broken console rather than a missing parameter.
    entry = f"{url}?t={token}" if token else url
    print(f"Yaobi 控制台已启动: {entry}")
    if token:
        print(f"  访问令牌 : {token}   （链接已包含；也可用 X-Yaobi-Token 头调用 API）")
    print(f"  LLM      : {describe_client(service.llm)}")
    print(f"  知识库   : {service.knowledge_store_path or '未配置（指南/药典证据为占位数据）'}")
    print(f"  视觉模型 : {service.vision.model if service.vision else '未配置（影像/舌象工具不可用）'}")
    print(f"  会诊并发 : {service.panel_concurrency or _env_concurrency()} 线程（页面可逐次调整；1 为顺序执行）")
    print("  按 Ctrl+C 停止")
    tunnel = None
    if public:  # pragma: no cover - network path
        try:
            tunnel = open_ngrok(port, token=token, authtoken=ngrok_authtoken, region=ngrok_region)
            print(banner(tunnel, local_url=entry))
        except TunnelError as exc:
            print(f"公网隧道未开启: {exc}")
            print("控制台仍在本地可用。")

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
