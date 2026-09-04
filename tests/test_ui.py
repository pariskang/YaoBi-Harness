"""Tests for the operator console.

The console must never become a side door: it runs cases through the ordinary
runner, it scopes the delivered answer by role exactly like the CLI does, and
it keeps the reasoning audit in a separate object that is explicitly
operator-only. These tests pin that separation along with the API contract the
single-page app depends on.
"""

from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("YAOBI_DEID_KEY", "unit-test-fixed-key")
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from yaobi_harness.llm.base import LLMResponse
from yaobi_harness.render import console_payload
from yaobi_harness.ui.server import EXAMPLE_CASES, ConsoleService, create_server

STATIC = Path(__file__).resolve().parents[1] / "yaobi_harness" / "ui" / "static" / "index.html"


def post(url: str, payload: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def get(url: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            body = response.read().decode("utf-8")
            return response.status, (json.loads(body) if body.startswith(("{", "[")) else body)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


class ConsoleApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ConsoleService()
        cls.httpd = create_server(cls.service, "127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    # ------------------------------------------------------------------ chrome
    def test_index_is_self_contained(self):
        status, page = get(self.base + "/")
        self.assertEqual(status, 200)
        self.assertIn("<title>Yaobi", page)
        for external in ("src=\"http", "href=\"http://cdn", "cdn.jsdelivr", "unpkg.com", "googleapis.com/css"):
            self.assertNotIn(external, page, f"page must not load {external}")

    def test_health_and_unknown_route(self):
        self.assertEqual(get(self.base + "/api/health"), (200, {"ok": True}))
        status, _ = get(self.base + "/api/nope")
        self.assertEqual(status, 404)

    def test_bootstrap_describes_the_deployment(self):
        status, data = get(self.base + "/api/bootstrap")
        self.assertEqual(status, 200)
        # The vendor and model are redacted; what the page needs is whether a
        # model is driving the run at all.
        self.assertIn("configured", data["llm"])
        self.assertNotIn("provider", data["llm"])
        self.assertNotIn("model", data["llm"])
        self.assertFalse(data["knowledge"]["configured"])
        self.assertTrue(data["license"]["sources"])
        self.assertEqual(data["rules"]["rule_count"], 18)
        self.assertTrue(data["conditions"])
        self.assertEqual(len(data["examples"]), len(EXAMPLE_CASES))

    def test_rules_endpoint_exposes_the_pack(self):
        status, data = get(self.base + "/api/rules")
        self.assertEqual(status, 200)
        self.assertEqual(len(data["rules"]), 18)
        first = data["rules"][0]
        for key in ("rule_id", "title", "severity", "mechanism", "management", "requires_all"):
            self.assertIn(key, first)

    # -------------------------------------------------------------------- runs
    def test_run_returns_delivered_audit_and_meta(self):
        status, data = post(self.base + "/api/run", {"complaint": "腰痛3月，久坐加重", "role": "physician"})
        self.assertEqual(status, 200)
        self.assertEqual(set(data), {"delivered", "audit", "meta"})
        self.assertTrue(data["audit"]["tasks"])
        self.assertTrue(data["audit"]["evidence"])
        self.assertIn("checks_run", data["audit"]["safety_audit"])
        self.assertIn("release_status", data["meta"])

    def test_urgent_case_is_reported_as_urgent(self):
        status, data = post(self.base + "/api/run", {
            "complaint": "去年做过腰椎手术，今天突然不能排尿、会阴麻木",
            "role": "patient", "allow_prescription": True,
        })
        self.assertEqual(status, 200)
        self.assertEqual(data["meta"]["risk_mode"], "urgent")
        self.assertEqual(data["meta"]["release_status"], "urgent_action_plan")
        self.assertIn("urgent", data["delivered"])
        self.assertNotIn("prescription_draft", data["delivered"])

    def test_patient_delivery_never_carries_the_evidence_ledger(self):
        """The console shows the audit, but not inside the patient's answer."""
        status, data = post(self.base + "/api/run", {"complaint": "腰痛3月，久坐加重", "role": "patient"})
        self.assertEqual(status, 200)
        self.assertNotIn("evidence_ledger", data["delivered"])
        self.assertNotIn("citations", data["delivered"])
        self.assertNotIn("research_patient_id", json.dumps(data["delivered"], ensure_ascii=False))
        # ...while the operator audit still has it.
        self.assertTrue(data["audit"]["evidence"])

    def test_medication_findings_reach_the_console(self):
        facts = {"medications": ["布洛芬 0.3g bid", "华法林 3mg qd"], "conditions": ["elderly"]}
        status, data = post(self.base + "/api/run", {
            "complaint": "腰痛3月，久坐加重", "role": "physician", "facts": facts})
        self.assertEqual(status, 200)
        findings = data["audit"]["medication_safety"]["findings"]
        self.assertTrue(any(f["rule_id"] == "ORTHO-001" for f in findings))
        self.assertEqual(data["meta"]["release_status"], "needs_examination")
        # The physician gets the full finding; the patient gets plain-language advice.
        self.assertTrue(data["delivered"]["medication_safety"]["findings"])

        _, as_patient = post(self.base + "/api/run", {
            "complaint": "腰痛3月，久坐加重", "role": "patient", "facts": facts})
        warnings = as_patient["delivered"]["medication_warnings"]
        self.assertTrue(warnings)
        self.assertIn("what_to_do", warnings[0])
        self.assertNotIn("mechanism", warnings[0])

    def test_every_bundled_example_runs(self):
        for example in EXAMPLE_CASES:
            with self.subTest(example=example["label"]):
                status, data = post(self.base + "/api/run", {
                    "complaint": example["complaint"], "role": example["role"],
                    "facts": example["facts"], "allow_prescription": example["role"] == "physician",
                })
                self.assertEqual(status, 200, data)
                self.assertIn(data["meta"]["release_status"], {
                    "urgent_action_plan", "needs_more_information", "needs_examination",
                    "insufficient_evidence", "treatment_advice_only", "draft_for_physician",
                    "approved_by_physician", "blocked", "failed_closed",
                })

    def test_bad_requests_are_rejected_with_a_message(self):
        for payload, fragment in (({}, "主诉"), ({"complaint": "x", "role": "admin"}, "角色")):
            status, data = post(self.base + "/api/run", payload)
            self.assertEqual(status, 400)
            self.assertIn(fragment, data["error"])

    def test_malformed_json_returns_400_not_500(self):
        request = urllib.request.Request(
            self.base + "/api/run", data=b"{not json", headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request, timeout=30)
        self.assertEqual(ctx.exception.code, 400)

    # ------------------------------------------------------------ interactions
    def test_the_console_opens_the_conversation_itself(self):
        status, body = post(f"{self.base}/api/chat/open", {"role": "patient"})
        self.assertEqual(status, 200, body)
        self.assertTrue(body["session_id"])
        self.assertTrue(body["reply"]["message"])
        self.assertTrue(body["reply"]["questions"], "an opening with no question is not an opening")
        self.assertEqual(body["turn_count"], 1)

    def test_the_opened_session_continues_normally(self):
        _, opened = post(f"{self.base}/api/chat/open", {"role": "patient"})
        status, turn = post(f"{self.base}/api/chat", {
            "message": "腰痛3个月，久坐加重", "role": "patient",
            "session_id": opened["session_id"]})
        self.assertEqual(status, 200, turn)
        self.assertEqual(turn["session_id"], opened["session_id"])
        self.assertEqual(turn["turn_count"], 3, "opening + user + agent")

    def test_replay_route_is_reachable_over_http(self):
        status, recorded = post(f"{self.base}/api/run", {
            "complaint": "腰痛3月，久坐加重", "role": "physician", "record_journal": True})
        self.assertEqual(status, 200, recorded)
        status, replayed = post(f"{self.base}/api/replay", {"run_id": recorded["journal"]["run_id"]})
        self.assertEqual(status, 200, replayed)
        self.assertTrue(replayed["fidelity"]["reproduced"], replayed["fidelity"])

    def test_replaying_an_unknown_run_is_a_400(self):
        status, body = post(f"{self.base}/api/replay", {"run_id": "nope"})
        self.assertEqual(status, 400)
        self.assertIn("录制", body["error"])

    def test_interaction_endpoint(self):
        status, data = post(self.base + "/api/interactions", {"medications": ["秋水仙碱", "克拉霉素"]})
        self.assertEqual(status, 200)
        self.assertFalse(data["pass"])
        self.assertEqual(data["rule_findings"][0]["rule_id"], "ORTHO-017")
        self.assertTrue(data["blocking"])

    def test_interaction_endpoint_requires_medications(self):
        status, data = post(self.base + "/api/interactions", {"medications": []})
        self.assertEqual(status, 400)
        self.assertIn("药物", data["error"])

    def test_condition_gated_rule_needs_the_condition(self):
        _, without = post(self.base + "/api/interactions", {"medications": ["阿仑膦酸钠"]})
        _, with_it = post(self.base + "/api/interactions",
                          {"medications": ["阿仑膦酸钠"], "conditions": ["renal_impairment"]})
        self.assertEqual(without["rule_findings"], [])
        self.assertTrue(any(f["rule_id"] == "ORTHO-010" for f in with_it["rule_findings"]))

    # ------------------------------------------------------------- concurrency
    def test_concurrent_requests_do_not_break_the_shared_store(self):
        """Regression: the SQLite store was opened on one thread and used on others."""
        from yaobi_harness.knowledge.store import KnowledgeStore

        store = KnowledgeStore(":memory:")
        store.register_source("openfda", version="test")
        results, errors = [], []

        def worker():
            try:
                results.append(store.enabled_sources())
            except Exception as exc:  # noqa: BLE001 - the point of the test
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        store.close()
        self.assertEqual(errors, [])
        self.assertTrue(all(r == ["openfda"] for r in results))


class PageTokenPlumbingTests(unittest.TestCase):
    """The page must carry the token, not just be *served* one.

    The server accepted a token four ways — bearer, ``X-Yaobi-Token``, ``?t=`` and
    a cookie — and the page sent none of them. Only the initial HTML load carried
    ``?t=``; every XHR after it got a 401, so a token-protected console (Colab's
    embedded iframe, any ``--public`` tunnel) rendered its chrome and then failed
    on the first action with "缺少或错误的访问令牌".

    Asserted statically because the check has to hold without a browser in the
    test dependencies. It is deliberately about *mechanism*, not wording: what
    broke was the absence of any token plumbing at all, and that is what this
    detects if someone rewrites ``api()`` again.
    """

    @classmethod
    def setUpClass(cls):
        cls.page = STATIC.read_text(encoding="utf-8")

    def test_the_page_reads_the_token_from_the_url(self):
        self.assertIn("URLSearchParams(location.search)", self.page)
        self.assertIn('get("t")', self.page)

    def test_the_page_sends_the_token_on_every_request(self):
        self.assertIn("X-Yaobi-Token", self.page,
                      "api() must attach the token; without it every XHR 401s")

    def test_the_token_survives_a_reload_without_the_query_string(self):
        """Storage and cookie are both attempted: an embedded frame may block either."""
        self.assertIn("sessionStorage", self.page)
        self.assertIn("yaobi_token", self.page)

    def test_reading_the_token_tolerates_blocked_storage(self):
        """A third-party iframe can make sessionStorage throw on access."""
        marker = self.page[self.page.index("function readToken"):]
        marker = marker[: marker.index("/* ─── API")]
        self.assertIn("try", marker)
        self.assertIn("catch", marker)

    def test_the_query_parameter_is_not_stripped_from_the_url(self):
        """It is the only channel that always works, so it stays as the fallback."""
        self.assertNotIn("history.replaceState", self.page)

    def test_a_missing_token_is_reported_as_a_setup_problem(self):
        self.assertIn("需要访问令牌", self.page)


class AccessTokenTests(unittest.TestCase):
    """A public tunnel is only acceptable with a token gate in front of it."""

    @classmethod
    def setUpClass(cls):
        cls.service = ConsoleService(access_token="s3cret")
        cls.httpd = create_server(cls.service, "127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _get(self, path, headers=None):
        request = urllib.request.Request(self.base + path, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_requests_without_a_token_are_rejected(self):
        self.assertEqual(self._get("/api/bootstrap"), 401)
        self.assertEqual(self._get("/"), 401)

    def test_wrong_token_is_rejected(self):
        self.assertEqual(self._get("/api/bootstrap", {"X-Yaobi-Token": "nope"}), 401)
        self.assertEqual(self._get("/api/bootstrap?t=nope"), 401)

    def test_token_accepted_from_header_query_and_cookie(self):
        self.assertEqual(self._get("/api/bootstrap", {"X-Yaobi-Token": "s3cret"}), 200)
        self.assertEqual(self._get("/api/bootstrap", {"Authorization": "Bearer s3cret"}), 200)
        self.assertEqual(self._get("/api/bootstrap?t=s3cret"), 200)
        self.assertEqual(self._get("/api/bootstrap", {"Cookie": "yaobi_token=s3cret"}), 200)

    def test_posting_without_a_token_is_rejected(self):
        status, _ = post(self.base + "/api/interactions", {"medications": ["布洛芬"]})
        self.assertEqual(status, 401)

    def test_bootstrap_tells_the_page_auth_is_on(self):
        request = urllib.request.Request(self.base + "/api/bootstrap", headers={"X-Yaobi-Token": "s3cret"})
        with urllib.request.urlopen(request, timeout=30) as response:
            self.assertTrue(json.loads(response.read().decode())["auth_required"])


class TunnelTests(unittest.TestCase):
    def test_a_tunnel_cannot_be_opened_without_a_token(self):
        from yaobi_harness.ui.tunnel import TunnelError, open_ngrok

        with self.assertRaises(TunnelError):
            open_ngrok(8000, token="")

    def test_missing_authtoken_is_reported_clearly(self):
        import os as _os

        from yaobi_harness.ui.tunnel import TunnelError, open_ngrok

        saved = {k: _os.environ.pop(k, None) for k in ("NGROK_AUTHTOKEN", "NGROK_AUTH_TOKEN")}
        try:
            with self.assertRaises(TunnelError) as ctx:
                open_ngrok(8000, token="abc")
            self.assertTrue("pyngrok" in str(ctx.exception) or "authtoken" in str(ctx.exception))
        finally:
            for key, value in saved.items():
                if value is not None:
                    _os.environ[key] = value

    def test_tokens_are_long_and_unique(self):
        from yaobi_harness.ui.tunnel import new_token

        tokens = {new_token() for _ in range(20)}
        self.assertEqual(len(tokens), 20)
        self.assertTrue(all(len(t) >= 40 for t in tokens))

    def test_banner_warns_before_sharing(self):
        from yaobi_harness.ui.tunnel import Tunnel, banner

        text = banner(Tunnel("https://x.ngrok.app", "ngrok", "tok"), local_url="http://127.0.0.1:8000/")
        self.assertIn("不是临床部署", text)
        self.assertIn("真实患者可识别信息", text)
        self.assertIn("https://x.ngrok.app/?t=tok", text)


class ConsolePayloadTests(unittest.TestCase):
    def test_console_payload_separates_delivery_from_audit(self):
        from yaobi_harness.graph import YaobiGraphRunner
        from yaobi_harness.state import ClinicalRunState

        out = YaobiGraphRunner().run(ClinicalRunState("腰痛3月，久坐加重", role="patient"))
        payload = console_payload(out, "patient")
        self.assertEqual(set(payload), {"delivered", "audit", "meta"})
        self.assertNotIn("evidence_ledger", payload["delivered"])
        self.assertIn("safety_audit", payload["audit"])
        self.assertIn("planner_mode", payload["meta"])

    def test_console_payload_carries_the_autonomy_trace(self):
        from yaobi_harness.graph import YaobiGraphRunner
        from yaobi_harness.state import ClinicalRunState

        out = YaobiGraphRunner().run(ClinicalRunState("腰痛3月，久坐加重", role="physician"))
        self.assertIn("autonomy", console_payload(out, "physician")["audit"])

    def test_console_payload_is_json_serialisable(self):
        from yaobi_harness.graph import YaobiGraphRunner
        from yaobi_harness.state import ClinicalRunState

        out = YaobiGraphRunner().run(ClinicalRunState("突发胸痛、大汗", role="patient"))
        json.dumps(console_payload(out, "patient"), ensure_ascii=False)


class LongTurnTests(unittest.TestCase):
    """A turn is ~13 sequential model calls.

    Holding an HTTP request open for that is what produced 「出错了：Failed to
    fetch」 — the browser's own message for a connection that died, which reads as
    a backend crash and is not one. Turns run in the background and the page polls.
    """

    def setUp(self):
        self.service = ConsoleService()

    def test_a_turn_starts_immediately_and_reports_a_job(self):
        job = self.service.start_chat({"message": "腰痛3个月", "role": "patient"})
        self.assertTrue(job["job_id"])
        self.assertEqual(job["status"], "running")

    def _drain(self, job_id: str, limit: int = 400) -> dict:
        import time as _time

        for _ in range(limit):
            poll = self.service.poll_chat({"job_id": job_id})
            if poll["status"] != "running":
                return poll
            _time.sleep(0.05)
        raise AssertionError("job never finished")

    def test_polling_returns_the_turn_when_it_finishes(self):
        job = self.service.start_chat({"message": "腰痛3个月，久坐加重", "role": "patient"})
        poll = self._drain(job["job_id"])
        self.assertEqual(poll["status"], "done")
        self.assertTrue(poll["reply"]["message"])
        self.assertTrue(poll["session_id"])

    def test_progress_counts_the_agents_calls_not_just_the_sessions(self):
        """Wrapping the session's client counted 1 of 13: the runner binds the
        client into each agent at construction, so most calls never pass through
        the session's reference."""
        class Counter:
            name, model, available = "counter", "counter", True

            def chat(self, messages, tools=None, **kwargs):
                from yaobi_harness.llm.base import LLMResponse
                return LLMResponse(text="{}")

        from yaobi_harness.ui.server import _CountingLLM

        self.service.llm = _CountingLLM(Counter())
        job = self.service.start_chat({"message": "腰痛3个月", "role": "patient"})
        poll = self._drain(job["job_id"])
        self.assertGreater(poll["progress"]["llm_calls"], 5,
                           "a turn is a dozen calls, not one")

    def test_a_failed_turn_is_reported_rather_than_lost(self):
        job = self.service.start_chat({"message": "", "role": "patient"})   # empty → ValueError
        poll = self._drain(job["job_id"])
        self.assertEqual(poll["status"], "error")
        self.assertIn("消息不能为空", poll["error"])

    def test_an_unknown_job_is_a_request_error(self):
        with self.assertRaises(ValueError):
            self.service.poll_chat({"job_id": "nope"})

    def test_jobs_are_bounded(self):
        from yaobi_harness.ui.server import MAX_JOBS

        for i in range(MAX_JOBS + 5):
            self.service.start_chat({"message": f"腰痛{i}个月", "role": "patient"})
        self.assertLessEqual(len(self.service.jobs), MAX_JOBS + 1)


class ConsoleReplayTests(unittest.TestCase):
    """The console's offline re-derivation surface.

    Recording and replaying is only worth having if a replay that *did not*
    reproduce the decision says so. These tests pin the loud-failure direction as
    hard as the happy path.
    """

    CASE = {"complaint": "腰痛3月，久坐加重，右下肢麻木，无大小便异常", "role": "physician"}

    def setUp(self):
        self.service = ConsoleService()

    def _recorded(self, **extra):
        return self.service.run_case({**self.CASE, "record_journal": True, **extra})

    def test_a_run_without_recording_offers_no_journal(self):
        self.assertNotIn("journal", self.service.run_case(dict(self.CASE)))

    def test_recording_reports_what_it_captured(self):
        out = self._recorded()
        journal = out["journal"]
        self.assertEqual(journal["mode"], "record")
        self.assertGreater(journal["entries"], 0)
        self.assertIsNone(journal["path"], "a console recording must never touch disk")
        self.assertIn("panel_concurrency", journal["replay_hint"])

    def test_replaying_the_same_case_reproduces_the_decision(self):
        out = self._recorded()
        replay = self.service.replay_case({"run_id": out["journal"]["run_id"]})
        fidelity = replay["fidelity"]
        self.assertTrue(fidelity["reproduced"], fidelity)
        self.assertEqual(fidelity["differences"], [])
        self.assertEqual(fidelity["against"], "recording")
        self.assertEqual(replay["journal"]["live_after_exhaustion"], 0,
                         "every call must come from the journal, or it is not a replay")
        self.assertEqual(replay["meta"]["release_status"], out["meta"]["release_status"])

    def test_a_recording_can_be_replayed_more_than_once(self):
        """Replaying advances a cursor; the stored recording must not be consumed."""
        run_id = self._recorded()["journal"]["run_id"]
        for _ in range(3):
            self.assertTrue(self.service.replay_case({"run_id": run_id})["fidelity"]["reproduced"])

    def test_replaying_against_a_changed_case_fails_loudly(self):
        run_id = self._recorded()["journal"]["run_id"]
        replay = self.service.replay_case({
            "run_id": run_id,
            "complaint": "去年做过腰椎手术，今天突然不能排尿、会阴麻木，双腿越来越无力",
        })
        fidelity = replay["fidelity"]
        self.assertFalse(fidelity["reproduced"])
        self.assertEqual(fidelity["against"], "modified")
        self.assertTrue(fidelity["divergences"], "a content-addressed miss must be recorded")
        self.assertEqual(fidelity["divergences"][0]["differs_by"], "arguments")

    def test_an_unknown_run_id_is_a_request_error_not_a_crash(self):
        with self.assertRaises(ValueError):
            self.service.replay_case({"run_id": "no-such-run"})

    def test_recordings_are_bounded(self):
        from yaobi_harness.ui.server import MAX_RECORDINGS

        for i in range(MAX_RECORDINGS + 3):
            self.service.run_case({**self.CASE, "complaint": f"腰痛{i}月，久坐加重",
                                   "record_journal": True})
        self.assertLessEqual(len(self.service.recordings), MAX_RECORDINGS)

    def test_the_audit_says_why_the_deterministic_plan_was_used(self):
        """`planner_mode: rule` on its own is undiagnosable; the note is the fix."""
        out = self.service.run_case(dict(self.CASE))
        self.assertEqual(out["audit"]["plan"]["note"], "llm_not_configured")


class ConsoleConcurrencyControlTests(unittest.TestCase):
    def test_the_requested_concurrency_reaches_the_runner(self):
        service = ConsoleService()
        out = service.run_case({"complaint": "腰痛3月，久坐加重", "role": "physician",
                                "panel_concurrency": 3})
        self.assertEqual(out["meta"]["panel_concurrency"], 3)

    def test_an_absurd_value_is_clamped_rather_than_rejected(self):
        from yaobi_harness.ui.server import _coerce_concurrency

        self.assertEqual(_coerce_concurrency(99), 8)
        self.assertEqual(_coerce_concurrency(-4), 1)
        self.assertIsNone(_coerce_concurrency(None))
        self.assertIsNone(_coerce_concurrency(0), "0 means 'unspecified', not 'no threads'")
        with self.assertRaises(ValueError):
            _coerce_concurrency("四")

    def test_bootstrap_tells_the_page_the_default(self):
        service = ConsoleService()
        panel = service.bootstrap()["panel"]
        self.assertGreaterEqual(panel["concurrency_default"], 1)
        self.assertEqual(panel["concurrency_max"], 8)


class StaticAssetTests(unittest.TestCase):
    def test_page_declares_both_themes_and_a_favicon_free_shell(self):
        page = STATIC.read_text(encoding="utf-8")
        self.assertIn("prefers-color-scheme: dark", page)
        self.assertIn('data-theme="dark"', page)
        self.assertIn('data-theme="light"', page)

    def test_hidden_views_are_forced_hidden(self):
        """`.split` uses display:grid, which beats the UA rule for [hidden]."""
        page = STATIC.read_text(encoding="utf-8")
        self.assertIn("[hidden] { display:none !important; }", page)

    def test_grid_items_can_shrink_below_their_content(self):
        """Without min-width:0 a wide result table widens the whole layout."""
        page = STATIC.read_text(encoding="utf-8")
        self.assertIn(".split > * { min-width:0; }", page)
        self.assertIn("overflow-wrap:anywhere", page)

    def test_wide_content_scrolls_inside_its_own_container(self):
        page = STATIC.read_text(encoding="utf-8")
        self.assertIn(".tbl-wrap { overflow-x:auto", page)
        self.assertIn("pre.json", page)

    def test_page_escapes_interpolated_values(self):
        page = STATIC.read_text(encoding="utf-8")
        self.assertIn("const esc =", page)
        self.assertIn("&quot;", page)

    def test_the_page_exposes_the_replay_and_concurrency_controls(self):
        """A backend feature with no control on the page is not shipped."""
        page = STATIC.read_text(encoding="utf-8")
        for marker in ('id="recordJournal"', 'id="panelConc"', 'id="replayBtn"',
                       '"/api/replay"', "function tabReplay", "function planNote"):
            self.assertIn(marker, page, marker)

    def test_the_page_polls_instead_of_holding_a_long_request(self):
        page = STATIC.read_text(encoding="utf-8")
        self.assertIn('"/api/chat/start"', page)
        self.assertIn('"/api/chat/poll"', page)
        self.assertIn("AbortController", page)
        self.assertIn("Failed to fetch", page,
                      "the browser's own wording must be explained, not echoed blindly")

    def test_the_page_lets_the_agent_speak_first(self):
        page = STATIC.read_text(encoding="utf-8")
        self.assertIn('"/api/chat/open"', page)
        self.assertIn("boot().then(chatOpen)", page)

    def test_the_page_no_longer_claims_questions_were_blocked(self):
        """The model's questions are always asked, so the copy must not say otherwise."""
        page = STATIC.read_text(encoding="utf-8")
        self.assertNotIn("提问已被拦下", page)
        self.assertNotIn("被拦下的提问", page)

    def test_the_page_never_reuses_the_run_payload_for_a_replay(self):
        """REPLAY is separate state; overwriting LAST would erase the comparison."""
        page = STATIC.read_text(encoding="utf-8")
        self.assertIn("let REPLAY = null;", page)
        self.assertIn("REPLAY = await api(", page)


def _png(width: int = 900, height: int = 700) -> bytes:
    """A real PNG, deliberately noisy so it does not compress away.

    Size is the point of these tests: the console's JSON limit is 256 KB and this
    lands near 2 MB, which is an ordinary phone photo of an X-ray on a light box.
    """
    import random
    import struct
    import zlib

    random.seed(7)
    raw = b"".join(b"\x00" + bytes(random.randrange(256) for _ in range(width * 3))
                   for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 1))
            + chunk(b"IEND", b""))


def post_bytes(url: str, body: bytes, headers: dict) -> tuple[int, dict]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class ImageUploadTests(unittest.TestCase):
    """An X-ray does not fit through the JSON control channel, and never did.

    The console used to keep the whole base64 data URI in the page and re-send it
    inside every chat message. A 12 MB film became a 16 MB JSON body, the server
    refused it at 256 KB, and the operator saw 「出错了：请求体过大」 on the first
    question after attaching — with the attach itself having looked successful,
    because nothing had left the browser yet.
    """

    @classmethod
    def setUpClass(cls):
        cls.service = ConsoleService()
        cls.server = create_server(cls.service, "127.0.0.1", 8734)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:8734"
        cls.film = _png()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def upload(self, body=None, **overrides):
        headers = {"Content-Type": "image/png", "X-Image-Kind": "radiograph",
                   "X-Deidentified": "1", **overrides}
        return post_bytes(self.base + "/api/image/upload", body or self.film, headers)

    def test_the_test_image_is_actually_bigger_than_the_json_limit(self):
        """Otherwise every test below would pass for the wrong reason."""
        from yaobi_harness.ui.server import MAX_BODY_BYTES

        self.assertGreater(len(self.film), MAX_BODY_BYTES * 4)

    def test_an_xray_uploads_and_comes_back_as_a_handle(self):
        status, data = self.upload()
        self.assertEqual(status, 200)
        self.assertTrue(data["handle"].startswith("img_"))
        self.assertEqual(data["bytes"], len(self.film))
        self.assertEqual(len(data["sha256"]), 64)

    def test_a_handle_is_all_the_chat_payload_has_to_carry(self):
        _, up = self.upload()
        status, out = post(self.base + "/api/chat",
                           {"message": "帮我看看片子，我腰痛3个月",
                            "images": [{"kind": "radiograph", "handle": up["handle"]}]})
        self.assertEqual(status, 200, out)
        self.assertTrue(out["session_id"])

    def test_an_inline_film_is_refused_with_an_answer_not_a_dropped_connection(self):
        """Rejecting on the header alone left the client still writing into a
        socket nobody was reading — a broken pipe, which a browser reports as yet
        another 「Failed to fetch」."""
        import base64

        status, out = post(self.base + "/api/chat", {
            "message": "看片",
            "images": [{"kind": "radiograph", "deidentified": True,
                        "ref": "data:image/png;base64," + base64.b64encode(self.film).decode()}],
        })
        self.assertEqual(status, 413)
        self.assertIn("请求体过大", out["error"])
        self.assertIn("/api/image/upload", out["error"], "the error must say what to do instead")

    def test_the_attestation_is_required_per_upload(self):
        status, out = self.upload(**{"X-Deidentified": "0"})
        self.assertEqual(status, 400)
        self.assertIn("去标识化", out["error"])

    def test_an_unknown_kind_is_refused(self):
        status, out = self.upload(**{"X-Image-Kind": "chest_ct_but_made_up"})
        self.assertEqual(status, 400)

    def test_a_non_image_body_is_refused(self):
        status, out = post_bytes(
            self.base + "/api/image/upload", b"not an image at all",
            {"Content-Type": "application/pdf", "X-Image-Kind": "radiograph", "X-Deidentified": "1"})
        self.assertEqual(status, 400)
        self.assertIn("不支持", out["error"])

    def test_an_expired_handle_says_so_rather_than_running_without_the_image(self):
        status, out = post(self.base + "/api/chat",
                           {"message": "看片", "images": [{"kind": "radiograph",
                                                           "handle": "img_neverexisted"}]})
        self.assertEqual(status, 400)
        self.assertIn("重新上传", out["error"])

    def test_the_store_is_bounded_by_bytes_not_by_count(self):
        """A count bound over a variable-size object is not a bound: twenty-four
        films at the 12 MB ceiling is 288 MB resident, which a Colab kernel
        notices."""
        service = ConsoleService()
        service.max_upload_store_bytes = 300 * 1024
        for i in range(12):
            service.store_image(_png(200, 150 + i), mime="image/png", kind="other")
        resident = sum(e["bytes"] for e in service.uploads.values())
        self.assertLessEqual(resident, service.max_upload_store_bytes)
        self.assertLess(len(service.uploads), 12, "nothing was evicted")
        self.assertTrue(service.uploads, "eviction must never empty the store")

    def test_raw_bytes_are_stored_rather_than_the_inflated_encoding(self):
        service = ConsoleService()
        film = _png(400, 300)
        handle = service.store_image(film, mime="image/png", kind="other")["handle"]
        self.assertEqual(service.uploads[handle]["raw"], film)
        self.assertTrue(ConsoleService._data_uri(service.uploads[handle]).startswith("data:image/png;base64,"))

    def test_the_same_photo_uploaded_as_two_kinds_is_two_entries(self):
        """Keying on content alone made the second upload silently rewrite the
        first one's kind."""
        service = ConsoleService()
        film = _png(300, 200)
        first = service.store_image(film, mime="image/png", kind="radiograph")
        second = service.store_image(film, mime="image/png", kind="tongue")
        self.assertNotEqual(first["handle"], second["handle"])
        self.assertEqual(service.uploads[first["handle"]]["kind"], "radiograph")
        self.assertEqual(service.uploads[second["handle"]]["kind"], "tongue")

    def test_the_kind_comes_from_the_upload_not_from_the_chat_payload(self):
        """The de-identification attestation was made against that kind."""
        from yaobi_harness.ui.server import _coerce_images

        service = ConsoleService()
        handle = service.store_image(_png(120, 90), mime="image/png", kind="radiograph")["handle"]
        coerced = _coerce_images([{"handle": handle, "kind": "tongue"}], service.uploads)
        self.assertEqual(coerced[0]["kind"], "radiograph")


class UnreadImageTests(unittest.TestCase):
    """An attached image must never end a run in silence.

    Someone photographed a film, ticked the box and waited. The one unacceptable
    outcome is an answer that reads as though nothing was attached.
    """

    def test_a_run_with_no_vision_model_says_the_image_was_not_read(self):
        from yaobi_harness.graph import YaobiGraphRunner
        from yaobi_harness.state import ClinicalRunState

        state = ClinicalRunState("腰痛3个月", role="patient")
        state.images = [{"kind": "radiograph", "ref": "data:image/png;base64,AA==",
                         "deidentified": True}]
        YaobiGraphRunner().run(state)
        self.assertTrue(any("未配置视觉模型" in w and "未被判读" in w for w in state.warnings),
                        f"nothing explained the unread image: {state.warnings}")

    def test_a_plan_that_omits_the_read_still_reports_the_image_as_unread(self):
        from yaobi_harness.graph import YaobiGraphRunner
        from yaobi_harness.state import ClinicalRunState, Task

        state = ClinicalRunState("腰痛3个月", role="patient")
        state.images = [{"kind": "radiograph", "ref": "data:image/png;base64,AA==",
                         "deidentified": True}]
        state.tasks = [Task("T1", "TimelineAgent", "标准化病历与时间线")]
        YaobiGraphRunner().run(state)
        self.assertNotIn("image_findings", state.outputs)
        self.assertTrue(any("图片" in w and "未" in w for w in state.warnings),
                        f"nothing explained the unread image: {state.warnings}")

    def test_the_planner_is_told_that_images_are_attached(self):
        """A model-authored plan never scheduled the read, because the planner's
        context did not mention that anything had been uploaded."""
        from yaobi_harness.agent.planner import build_planner_prompt
        from yaobi_harness.state import ClinicalRunState

        state = ClinicalRunState("腰痛3个月", role="patient")
        state.images = [{"kind": "radiograph", "deidentified": True}]
        prompt = build_planner_prompt(state)
        self.assertIn("radiograph", prompt[1]["content"])
        self.assertIn("VisionAgent", prompt[0]["content"])


class ProgressStreamTests(unittest.TestCase):
    """A turn must be watchable, not a spinner."""

    def setUp(self):
        self.service = ConsoleService()

    def _drain(self, job_id: str, limit: int = 400):
        import time as _time

        cursor, events = 0, []
        for _ in range(limit):
            poll = self.service.poll_chat({"job_id": job_id, "cursor": cursor})
            cursor = poll["cursor"]
            events += poll["events"]
            if poll["status"] != "running":
                return poll, events
            _time.sleep(0.02)
        raise AssertionError("job never finished")

    def test_a_turn_streams_its_agents_and_tool_calls(self):
        job = self.service.start_chat({"message": "腰痛3个月，久坐加重", "role": "patient"})
        poll, events = self._drain(job["job_id"])
        self.assertEqual(poll["status"], "done")
        kinds = {e["kind"] for e in events}
        self.assertIn("agent", kinds)
        self.assertIn("tool", kinds)
        labels = {e["label"] for e in events}
        self.assertIn("IntakeAgent", labels)
        self.assertIn("CriticAgent", labels)

    def test_every_event_arrives_exactly_once(self):
        job = self.service.start_chat({"message": "腰痛3个月", "role": "patient"})
        _, events = self._drain(job["job_id"])
        seqs = [e["seq"] for e in events]
        self.assertEqual(len(seqs), len(set(seqs)))
        self.assertEqual(seqs, sorted(seqs))

    def test_tool_events_report_names_never_the_patients_words(self):
        """The stream renders in a browser tab that may be on a shared screen."""
        job = self.service.start_chat(
            {"message": "我叫张三，住在城东，腰痛3个月，晚上疼得睡不着", "role": "patient"})
        _, events = self._drain(job["job_id"])
        for event in events:
            if event["kind"].startswith("tool"):
                self.assertNotIn("张三", event["detail"])
                self.assertNotIn("城东", event["detail"])

    def test_no_tool_argument_can_carry_free_text_into_the_stream(self):
        """Driven into *every* argument of *every* tool, in three shapes.

        The first attempt allowlisted argument names and truncated their values,
        and both halves were wrong: a model-driven tool loop picks its own
        arguments so any field can receive anything, and twenty-four characters
        of Chinese is a whole sentence — 「我叫张三，住城东，身份证110101…」 went
        through a 24-character cap with the name and the ID intact.
        """
        import inspect

        from yaobi_harness.tools import ToolRegistry, _tool_arg_preview

        secret = "我叫张三，住城东，身份证110101，腰痛三个月了每天晚上都疼"
        registry = ToolRegistry()
        for name in sorted(registry.tools):
            fn = getattr(registry, name, None)
            if fn is None:
                continue
            params = [p for p in inspect.signature(fn).parameters if p != "self"]
            for shape in (lambda s: s, lambda s: [s, s], lambda s: {"a": s}):
                with self.subTest(tool=name):
                    preview = _tool_arg_preview({p: shape(secret) for p in params})
                    self.assertNotIn(secret[:4], preview)

    def test_closed_vocabulary_values_still_show_because_that_is_the_point(self):
        from yaobi_harness.tools import _tool_arg_preview

        self.assertEqual(_tool_arg_preview({"axis_id": "cauda_equina", "tier": "RED_FLAG"}),
                         "axis_id=cauda_equina, tier=RED_FLAG")
        self.assertEqual(_tool_arg_preview({"kind": "radiograph"}), "kind=radiograph")
        self.assertEqual(_tool_arg_preview({"conditions": ["elderly"]}), "conditions=elderly")

    def test_a_value_outside_its_vocabulary_fails_to_the_name_alone(self):
        from yaobi_harness.tools import _tool_arg_preview

        self.assertEqual(_tool_arg_preview({"axis_id": "不是真的轴"}), "axis_id")
        self.assertEqual(_tool_arg_preview({"kind": "<整段主诉>"}), "kind")

    def test_a_denied_tool_call_is_visible(self):
        """「为什么模型没调用工具」 is usually 「调用了，被技能策略拒了」."""
        from yaobi_harness import progress
        from yaobi_harness.tools import CapabilityBroker, ToolRegistry

        sink = progress.ProgressSink()
        broker = CapabilityBroker("patient", "urgent", skill_registry=None)
        with progress.bound(sink, "FormulaAgent"):
            ToolRegistry().call(broker, "formula_composition_search", pattern="气滞血瘀证")
        denied = [e for e in sink.since(0) if e["kind"] == "tool_denied"]
        self.assertTrue(denied)
        self.assertIn("urgent_mode_forbids", denied[0]["detail"])

    def test_the_page_renders_the_stream(self):
        page = STATIC.read_text(encoding="utf-8")
        self.assertIn("renderEvents(", page)
        self.assertIn("poll.events", page)
        self.assertIn("cursor", page)
        self.assertIn("思考过程", page, "the model's reasoning must be reachable in the UI")

    def test_the_page_uploads_images_out_of_band(self):
        page = STATIC.read_text(encoding="utf-8")
        self.assertIn("/api/image/upload", page)
        self.assertIn("X-Deidentified", page)
        self.assertNotIn("readAsDataURL", page,
                         "a film must not be turned into base64 in the page again")
        self.assertIn("i.handle", page)


class ProductSurfaceTests(unittest.TestCase):
    """Behaviour of the console as a product, not as an API.

    Everything here is a contract the page has to keep, checked against the page
    source because there is no browser in this suite. They are deliberately
    behavioural — 'the deliverable is reachable from where it was produced' —
    rather than assertions about markup.
    """

    def setUp(self):
        self.page = STATIC.read_text(encoding="utf-8")

    def test_the_note_is_delivered_inside_the_conversation(self):
        """It used to be announced with a line telling the clinician to switch to
        another page and find one of eight tabs — four steps and a context switch
        to reach the one artefact the consultation exists to produce."""
        self.assertIn("renderNoteCard(reply.clinical_note)", self.page)
        self.assertNotIn("在「单次运行」页的「病历摘要」标签查看与复制", self.page)

    def test_the_note_can_be_copied_and_downloaded(self):
        self.assertIn("function copyText", self.page)
        self.assertIn("function downloadText", self.page)
        self.assertIn('data-act="copy"', self.page)
        self.assertIn('data-act="download"', self.page)

    def test_copy_never_claims_a_success_it_did_not_have(self):
        """Clipboard access is blocked in a cross-origin iframe and on plain http
        — Colab, ngrok-over-http and a LAN address between them. A note the
        clinician believes is on the clipboard and is not is worse than a button
        that says it failed."""
        self.assertIn("复制失败", self.page)
        self.assertIn("execCommand", self.page)

    def test_the_downloaded_filename_carries_no_patient_identifier(self):
        self.assertIn("function noteFilename", self.page)
        self.assertIn("run_id", self.page.split("function noteFilename")[1][:400])

    def test_a_question_chip_keeps_what_the_user_already_typed(self):
        """The old handler cleared the box and moved the question into the
        placeholder: a half-written answer was destroyed, and the question then
        vanished the moment typing resumed."""
        self.assertIn("function answerChip", self.page)
        body = self.page.split("function answerChip")[1][:400]
        self.assertIn("input.value.trim()", body)
        self.assertNotIn("input.placeholder = item.question", self.page)

    def test_the_input_grows_with_the_answer(self):
        self.assertIn("function autoGrow", self.page)
        self.assertIn('$("#chatInput").addEventListener("input"', self.page)

    def test_resetting_re_opens_with_the_agent_speaking(self):
        """Reset used to drop the user back to a blank box — the precise thing
        the agent opening first exists to avoid, on the one path that looks most
        like starting over."""
        handler = self.page.split('$("#chatReset").addEventListener')[1][:1400]
        self.assertIn("chatOpen()", handler)
        self.assertNotIn("chat-empty", handler,
                         "reset must not rebuild the empty-state block itself")

    def test_resetting_confirms_before_destroying_a_consultation(self):
        handler = self.page.split('$("#chatReset").addEventListener')[1][:1400]
        self.assertIn("confirm(", handler)
        self.assertIn("CHAT.turns > 0", handler, "a fresh session must not prompt")

    def test_the_patient_can_say_they_have_nothing_more_to_add(self):
        """Until now the only way to reach the model's "enough" was to keep
        answering; someone with nothing more had to say 不知道 three times before
        the stall detector noticed."""
        self.assertIn("DONE_MESSAGE", self.page)
        self.assertIn("我说完了，请给结论", self.page)
        # It is an ordinary message through the ordinary pipeline, not a flag.
        self.assertIn("chatSend(DONE_MESSAGE)", self.page)

    def test_the_way_out_is_not_offered_in_an_emergency(self):
        self.assertIn('reply.risk_mode !== "urgent"', self.page)

    def test_the_llm_toggle_describes_what_it_actually_does(self):
        """Its label said 「用于信息抽取与措辞改写；不新增临床内容」, which describes a
        design three commits dead — the model now triages, questions and writes
        every reply. A setting that misdescribes itself is a defect: the operator
        turning it off does not know what they are turning off."""
        self.assertNotIn("用于信息抽取与措辞改写", self.page)
        self.assertIn("由模型驱动本次对话", self.page)

    def test_an_unconfigured_model_is_reported_at_the_point_of_use(self):
        self.assertIn('$("#llmOff")', self.page)
        self.assertIn('id="llmOff"', self.page)
        self.assertIn('$("#chatUseLlm").disabled = true', self.page)

    def test_the_conversation_comes_first_on_a_narrow_screen(self):
        """Collapsed to one column the settings card came first by DOM order, so a
        phone opened onto a role selector, two checkboxes, a coverage ring, an
        image uploader and a JSON dump — every one with a working default."""
        self.assertIn("#view-chat > .composer { order:2; }", self.page)
        self.assertIn("#view-chat > .card:not(.composer) { order:1; }", self.page)

    def test_an_image_request_scrolls_the_uploader_into_view(self):
        """The uploader is in the settings column, which on a narrow screen sits
        below the conversation — 「请拍一张舌象」 would otherwise arrive with
        nowhere on screen to do it."""
        self.assertIn('block: "center"', self.page)

    def test_a_removed_or_sent_image_releases_its_preview(self):
        # Removed from the tray, sent with a turn, and cleared by reset.
        self.assertEqual(self.page.count("URL.revokeObjectURL"), 4)


class ModelIdentityRedactionTests(unittest.TestCase):
    """The vendor and model names never reach a screen.

    Which model sits behind a clinical assistant is a procurement and
    contractual matter, not something a screen share, a demo or a Colab notebook
    committed with its cell outputs should settle — and it is of no use to the
    clinician reading the answer. It stays available everywhere an operator
    actually works: the CLI, the logs, and the journal file, which needs the real
    name because it is part of every request's content address.

    Written as a sweep over whole response bodies rather than as assertions on
    known fields. The field-by-field version of this passed while
    ``journal.meta.llm_model`` was still going out on two routes.
    """

    VENDOR = "acmevendor"
    MODEL = "AcmeModel-XYZ-9"
    VISION_MODEL = "AcmeVision-Secret"

    class Named:
        available = True

        def __init__(self, name, model):
            self.name, self.model = name, model

        def chat(self, messages, **kwargs):
            return LLMResponse(text=json.dumps({
                "triage": "routine", "adequate": True, "workup_now": True,
                "questions": [], "facts": {}, "message": "好的", "reply": "好的",
                "differentials": ["腰肌劳损"], "primary_pattern": "气滞血瘀证",
                "evidence": ["刺痛"], "differential_patterns": ["寒湿"],
                "counter_evidence_needed": ["舌脉"], "similar": ["12例"],
                "counterexamples": [], "limitation": "单一专家经验",
            }, ensure_ascii=False), model=self.model)

    class Vision:
        available = True
        phi_precheck = True
        borrowed = True

        def __init__(self, model, chat_client):
            self.model, self.chat_client = model, chat_client

        def read(self, image, kind="other", context="", budget=None):
            from yaobi_harness.vision.client import ImageRead

            return ImageRead(image_sha256="a" * 64, image_kind=kind, readable=True,
                             observations=["骨皮质连续"], model=self.model)

    @classmethod
    def setUpClass(cls):
        from yaobi_harness.ui.server import _CountingLLM

        cls.service = ConsoleService()
        client = cls.Named(cls.VENDOR, cls.MODEL)
        cls.service.llm = _CountingLLM(client)
        cls.service.vision = cls.Vision(cls.VISION_MODEL, client)
        cls.service.tools.vision = cls.service.vision
        cls.httpd = create_server(cls.service, "127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def bodies(self):
        """Every response a browser can obtain, as raw text."""
        out = {"GET /": get(self.base + "/")[1],
               "GET /api/bootstrap": json.dumps(get(self.base + "/api/bootstrap")[1],
                                                ensure_ascii=False)}
        _, run = post(self.base + "/api/run", {
            "complaint": "腰痛3月，刺痛固定，夜间不痛醒，大小便正常", "role": "physician",
            "allow_prescription": True, "record_journal": True,
            "images": [{"kind": "radiograph", "deidentified": True,
                        "ref": "data:image/png;base64,AA=="}]})
        out["POST /api/run"] = json.dumps(run, ensure_ascii=False)
        run_id = (run.get("journal") or {}).get("run_id")
        if run_id:
            _, replay = post(self.base + "/api/replay", {"run_id": run_id})
            out["POST /api/replay"] = json.dumps(replay, ensure_ascii=False)
        _, chat = post(self.base + "/api/chat", {"message": "腰痛3个月", "role": "patient"})
        out["POST /api/chat"] = json.dumps(chat, ensure_ascii=False)
        return out

    def test_no_response_a_browser_can_obtain_names_the_model(self):
        for label, body in self.bodies().items():
            for needle in (self.VENDOR, self.MODEL, self.VISION_MODEL):
                with self.subTest(route=label, needle=needle):
                    self.assertNotIn(needle.lower(), body.lower())

    def test_a_replay_of_a_recorded_run_does_not_leak_it_either(self):
        """The route that kept publishing ``journal.meta.llm_model`` after the
        other two were fixed."""
        bodies = self.bodies()
        self.assertIn("POST /api/replay", bodies, "the replay route did not run")

    def test_the_page_answers_whether_a_model_is_driving_the_run(self):
        """Redacting must not cost the one thing the badge is for: a
        deterministic run and a model-driven one produce very different
        consultations, and confusing them is a real error."""
        _, data = get(self.base + "/api/bootstrap")
        self.assertTrue(data["llm"]["available"])
        self.assertTrue(data["llm"]["configured"])
        self.assertTrue(data["vision"]["configured"])

    def test_the_operator_paths_keep_the_real_name(self):
        from yaobi_harness.llm.factory import describe_client

        described = describe_client(self.Named(self.VENDOR, self.MODEL))
        self.assertEqual(described["provider"], self.VENDOR)
        self.assertEqual(described["model"], self.MODEL)

    def test_the_journal_file_keeps_the_real_name(self):
        """It is part of every request's content address; an offline replay
        diverges on the name alone without it."""
        import tempfile

        from yaobi_harness.journal import Journal

        with tempfile.TemporaryDirectory() as tmp:
            journal = Journal(Path(tmp) / "run.jsonl", mode="record")
            journal.write_meta({"llm_model": self.MODEL, "llm_provider": self.VENDOR})
            self.assertEqual(journal.meta["llm_model"], self.MODEL)
            self.assertNotIn("llm_model", journal.summary()["meta"])

    def test_an_operator_can_opt_back_in(self):
        import os

        from yaobi_harness.llm.factory import SHOW_MODEL_ENV, public_client_info

        os.environ[SHOW_MODEL_ENV] = "1"
        try:
            info = public_client_info(self.Named(self.VENDOR, self.MODEL))
            self.assertEqual(info["model"], self.MODEL)
        finally:
            os.environ.pop(SHOW_MODEL_ENV, None)
        self.assertNotIn("model", public_client_info(self.Named(self.VENDOR, self.MODEL)))


if __name__ == "__main__":
    unittest.main()


class UploadPrereadTests(unittest.TestCase):
    """The vision model is activated at the earliest possible moment: the upload.

    Between clicking 上传 and finishing the question there are usually tens of
    seconds — enough for the PHI pre-check and the read to complete in the
    background. The turn then replays the finding from cache instead of paying a
    serial multimodal call at its slowest point. The attestation gate is
    unchanged: nothing reaches the store without the de-identification header.
    """

    class CountingVision:
        available = True
        model = "v"
        chat_client = None
        phi_precheck = True
        borrowed = False

        def __init__(self):
            self.reads = 0

        def read(self, image, kind="other", context="", budget=None):
            from yaobi_harness.vision.client import ImageRead

            self.reads += 1
            return ImageRead(image_sha256="a" * 64, image_kind=kind, readable=True,
                             observations=["预判读：骨皮质连续"], model="v")

    def _service(self):
        service = ConsoleService()
        service.vision = self.CountingVision()
        service.tools.vision = service.vision
        return service

    def _upload(self, service, raw=b"fake-png-bytes", kind="radiograph"):
        out = service.store_image(raw, mime="image/png", kind=kind)
        # The pre-read runs on a background thread; tests wait for it the same
        # way a turn does, through the record's completion event.
        for record in list(service.image_prereads.values()):
            record["done"].wait(5)
        return out

    def test_an_upload_starts_the_read_immediately(self):
        service = self._service()
        out = self._upload(service)
        self.assertTrue(out["preread"])
        self.assertEqual(service.vision.reads, 1)
        (record,) = service.image_prereads.values()
        self.assertIn("预判读：骨皮质连续", record["read"]["observations"])

    def test_the_first_turn_replays_the_upload_read_for_free(self):
        service = self._service()
        out = self._upload(service)
        result = service.chat({"message": "帮我看看片子，腰痛3个月", "use_llm": False,
                               "images": [{"kind": "radiograph", "handle": out["handle"]}]})
        self.assertEqual(service.vision.reads, 1, "the turn re-read a film pre-read at upload")
        session = service.sessions[result["session_id"]]
        self.assertIn("预判读：骨皮质连续",
                      session.state.outputs["image_findings"]["observations"])

    def test_a_one_shot_run_adopts_the_upload_read_too(self):
        service = self._service()
        out = self._upload(service)
        result = service.run_case({"complaint": "腰痛3月，请看片子", "use_llm": False,
                                   "images": [{"kind": "radiograph", "handle": out["handle"]}]})
        self.assertEqual(service.vision.reads, 1)
        self.assertIn("预判读：骨皮质连续",
                      json.dumps(result, ensure_ascii=False))

    def test_a_recorded_run_reads_fresh_so_the_journal_holds_the_call(self):
        """A pre-read cache hit cannot be replayed: the cache is process state,
        not journal content. A journaled run must make its calls for real."""
        service = self._service()
        out = self._upload(service)
        service.run_case({"complaint": "腰痛3月，请看片子", "use_llm": False,
                          "record_journal": True,
                          "images": [{"kind": "radiograph", "handle": out["handle"]}]})
        self.assertEqual(service.vision.reads, 2, "the recorded run silently used the pre-read")

    def test_re_uploading_the_same_film_does_not_read_it_again(self):
        service = self._service()
        self._upload(service)
        self._upload(service)
        self.assertEqual(service.vision.reads, 1)

    def test_no_vision_model_means_no_preread_and_says_so(self):
        service = ConsoleService()
        service.vision = None
        service.tools.vision = None
        out = service.store_image(b"fake", mime="image/png", kind="radiograph")
        self.assertFalse(out["preread"])
        self.assertEqual(service.image_prereads, {})

    def test_the_preread_cache_is_bounded(self):
        from yaobi_harness.ui.server import MAX_PREREADS

        service = self._service()
        for index in range(MAX_PREREADS + 4):
            self._upload(service, raw=b"film-%d" % index)
        self.assertLessEqual(len(service.image_prereads), MAX_PREREADS)

    def test_a_preread_failure_falls_through_to_the_ordinary_read(self):
        """A flaky endpoint at upload time must cost nothing but the retry."""
        service = self._service()

        original = service.vision.read
        calls = {"n": 0}

        def flaky(image, kind="other", context="", budget=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("端点抖动")
            return original(image, kind=kind, context=context, budget=budget)

        service.vision.read = flaky
        out = self._upload(service)
        self.assertTrue(out["preread"])
        result = service.chat({"message": "帮我看看片子", "use_llm": False,
                               "images": [{"kind": "radiograph", "handle": out["handle"]}]})
        session = service.sessions[result["session_id"]]
        self.assertIn("骨皮质连续",
                      json.dumps(session.state.outputs.get("image_findings") or {}, ensure_ascii=False))
