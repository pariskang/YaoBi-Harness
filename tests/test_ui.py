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
        self.assertIn("provider", data["llm"])
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


if __name__ == "__main__":
    unittest.main()
