"""Live-transport tests for the LLM adapters.

These run a local HTTP server that speaks the OpenAI-compatible contract, so
the request path (URL, headers, body) and the response path (content, tool
calls, usage, retries) are exercised end to end without touching a real
provider.
"""

from __future__ import annotations

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ.setdefault("YAOBI_DEID_KEY", "unit-test-fixed-key")
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from yaobi_harness.agent.planner import PlannerAgent
from yaobi_harness.llm.base import LLMError
from yaobi_harness.llm.factory import build_client
from yaobi_harness.state import ClinicalRunState

RECORDED: list[dict] = []
FAIL_TIMES = {"count": 0}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence test output
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        # urllib normalises header casing on the wire; compare case-insensitively.
        headers = {k.lower(): v for k, v in self.headers.items()}
        RECORDED.append({"path": self.path, "headers": headers, "body": body})

        if self.path.startswith("/flaky") and FAIL_TIMES["count"] > 0:
            FAIL_TIMES["count"] -= 1
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error":"upstream busy"}')
            return
        if self.path.startswith("/unauthorized"):
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":"bad key"}')
            return

        payload = {
            "model": body.get("model", "test-model"),
            "choices": [{"message": {"content": json.dumps({"tasks": [
                {"task_id": "P1", "agent": "BiomedicalAgent", "objective": "先做西医鉴别",
                 "required_tools": ["clinical_guideline_search"]},
            ]}, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 13},
        }
        if self.path.startswith("/v1/text/chatcompletion_v2") or "chatcompletion_v2" in self.path:
            payload["base_resp"] = {"status_code": 0, "status_msg": "success"}
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class LLMTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        RECORDED.clear()
        FAIL_TIMES["count"] = 0

    def test_litellm_round_trip(self):
        client = build_client("litellm", api_key="sk-test", model="gpt-4o", base_url=self.base)
        response = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(response.total_tokens, 24)
        self.assertEqual(RECORDED[0]["path"], "/v1/chat/completions")
        self.assertEqual(RECORDED[0]["headers"]["authorization"], "Bearer sk-test")
        self.assertEqual(RECORDED[0]["body"]["model"], "gpt-4o")

    def test_poe_round_trip(self):
        client = build_client("poe", api_key="poe-key", model="Claude-Sonnet-4.5", base_url=f"{self.base}/v1")
        client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(RECORDED[0]["headers"]["authorization"], "Bearer poe-key")
        self.assertEqual(RECORDED[0]["body"]["model"], "Claude-Sonnet-4.5")

    def test_minimax_round_trip_uses_the_openai_compatible_path(self):
        """MiniMax's documented surface is OpenAI-compatible now.

        The old ``/text/chatcompletion_v2`` path on ``api.minimax.chat`` is gone;
        pinning it here meant the harness shipped an endpoint that 404s.
        """
        client = build_client("minimax", api_key="mm-key", model="MiniMax-M3",
                              base_url=f"{self.base}/v1", group_id="g42")
        response = client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("/v1/chat/completions", RECORDED[0]["path"])
        self.assertIn("GroupId=g42", RECORDED[0]["path"])
        self.assertEqual(RECORDED[0]["body"]["model"], "MiniMax-M3")
        self.assertTrue(response.json())

    def test_minimax_without_a_group_id_sends_a_clean_path(self):
        client = build_client("minimax", api_key="mm-key", base_url=f"{self.base}/v1")
        client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(RECORDED[0]["path"], "/v1/chat/completions")

    def test_minimax_surfaces_a_base_resp_error_instead_of_an_empty_answer(self):
        """The API still returns this envelope; reading it as success is worse."""
        from yaobi_harness.llm.providers import MiniMaxClient

        client = MiniMaxClient(api_key="k", base_url=f"{self.base}/v1")
        with self.assertRaises(LLMError) as ctx:
            client.parse_response({"base_resp": {"status_code": 1004, "status_msg": "invalid api key"}})
        self.assertIn("1004", str(ctx.exception))

    def test_azure_round_trip_uses_api_key_header_and_deployment_path(self):
        client = build_client("azure", api_key="az-key", model="my-deploy", base_url=self.base,
                              api_version="2024-10-21")
        client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("/openai/deployments/my-deploy/chat/completions", RECORDED[0]["path"])
        self.assertIn("api-version=2024-10-21", RECORDED[0]["path"])
        self.assertEqual(RECORDED[0]["headers"]["api-key"], "az-key")
        self.assertNotIn("authorization", RECORDED[0]["headers"])

    def test_tool_schemas_are_sent_in_openai_shape(self):
        from yaobi_harness.tools import tool_specs

        client = build_client("litellm", api_key="k", model="m", base_url=self.base)
        client.chat([{"role": "user", "content": "hi"}], tools=tool_specs()[:2])
        sent = RECORDED[0]["body"]["tools"]
        self.assertEqual(sent[0]["type"], "function")
        self.assertEqual(sent[0]["function"]["name"], "red_flag_evidence_search")
        self.assertIn("properties", sent[0]["function"]["parameters"])

    def test_retryable_status_is_retried_then_succeeds(self):
        FAIL_TIMES["count"] = 2
        client = build_client("litellm", api_key="k", model="m", base_url=f"{self.base}/flaky", retries=3)
        # /flaky/v1/chat/completions -> handled by the same handler
        response = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(len(RECORDED), 3)
        self.assertEqual(response.total_tokens, 24)

    def test_non_retryable_status_raises_immediately(self):
        client = build_client("litellm", api_key="k", model="m", base_url=f"{self.base}/unauthorized", retries=3)
        with self.assertRaises(LLMError):
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(len(RECORDED), 1)

    def test_planner_accepts_a_plan_served_over_http(self):
        client = build_client("litellm", api_key="k", model="m", base_url=self.base)
        state = ClinicalRunState("腰痛3月，久坐加重", role="physician")
        PlannerAgent(client).run(state)
        self.assertEqual(state.planner_mode, "llm")
        self.assertEqual([t.agent for t in state.tasks], ["BiomedicalAgent", "CriticAgent"])
        self.assertEqual(state.budget.used_llm_tokens, 24)

    def test_unreachable_provider_falls_back_to_rule_plan(self):
        client = build_client("litellm", api_key="k", model="m", base_url="http://127.0.0.1:1/v1", retries=1)
        state = ClinicalRunState("腰痛3月，久坐加重", role="physician")
        PlannerAgent(client).run(state)
        self.assertEqual(state.planner_mode, "rule")
        self.assertTrue(state.tasks)


if __name__ == "__main__":
    unittest.main()
