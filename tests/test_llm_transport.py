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
from yaobi_harness.llm.base import LLMError, LLMResponse
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


class ReasoningModelTests(unittest.TestCase):
    """A reasoning model's scratch pad must never be displayed or parsed.

    Both halves were live defects. The reported transcript opened with a paragraph
    of the model talking to itself about the system prompt — and worse, a trace
    containing an example ``{"age": 99}`` was parsed as the answer, writing an
    invented age into the clinical facts. A fabricated fact is a much bigger
    problem than an ugly one.
    """

    def test_a_closed_block_is_separated(self):
        from yaobi_harness.llm.base import split_reasoning

        answer, reasoning = split_reasoning("<think>scratch</think>\n\n真正的回答")
        self.assertEqual(answer, "真正的回答")
        self.assertIn("scratch", reasoning)

    def test_every_tag_spelling_is_recognised(self):
        from yaobi_harness.llm.base import split_reasoning

        for tag in ("think", "thinking", "reasoning", "Thinking"):
            with self.subTest(tag=tag):
                answer, _ = split_reasoning(f"<{tag}>x</{tag}>答案")
                self.assertEqual(answer, "答案")

    def test_an_unclosed_block_yields_no_answer(self):
        """Truncated mid-thought: returning the partial trace as the answer would
        be worse than returning nothing."""
        from yaobi_harness.llm.base import split_reasoning

        answer, reasoning = split_reasoning("<think>cut off half way")
        self.assertEqual(answer, "")
        self.assertIn("cut off", reasoning)

    def test_text_with_no_block_is_untouched(self):
        from yaobi_harness.llm.base import split_reasoning

        self.assertEqual(split_reasoning("普通回答"), ("普通回答", ""))

    def test_json_inside_the_trace_is_never_parsed(self):
        """The bug in one line: the extractor took the first object it found."""
        from yaobi_harness.llm.base import extract_json_with_repairs

        payload, repairs = extract_json_with_repairs(
            '<think>I could output {"age": 99, "sex": "male"}</think>\n{"onset": "3个月"}')
        self.assertEqual(payload, {"onset": "3个月"})
        self.assertIn("stripped_reasoning", repairs)

    def test_a_trace_with_no_answer_parses_to_nothing(self):
        from yaobi_harness.llm.base import extract_json_with_repairs

        payload, repairs = extract_json_with_repairs('<think>{"age": 99}</think>')
        self.assertIsNone(payload)
        self.assertIn("reasoning_only", repairs)

    def _response_for(self, message: dict) -> LLMResponse:
        """Run one real HTTP round trip so the provider adapter is what is tested."""
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
                body = json.dumps({"choices": [{"message": message}], "usage": {}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = build_client("litellm", api_key="k", model="m",
                                  base_url=f"http://127.0.0.1:{server.server_port}/v1")
            return client.chat([{"role": "user", "content": "hi"}])
        finally:
            server.shutdown()
            server.server_close()

    def test_the_provider_strips_it_before_anyone_sees_it(self):
        response = self._response_for({"content": "<think>内心独白</think>\n您好，哪里不舒服？"})
        self.assertEqual(response.text, "您好，哪里不舒服？")
        self.assertIn("内心独白", response.reasoning)

    def test_a_side_channel_reasoning_field_is_captured_too(self):
        """DeepSeek-style gateways return it separately rather than inline."""
        response = self._response_for({"content": "答案", "reasoning_content": "推理过程"})
        self.assertEqual(response.text, "答案")
        self.assertIn("推理过程", response.reasoning)

    def test_the_opening_never_carries_a_trace(self):
        from yaobi_harness.conversation import ConversationSession
        from yaobi_harness.graph import YaobiGraphRunner

        class Reasoner:
            name, model, available = "r", "r", True

            def chat(self, messages, tools=None, **kwargs):
                return LLMResponse(text="<think>让我想想开场白</think>\n您好，哪里不舒服？")

        reply = ConversationSession(role="patient",
                                    runner=YaobiGraphRunner(llm=Reasoner())).open()
        self.assertNotIn("<think>", reply.message)
        self.assertEqual(reply.message, "您好，哪里不舒服？")

    def test_a_reply_that_is_only_a_trace_falls_back(self):
        from yaobi_harness.conversation import ConversationSession
        from yaobi_harness.graph import YaobiGraphRunner

        class OnlyThinks:
            name, model, available = "r", "r", True

            def chat(self, messages, tools=None, **kwargs):
                if "正在直接和" in messages[0]["content"]:
                    return LLMResponse(text="<think>还在想，没写完")
                return LLMResponse(text="{}")

        convo = ConversationSession(role="patient", runner=YaobiGraphRunner(llm=OnlyThinks()))
        reply = convo.send("腰痛3个月")
        self.assertEqual(reply.composer, "template")
        self.assertNotIn("<think>", reply.message)
        self.assertTrue(any("思考过程" in w for w in convo.state.warnings))
