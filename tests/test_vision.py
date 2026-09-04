"""Tests for the clinical image channel.

The vision reader is the newest input path and the one most likely to carry a
patient's name, so the properties pinned here are the refusals: no attestation,
no read; identifiers found, findings discarded; and under no circumstances a
radiology finding that claims to stand in for a report.
"""

from __future__ import annotations

import base64
import json
import struct
import unittest
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

from yaobi_harness import schemas
from yaobi_harness.llm.base import LLMResponse, ToolCall
from yaobi_harness.state import Budget, ClinicalRunState
from yaobi_harness.tools import CapabilityBroker, ToolRegistry
from yaobi_harness.vision.client import (
    IMAGE_KINDS, MAX_IMAGE_BYTES, ImageRead, VisionClient, VisionError,
    build_vision_client, decode_data_uri, describe_vision, encode_image,
)


def tiny_png() -> bytes:
    """A valid 1x1 PNG, built here so the repo carries no binary fixtures."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\xff\xff")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


class StubChat:
    """A chat client that replays scripted JSON bodies and records its prompts."""

    name = "stub"
    model = "stub-vision"
    available = True

    def __init__(self, payloads: list[dict | str]) -> None:
        self.payloads = list(payloads)
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, tools=None, temperature=0.0, max_tokens=1024, response_format_json=False):
        self.calls.append(messages)
        payload = self.payloads.pop(0) if self.payloads else {}
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return LLMResponse(text=text, provider="stub", model="stub-vision",
                           prompt_tokens=10, completion_tokens=10)


CLEAN_PHI = {"has_identifiers": False}
GOOD_READ = {
    "image_kind": "radiograph",
    "readable": True,
    "observations": ["正位腰椎，L4-L5 椎间隙略窄"],
    "not_assessable": ["翻拍照片无法评估骨小梁细节"],
    "urgent_signals": [],
    "suggest_ask": ["身高有没有变矮？"],
    "suggest_exam": ["测量身高并与年轻时最高身高比较"],
    "confidence": "low",
    "caveat": "翻拍照片，需正式阅片确认",
}


class EncodingTests(unittest.TestCase):
    def test_encode_returns_a_data_uri_and_a_digest_of_the_raw_bytes(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.png"
            path.write_bytes(tiny_png())
            uri, digest, mime = encode_image(path)
            self.assertTrue(uri.startswith("data:image/png;base64,"))
            self.assertEqual(len(digest), 64)
            self.assertEqual(mime, "image/png")
            # The digest identifies the file, not this encoding of it.
            import hashlib

            self.assertEqual(digest, hashlib.sha256(tiny_png()).hexdigest())

    def test_a_missing_file_is_an_error(self):
        with self.assertRaises(VisionError):
            encode_image("/nonexistent/nope.png")

    def test_an_unsupported_suffix_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "scan.dcm"
            path.write_bytes(b"not-an-image")
            with self.assertRaises(VisionError):
                encode_image(path)

    def test_an_empty_file_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.png"
            path.write_bytes(b"")
            with self.assertRaises(VisionError):
                encode_image(path)

    def test_data_uri_round_trip(self):
        uri = "data:image/png;base64," + base64.b64encode(tiny_png()).decode()
        back, digest = decode_data_uri(uri)
        self.assertEqual(back, uri)
        self.assertEqual(len(digest), 64)

    def test_a_non_data_uri_is_rejected(self):
        with self.assertRaises(VisionError):
            decode_data_uri("https://example.org/x.png")

    def test_bad_base64_is_rejected(self):
        with self.assertRaises(VisionError):
            decode_data_uri("data:image/png;base64,!!!not-base64!!!")

    def test_an_oversized_data_uri_is_rejected(self):
        payload = base64.b64encode(b"\x00" * (MAX_IMAGE_BYTES + 1)).decode()
        with self.assertRaises(VisionError):
            decode_data_uri("data:image/png;base64," + payload)


class VisionClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "x.png"
        self.path.write_bytes(tiny_png())

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_clean_image_is_read_into_structured_findings(self):
        client = VisionClient(StubChat([CLEAN_PHI, GOOD_READ]))
        read = client.read(self.path, kind="radiograph")
        self.assertTrue(read.readable)
        self.assertIn("L4-L5 椎间隙略窄", read.observations[0])
        self.assertFalse(read.phi_detected)

    def test_every_read_requires_a_formal_read(self):
        client = VisionClient(StubChat([CLEAN_PHI, {**GOOD_READ, "requires_formal_read": False}]))
        read = client.read(self.path, kind="radiograph")
        self.assertTrue(read.requires_formal_read)
        self.assertTrue(read.to_dict()["requires_formal_read"])
        self.assertTrue(read.to_dict()["not_a_radiology_report"])

    def test_identifiers_discard_the_findings(self):
        client = VisionClient(StubChat([
            {"has_identifiers": True, "kinds": ["burned_in_name", "hospital_id"], "note": "角标有姓名"},
            GOOD_READ,  # would be the read; must never be reached
        ]))
        read = client.read(self.path, kind="radiograph")
        self.assertTrue(read.phi_detected)
        self.assertEqual(read.observations, [])
        self.assertFalse(read.readable)
        self.assertEqual(read.image_kind, "rejected_phi")
        self.assertIn("burned_in_name", read.phi_kinds)

    def test_a_failed_phi_check_neither_blocks_nor_silently_passes_as_clean(self):
        class Flaky(StubChat):
            def chat(self, messages, **kwargs):
                if not self.calls:
                    self.calls.append(messages)
                    raise VisionError("phi check down")
                return super().chat(messages, **kwargs)

        read = VisionClient(Flaky([GOOD_READ])).read(self.path, kind="radiograph")
        self.assertFalse(read.phi_detected)
        self.assertTrue(read.readable, "a flaky pre-check must not block every image")

    def test_a_dose_in_the_output_is_stripped(self):
        client = VisionClient(StubChat([CLEAN_PHI, {
            **GOOD_READ,
            "observations": ["椎间隙变窄", "建议布洛芬 0.3g bid"],
            "suggest_exam": ["复查", "加用钙 600mg"],
        }]))
        read = client.read(self.path, kind="radiograph")
        blob = json.dumps(read.to_dict(), ensure_ascii=False)
        self.assertNotIn("0.3g", blob)
        self.assertNotIn("600mg", blob)
        self.assertIn("椎间隙变窄", read.observations)

    def test_an_unknown_kind_is_rejected(self):
        with self.assertRaises(VisionError):
            VisionClient(StubChat([CLEAN_PHI, GOOD_READ])).read(self.path, kind="ultrasound")

    def test_every_kind_has_its_own_prompt(self):
        from yaobi_harness.vision.client import READ_PROMPTS

        for kind in IMAGE_KINDS:
            self.assertIn(kind, READ_PROMPTS, f"{kind} would silently borrow another prompt")

    def test_the_image_is_sent_as_a_content_part(self):
        chat = StubChat([CLEAN_PHI, GOOD_READ])
        VisionClient(chat).read(self.path, kind="tongue", context="舌象照片")
        content = chat.calls[-1][-1]["content"]
        self.assertTrue(any(part.get("type") == "image_url" for part in content))
        self.assertTrue(any(part.get("type") == "text" for part in content))

    def test_an_unparseable_reply_is_an_error_not_an_empty_read(self):
        with self.assertRaises(VisionError):
            VisionClient(StubChat([CLEAN_PHI, "这不是 JSON"])).read(self.path, kind="radiograph")

    def test_an_unavailable_client_raises_rather_than_returning_nothing(self):
        class Off:
            name, model, available = "off", "none", False

            def chat(self, *a, **k):
                raise AssertionError("must not be called")

        with self.assertRaises(VisionError):
            VisionClient(Off()).read(self.path)

    def test_budget_exhaustion_stops_the_call(self):
        with self.assertRaises(VisionError):
            VisionClient(StubChat([CLEAN_PHI, GOOD_READ])).read(
                self.path, kind="radiograph", budget=Budget(max_llm_calls=0))

    def test_tokens_are_charged_to_the_budget(self):
        budget = Budget()
        VisionClient(StubChat([CLEAN_PHI, GOOD_READ])).read(
            self.path, kind="radiograph", budget=budget)
        self.assertGreater(budget.used_llm_tokens, 0)
        self.assertGreaterEqual(budget.used_llm_calls, 2)

    def test_describe_reports_configuration_honestly(self):
        self.assertFalse(describe_vision(None)["configured"])
        self.assertTrue(describe_vision(VisionClient(StubChat([])))["configured"])

    def test_build_returns_none_when_disabled(self):
        self.assertIsNone(build_vision_client(provider="none"))


class ImageToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "x.png"
        self.path.write_bytes(tiny_png())
        self.broker = CapabilityBroker("physician", "routine", skill_registry=None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_without_the_attestation_the_tool_refuses(self):
        tools = ToolRegistry(vision=VisionClient(StubChat([CLEAN_PHI, GOOD_READ])))
        result = tools.call(self.broker, "medical_image_read", image=str(self.path), deidentified=False)
        self.assertFalse(result.ok)
        self.assertTrue(result.recoverable)
        self.assertIn("去标识化", result.summary)

    def test_with_the_attestation_the_tool_reads(self):
        tools = ToolRegistry(vision=VisionClient(StubChat([CLEAN_PHI, GOOD_READ])))
        result = tools.call(self.broker, "medical_image_read", image=str(self.path),
                           kind="radiograph", deidentified=True)
        self.assertTrue(result.ok)
        self.assertIn("视觉所见", result.summary)

    def test_a_read_lands_in_the_ledger_as_non_releasable_evidence(self):
        from yaobi_harness.agent.agents import record_tool
        from yaobi_harness.state import NON_RELEASABLE_LEVELS

        tools = ToolRegistry(vision=VisionClient(StubChat([CLEAN_PHI, GOOD_READ])))
        result = tools.call(self.broker, "medical_image_read", image=str(self.path),
                           kind="radiograph", deidentified=True)
        state = ClinicalRunState(complaint="腰痛")
        evidence_id = record_tool(state, result)
        self.assertIn(state.evidence[evidence_id].level, NON_RELEASABLE_LEVELS)
        self.assertNotIn(evidence_id, state.releasable_evidence_ids())

    def test_a_phi_hit_is_a_successful_refusal_not_a_tool_failure(self):
        """A refusal is the tool working. Marking it failed would trip the breaker."""
        tools = ToolRegistry(vision=VisionClient(StubChat([
            {"has_identifiers": True, "kinds": ["face"]}, GOOD_READ])))
        result = tools.call(self.broker, "medical_image_read", image=str(self.path),
                           kind="limb_surface", deidentified=True)
        self.assertTrue(result.ok)
        self.assertTrue(result.data["phi_detected"])
        self.assertEqual(result.data["observations"], [])

    def test_without_a_vision_client_the_tool_reports_itself_as_a_stub(self):
        result = ToolRegistry().call(self.broker, "medical_image_read",
                                     image="x.png", deidentified=True)
        self.assertTrue(result.ok)
        self.assertTrue(result.is_stub)
        self.assertIn("how_to_fix", result.data)


class VisionAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "x.png"
        self.path.write_bytes(tiny_png())

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, payloads, complaint="左小腿肿胀2天", kind="limb_surface"):
        from yaobi_harness.graph import YaobiGraphRunner

        runner = YaobiGraphRunner(ToolRegistry(vision=VisionClient(StubChat(payloads))))
        state = ClinicalRunState(complaint=complaint, role="patient")
        state.images = [{"kind": kind, "ref": str(self.path), "deidentified": True}]
        return runner.run(state)

    def test_images_are_read_and_recorded(self):
        out = self._run([CLEAN_PHI, {**GOOD_READ, "image_kind": "limb_surface"}])
        findings = out.outputs["image_findings"]
        self.assertTrue(findings["observations"])
        self.assertTrue(findings["requires_formal_read"])

    def test_a_surface_emergency_escalates_the_run(self):
        out = self._run([CLEAN_PHI, {
            **GOOD_READ, "image_kind": "limb_surface",
            "urgent_signals": ["左小腿明显肿胀，皮色发紫，张力高"],
        }])
        self.assertEqual(out.risk_mode, "urgent")
        self.assertTrue(any("急症" in w for w in out.warnings))

    def test_a_phi_hit_becomes_a_safety_issue(self):
        out = self._run([{"has_identifiers": True, "kinds": ["burned_in_name"]}, GOOD_READ])
        self.assertTrue(any("未去标识化" in issue for issue in out.safety_issues))

    def test_an_unattested_image_does_not_stop_the_run(self):
        from yaobi_harness.graph import YaobiGraphRunner

        runner = YaobiGraphRunner(ToolRegistry(vision=VisionClient(StubChat([CLEAN_PHI, GOOD_READ]))))
        state = ClinicalRunState(complaint="腰痛3个月", role="patient")
        state.images = [{"kind": "radiograph", "ref": str(self.path), "deidentified": False}]
        out = runner.run(state)
        self.assertNotEqual(out.release_status, "failed_closed")
        self.assertTrue(any("判读未完成" in w for w in out.warnings))

    def test_no_images_means_no_vision_node_output(self):
        from yaobi_harness.graph import YaobiGraphRunner

        out = YaobiGraphRunner().run(ClinicalRunState(complaint="腰痛3个月", role="patient"))
        self.assertNotIn("image_findings", out.outputs)


class ImageSchemaTests(unittest.TestCase):
    def test_a_radiology_read_may_not_claim_to_be_a_report(self):
        ok, problems = schemas.validate("ImageFindings", {
            "image_kind": "radiograph", "readable": True,
            "observations": ["x"], "requires_formal_read": False,
        })
        self.assertFalse(ok)
        self.assertTrue(any("正式阅片" in p for p in problems))

    def test_a_tongue_read_is_not_held_to_the_radiology_rule(self):
        ok, _ = schemas.validate("ImageFindings", {
            "image_kind": "tongue", "readable": True,
            "observations": ["舌淡红"], "requires_formal_read": True,
        })
        self.assertTrue(ok)

    def test_missing_observations_fail_the_contract(self):
        ok, _ = schemas.validate("ImageFindings", {
            "image_kind": "tongue", "readable": True, "requires_formal_read": True})
        self.assertFalse(ok)


class ImageReadShapeTests(unittest.TestCase):
    def test_the_dict_always_carries_the_two_disclaimers(self):
        payload = ImageRead("abc", "radiograph").to_dict()
        self.assertTrue(payload["requires_formal_read"])
        self.assertTrue(payload["not_a_radiology_report"])

    def test_bytes_are_never_carried_in_the_result(self):
        payload = ImageRead("abc", "radiograph", observations=["x"]).to_dict()
        blob = json.dumps(payload)
        self.assertNotIn("data:image", blob)
        self.assertNotIn("base64", blob)


if __name__ == "__main__":
    unittest.main()


class ModelRequestsAnImageTests(unittest.TestCase):
    """The model decides when a picture would change its judgement, and asks.

    Two things it may not do, and both are about the attestation rather than the
    picture: it cannot tick 「已去标识化」 on the uploader's behalf (a model
    asserting that about a file it has never seen would make the attestation
    meaningless), and it cannot request an upload nothing can read.
    """

    def loop(self, calls, *, vision: bool = True):
        from yaobi_harness.interview.adequacy import AdequacyJudge
        from yaobi_harness.interview.loop import InterviewLoop

        class Stub:
            name, model, available = "stub", "stub", True

            def chat(self, messages, tools=None, **kwargs):
                self.saw_tools = [t.name for t in (tools or [])]
                return LLMResponse(tool_calls=calls)

        model = Stub()
        loop = InterviewLoop(model, judge=AdequacyJudge())
        loop.vision_available = vision
        loop.model = model
        return loop

    def test_the_request_tool_is_offered_alongside_asking(self):
        from yaobi_harness.interview.loop import ASK_TOOL, IMAGE_TOOL

        loop = self.loop([ToolCall("ask_patient", {"questions": []}, "c1")])
        loop.next_round({}, "腰痛", budget=Budget())
        self.assertEqual(set(loop.model.saw_tools), {ASK_TOOL.name, IMAGE_TOOL.name})

    def test_a_request_riding_along_with_questions(self):
        """Many gateways surface one tool call per turn, so asking and requesting
        must be possible in the same call."""
        loop = self.loop([ToolCall("ask_patient", {
            "questions": [{"axis_id": "four_diagnoses", "question": "方便拍张舌头照片吗？"}],
            "image_requests": [{"kind": "tongue", "why": "舌象决定辨证方向"}]}, "c1")])
        result = loop.next_round({}, "腰痛", budget=Budget())
        self.assertEqual([q.question for q in result.questions], ["方便拍张舌头照片吗？"])
        self.assertEqual(result.image_requests[0].kind, "tongue")
        self.assertEqual(result.image_requests[0].why, "舌象决定辨证方向")

    def test_a_standalone_request_is_a_complete_round(self):
        """Asking for a picture and nothing else is an action, not an empty turn."""
        loop = self.loop([ToolCall("request_image",
                                   {"kind": "limb_surface", "why": "看小腿是否发紫肿胀"}, "c1")])
        result = loop.next_round({}, "小腿肿胀2天", budget=Budget())
        self.assertEqual(result.image_requests[0].kind, "limb_surface")
        self.assertEqual(result.questions, [])
        self.assertNotEqual(result.composer, "llm_complete",
                            "requesting an image is not deciding to stop")
        self.assertTrue(any("只请求了图片" in n for n in result.notes))

    def test_an_unknown_kind_becomes_other_rather_than_being_dropped(self):
        loop = self.loop([ToolCall("request_image", {"kind": "ultrasound", "why": "x"}, "c1")])
        result = loop.next_round({}, "腰痛", budget=Budget())
        self.assertEqual(result.image_requests[0].kind, "other")
        self.assertTrue(any("不在支持列表" in n for n in result.notes))

    def test_no_request_survives_when_no_vision_model_is_configured(self):
        """An upload nothing will look at wastes the patient's effort and trust."""
        loop = self.loop([ToolCall("request_image", {"kind": "tongue", "why": "x"}, "c1")],
                         vision=False)
        result = loop.next_round({}, "腰痛", budget=Budget())
        self.assertEqual(result.image_requests, [])
        self.assertTrue(any("未配置视觉模型" in n for n in result.notes))

    def test_the_model_is_told_whether_vision_is_available(self):
        import json as _json

        loop = self.loop([ToolCall("ask_patient", {"questions": []}, "c1")], vision=False)
        captured = []

        original = loop.llm.chat

        def spy(messages, **kwargs):
            captured.append(messages[-1]["content"])
            return original(messages, **kwargs)

        loop.llm.chat = spy
        loop.next_round({}, "腰痛", budget=Budget())
        self.assertIs(_json.loads(captured[0])["vision_available"], False)

    def test_an_attached_image_is_reported_back_to_the_model(self):
        """So it can see it has the tongue photo and ask for the radiograph instead."""
        import json as _json

        from yaobi_harness.conversation import ConversationSession

        convo = ConversationSession(role="patient")
        convo.attach_image("data:image/png;base64,AAAA", kind="tongue", deidentified=True)
        self.assertEqual(convo.interview.attached_image_kinds, ["tongue"])
        _json.dumps(convo.interview.attached_image_kinds)

    def test_the_request_reaches_the_reply(self):
        from yaobi_harness.conversation import ConversationSession
        from yaobi_harness.graph import YaobiGraphRunner

        class Asker:
            name, model, available = "asker", "asker", True

            def chat(self, messages, tools=None, **kwargs):
                if "问诊智能体" in messages[0]["content"]:
                    return LLMResponse(tool_calls=[ToolCall("request_image", {
                        "kind": "tongue", "why": "舌象决定辨证"}, "c1")])
                return LLMResponse(text="{}")

        runner = YaobiGraphRunner(llm=Asker())
        convo = ConversationSession(role="patient", runner=runner)
        convo.interview.vision_available = True
        reply = convo.send("腰痛3个月，久坐加重")
        self.assertEqual(reply.image_requests[0]["kind"], "tongue")
        self.assertNotIn("deidentified", reply.image_requests[0],
                         "the attestation is the uploader's, never the model's")


class VisionFallbackTests(unittest.TestCase):
    """Where 「上传 X 片无法自动解析」 actually came from.

    ``build_vision_client`` looks for a *separate* vision provider, defaulting to
    Poe. Start the console with an OpenAI-compatible endpoint and no
    ``YAOBI_VISION_*`` and that lookup finds nothing: image reading is off, the
    tool returns a cheerful ``ok=True`` stub, and the film is accepted, stored and
    never looked at.

    Every endpoint this harness speaks to is OpenAI-shaped and takes an
    ``image_url`` content part, so a multimodal chat model *is* a working vision
    model. Borrowing it is right; guessing "probably not multimodal" and staying
    dark is how the silence happened.
    """

    class Chat:
        name, model, available = "openai-compatible", "some-multimodal-1", True

        def chat(self, messages, **kwargs):
            return LLMResponse(text="{}")

    def setUp(self):
        import os

        self._saved = {k: os.environ.pop(k, None)
                       for k in ("YAOBI_VISION_PROVIDER", "YAOBI_VISION_MODEL",
                                 "POE_API_KEY", "POE_VISION_MODEL")}

    def tearDown(self):
        import os

        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def service_with(self, client):
        from yaobi_harness.ui.server import ConsoleService, _CountingLLM

        service = ConsoleService.__new__(ConsoleService)
        service.llm = _CountingLLM(client)
        return service

    def test_the_chat_model_is_borrowed_when_no_vision_provider_is_set(self):
        service = self.service_with(self.Chat())
        vision = service._open_vision(True)
        self.assertIsNotNone(vision, "an available chat model must not leave vision dark")
        self.assertTrue(vision.available)
        self.assertTrue(vision.borrowed)
        self.assertEqual(vision.model, "some-multimodal-1")

    def test_the_borrowing_is_announced_rather_than_hidden(self):
        """If that model turns out not to be multimodal, this is the line that
        explains the failure."""
        from yaobi_harness.vision.client import describe_vision

        described = describe_vision(self.service_with(self.Chat())._open_vision(True))
        self.assertTrue(described["configured"])
        self.assertTrue(described["borrowed_chat_model"])

    def test_an_explicitly_configured_provider_that_fails_is_not_papered_over(self):
        """A stated intention that did not work is a configuration error to
        report, not a gap to quietly fill with something else."""
        import os

        os.environ["YAOBI_VISION_PROVIDER"] = "poe"      # set, but no POE_API_KEY
        self.assertIsNone(self.service_with(self.Chat())._open_vision(True))

    def test_no_chat_model_means_no_vision_either(self):
        class Absent:
            name, model, available = "none", "none", False

            def chat(self, messages, **kwargs):
                raise AssertionError("must not be called")

        self.assertIsNone(self.service_with(Absent())._open_vision(True))

    def test_vision_stays_off_when_it_was_switched_off(self):
        self.assertIsNone(self.service_with(self.Chat())._open_vision(False))


class ImageIsReadOncePerConversationTests(unittest.TestCase):
    """A film attached on turn two must not be re-read on turns three and four.

    Every turn is a fresh audited run over the accumulated case — that is what
    makes a red flag disclosed on turn three get screened on turn three. Applied
    to images it meant the same X-ray went to a paid multimodal endpoint on every
    subsequent turn, forever, for content that cannot change.
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
            return ImageRead(image_sha256="abc", image_kind=kind, readable=True,
                             observations=["骨皮质连续，未见明确骨折线"], model="v")

    class Stub:
        name, model, available = "s", "s", True

        def chat(self, messages, **kwargs):
            return LLMResponse(text=json.dumps({
                "triage": "routine", "adequate": False, "workup_now": False,
                "questions": [], "facts": {}, "message": "好的", "reply": "好的",
            }, ensure_ascii=False))

    def _session(self):
        from yaobi_harness.conversation import ConversationSession
        from yaobi_harness.graph import YaobiGraphRunner

        vision = self.CountingVision()
        runner = YaobiGraphRunner(tools=ToolRegistry(vision=vision), llm=self.Stub())
        return ConversationSession(role="patient", runner=runner), vision

    def test_one_attachment_costs_exactly_one_vision_call(self):
        session, vision = self._session()
        session.send("腰痛3个月")
        session.attach_image("data:image/png;base64,AA==", kind="radiograph", deidentified=True)
        for message in ("这是我的片子", "还有别的要问吗", "我不知道", "嗯"):
            session.send(message)
        self.assertEqual(vision.reads, 1, "the film was re-read on later turns")

    def test_the_finding_still_appears_in_every_later_turns_ledger(self):
        """A citation has to resolve inside the run that made it, so the carried
        finding is recorded again — labelled as carried, not as a fresh read."""
        session, _ = self._session()
        session.attach_image("data:image/png;base64,AA==", kind="radiograph", deidentified=True)
        session.send("这是我的片子")
        session.send("还有呢")
        entries = [e for e in session.state.evidence.values() if e.source == "medical_image_read"]
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0].payload.get("carried_forward"))
        findings = session.state.outputs["image_findings"]
        self.assertIn("骨皮质连续，未见明确骨折线", findings["observations"])

    def test_a_second_image_is_read_when_it_is_attached(self):
        session, vision = self._session()
        session.attach_image("data:image/png;base64,AA==", kind="radiograph", deidentified=True)
        session.send("这是片子")
        session.attach_image("data:image/png;base64,BB==", kind="tongue", deidentified=True)
        session.send("这是舌象")
        self.assertEqual(vision.reads, 2)

    def test_the_same_bytes_as_a_different_kind_are_a_different_attachment(self):
        """A tongue reading and a radiograph reading are different tasks with
        different prompts, so sharing one cached finding would be wrong."""
        session, vision = self._session()
        session.attach_image("data:image/png;base64,AA==", kind="radiograph", deidentified=True)
        session.attach_image("data:image/png;base64,AA==", kind="tongue", deidentified=True)
        session.send("看看这两张")
        self.assertEqual(vision.reads, 2)


class VisionRunsBeforeTheInterviewTests(unittest.TestCase):
    """The film is read before the questions are composed, not after.

    ``VisionAgent`` depends only on intake, and its ``suggest_ask`` exists to
    steer the questioning. Scheduled after ``InterviewAgent`` — where it sat —
    the finding arrived one full turn late: the model composed its questions
    blind to a film it already had, and the patient answered a round of
    questions the image had already answered.
    """

    def _plan_agents(self, **state_kwargs):
        from yaobi_harness.agent.planner import rule_plan, validate_plan

        state = ClinicalRunState(complaint="腰痛3月", role="physician", **state_kwargs)
        tasks = rule_plan(state)
        ok, problems = validate_plan(tasks, state)
        self.assertTrue(ok, problems)
        return [t.agent for t in tasks]

    def test_vision_is_planned_before_the_interview(self):
        agents = self._plan_agents(images=[{"kind": "radiograph", "ref": "data:image/png;base64,AA=="}])
        self.assertIn("VisionAgent", agents)
        self.assertLess(agents.index("VisionAgent"), agents.index("InterviewAgent"))

    def test_no_images_means_no_vision_task(self):
        self.assertNotIn("VisionAgent", self._plan_agents())

    def test_the_interview_hands_the_findings_to_the_composer(self):
        """The compose payload carries what the film showed, so this round's
        questions can chase it."""
        from yaobi_harness.interview.loop import InterviewLoop

        seen: list[str] = []

        class Recorder:
            name, model, available = "r", "r", True

            def chat(self, messages, **kwargs):
                seen.append(json.dumps(messages, ensure_ascii=False))
                return LLMResponse(text=json.dumps(
                    {"adequate": False, "questions": [], "reason": "看过片子了"},
                    ensure_ascii=False))

        loop = InterviewLoop(Recorder())
        loop.next_round(
            {}, "腰痛3月", role="patient", budget=Budget(),
            image_findings={
                "observations": ["L1椎体上缘骨皮质可疑中断"],
                "suggest_ask": ["近期有没有摔倒或搬重物？"],
                "urgent_signals": [],
            },
        )
        joined = "\n".join(seen)
        self.assertIn("L1椎体上缘骨皮质可疑中断", joined)
        self.assertIn("近期有没有摔倒或搬重物", joined)

    def test_a_run_without_findings_sends_an_empty_block_not_a_crash(self):
        from yaobi_harness.interview.loop import InterviewLoop

        class Recorder:
            name, model, available = "r", "r", True

            def chat(self, messages, **kwargs):
                return LLMResponse(text=json.dumps({"adequate": False, "questions": []}))

        result = InterviewLoop(Recorder()).next_round({}, "腰痛3月", budget=Budget())
        self.assertIsNotNone(result.verdict)


class AttachmentIdempotenceTests(unittest.TestCase):
    """Re-attaching the same image is one attachment, not six.

    The page sends each handle once, but a retried turn or a naive API caller
    re-sends what it has. Without the guard the duplicates crowd real images out
    of the per-run cap and repeat every finding in the aggregate output.
    """

    def _session(self):
        from yaobi_harness.conversation import ConversationSession
        from yaobi_harness.graph import YaobiGraphRunner

        return ConversationSession(role="patient", runner=YaobiGraphRunner(tools=ToolRegistry()))

    def test_the_same_ref_and_kind_attach_once(self):
        session = self._session()
        first = session.attach_image("data:image/png;base64,AA==", kind="radiograph", deidentified=True)
        second = session.attach_image("data:image/png;base64,AA==", kind="radiograph", deidentified=True)
        self.assertEqual(len(session.images), 1)
        self.assertIs(first, second)

    def test_a_different_kind_is_a_different_attachment(self):
        session = self._session()
        session.attach_image("data:image/png;base64,AA==", kind="radiograph", deidentified=True)
        session.attach_image("data:image/png;base64,AA==", kind="tongue", deidentified=True)
        self.assertEqual(len(session.images), 2)


class SeededPrereadTests(unittest.TestCase):
    """A finding obtained at upload time is adopted, and the turn pays nothing.

    The earliest the vision model can be activated is the upload itself — the
    read runs while the user is still typing. What the session needs is a way to
    adopt that finding as if a turn had produced it.
    """

    class Quiet:
        name, model, available = "s", "s", True

        def chat(self, messages, **kwargs):
            return LLMResponse(text=json.dumps({
                "triage": "routine", "adequate": False, "workup_now": False,
                "questions": [], "facts": {}, "message": "好的", "reply": "好的",
            }, ensure_ascii=False))

    def _session(self):
        from tests.test_vision import ImageIsReadOncePerConversationTests as Base
        from yaobi_harness.conversation import ConversationSession
        from yaobi_harness.graph import YaobiGraphRunner

        vision = Base.CountingVision()
        runner = YaobiGraphRunner(tools=ToolRegistry(vision=vision), llm=self.Quiet())
        return ConversationSession(role="patient", runner=runner), vision

    def test_a_seeded_read_means_the_turn_makes_no_vision_call(self):
        session, vision = self._session()
        entry = session.attach_image("data:image/png;base64,AA==", kind="radiograph", deidentified=True)
        session.seed_image_read(entry["image_id"], {
            "image_kind": "radiograph", "readable": True,
            "observations": ["上传时已判读：骨皮质连续"], "requires_formal_read": True,
        })
        session.send("这是我的片子")
        self.assertEqual(vision.reads, 0, "the turn re-read a film that was pre-read at upload")
        findings = session.state.outputs["image_findings"]
        self.assertIn("上传时已判读：骨皮质连续", findings["observations"])

    def test_a_seed_never_overwrites_what_a_run_recorded(self):
        session, _ = self._session()
        entry = session.attach_image("data:image/png;base64,AA==", kind="radiograph", deidentified=True)
        session.send("看片子")
        recorded = dict(session.image_reads[entry["image_id"]])
        session.seed_image_read(entry["image_id"], {"observations": ["别的东西"]})
        self.assertEqual(session.image_reads[entry["image_id"]], recorded)

    def test_an_empty_seed_is_ignored(self):
        session, _ = self._session()
        session.seed_image_read("radiograph:abc", {})
        session.seed_image_read("", {"observations": ["x"]})
        self.assertEqual(session.image_reads, {})

    def test_a_cached_phi_rejection_keeps_warning_on_later_turns(self):
        """A PHI rejection is cached like any read; without a warning on replay
        every later turn reported findings present while the film stayed unread."""
        session, vision = self._session()
        entry = session.attach_image("data:image/png;base64,AA==", kind="radiograph", deidentified=True)
        session.seed_image_read(entry["image_id"], {
            "image_kind": "rejected_phi", "readable": False, "phi_detected": True,
            "phi_kinds": ["burned_in_name"], "observations": [],
        })
        session.send("帮我看看")
        self.assertEqual(vision.reads, 0)
        self.assertTrue(any("拒绝判读" in w for w in session.state.warnings),
                        session.state.warnings)
