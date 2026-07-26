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
from yaobi_harness.llm.base import LLMResponse
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
