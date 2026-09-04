"""Tests for the structured clinical note.

A generated medical record is only useful if a clinician trusts it, and the two
ways to lose that trust are opposite: inventing a finding nobody established, and
producing something that reads as machine output (`治法：['补益肝肾', '活血通络']`)
so it gets ignored. Both are pinned here.

The third property is the one that matters legally: a note built from a run that
never reached a physician's signature says so, in its structure rather than in a
footnote, and it carries a gram value only if the deterministic dose pipeline
produced one.
"""

from __future__ import annotations

import json
import os
import unittest

os.environ.setdefault("YAOBI_DEID_KEY", "unit-test-fixed-key")

from yaobi_harness.graph import YaobiGraphRunner
from yaobi_harness.llm.base import LLMResponse
from yaobi_harness.state import ClinicalRunState
from yaobi_harness.summary import (
    NOT_COLLECTED, SECTIONS, build_note, is_concluded, redact_doses,
)
from yaobi_harness.tools import ToolRegistry

TEST_KEY = "unit-test-fixed-key"
CORE = ["独活", "桑寄生", "杜仲", "牛膝", "当归", "川芎"]
#: Every herb the candidate formula may reach. A range must exist for all of them
#: or the dose gate stops at ``treatment_advice_only``, which is correct behaviour
#: and the wrong fixture for testing what a prescription section looks like.
FORMULA_HERBS = CORE + ["白芍", "熟地黄", "党参", "茯苓", "甘草", "桃仁", "红花", "延胡索"]

PHYSICIAN_FACTS = {
    "special_population": {"pregnancy": False, "age": 63, "renal": "normal", "liver": "normal"},
    "medications_confirmed": True, "allergies_confirmed": True,
    "onset": "3个月", "pain_location": "右侧腰部", "radiation": "右下肢后侧",
    "neuro_symptoms": "右小腿外侧发麻", "bowel_bladder": "否认",
    "fever_trauma_tumor": "否认", "night_pain": "否认",
    "four_diagnoses": "舌暗有瘀点，苔薄白，脉弦", "vas": 6, "odi": 32,
}


def corpus(n: int = 12) -> list[dict]:
    body = "".join(f",{j}/{h}*1克/9克/用法：无/贴数:7\n" for j, h in enumerate(FORMULA_HERBS, 1))
    return [
        {"病案号": f"P{i}", "姓名": "张三", "性别": "男", "年龄": "60岁", "主诉": "腰痛",
         "中医诊断": "腰痹/证型：气滞血瘀证", "治疗方法": "中药内服", "中药": body}
        for i in range(n)
    ]


def concluded_run(*, allow_prescription: bool = False, llm=None) -> ClinicalRunState:
    tools = ToolRegistry(records=corpus(), deid_key=TEST_KEY,
                         authorized_ranges={h: (3.0, 15.0) for h in FORMULA_HERBS})
    state = ClinicalRunState("腰痛3月，久坐加重，右下肢麻木", role="physician")
    state.facts.update(PHYSICIAN_FACTS)
    return YaobiGraphRunner(tools, llm=llm).run(state, allow_prescription=allow_prescription)


class WhenANoteIsProducedTests(unittest.TestCase):
    def test_an_unfinished_consultation_gets_no_note(self):
        """A note over an incomplete history is a misleading document."""
        state = ClinicalRunState("腰痛", role="patient")
        out = YaobiGraphRunner().run(state)
        self.assertEqual(out.release_status, "needs_more_information")
        self.assertFalse(is_concluded(out))
        self.assertIsNone(out.outputs.get("clinical_note"))

    def test_a_concluded_run_gets_one(self):
        out = concluded_run()
        self.assertTrue(is_concluded(out))
        self.assertIsNotNone(out.outputs.get("clinical_note"))

    def test_an_emergency_gets_one_too(self):
        state = ClinicalRunState("突然不能排尿、会阴麻木，双腿越来越无力", role="patient")
        out = YaobiGraphRunner().run(state)
        self.assertEqual(out.release_status, "urgent_action_plan")
        note = out.outputs["clinical_note"]
        sections = {s["key"]: s["text"] for s in note["sections"]}
        self.assertIn("立即处置", sections["treatment_plan"])

    def test_a_failed_run_still_produces_no_note_and_does_not_crash(self):
        class Broken(ToolRegistry):
            def call(self, *a, **k):
                raise RuntimeError("tool layer down")

        out = YaobiGraphRunner(Broken()).run(ClinicalRunState("腰痛", role="patient"))
        self.assertEqual(out.release_status, "failed_closed")
        self.assertIsNone(out.outputs.get("clinical_note"))


class StructureTests(unittest.TestCase):
    def setUp(self):
        self.out = concluded_run()
        self.note = self.out.outputs["clinical_note"]

    def test_every_section_is_present_and_ordered(self):
        keys = [s["key"] for s in self.note["sections"]]
        self.assertEqual(keys, [key for key, _ in SECTIONS])

    def test_an_unfilled_section_says_so_rather_than_vanishing(self):
        """A record that silently drops 既往史 reads as "nothing relevant", which is
        a different clinical claim from "not asked"."""
        texts = {s["key"]: s["text"] for s in self.note["sections"]}
        self.assertTrue(all(text for text in texts.values()))
        self.assertIn(NOT_COLLECTED, texts.values())

    def test_the_history_actually_reaches_the_note(self):
        texts = {s["key"]: s["text"] for s in self.note["sections"]}
        self.assertIn("右下肢后侧", texts["present_illness"])
        self.assertIn("舌暗有瘀点", texts["four_diagnoses"])
        self.assertIn("VAS", texts["examination"])
        self.assertIn("63", texts["past_history"])

    def test_the_chief_complaint_is_one_line(self):
        """Pasting six turns of dialogue into 主诉 makes the note unreadable."""
        state = ClinicalRunState("x", role="physician")
        state.facts.update(PHYSICIAN_FACTS)
        state.release_status = "treatment_advice_only"
        note = build_note(state, narrative=["腰痛3个月", "63岁没怀孕", "在吃布洛芬"])
        self.assertEqual(note.sections["chief_complaint"], "腰痛3个月")
        self.assertIn("患者自述补充", note.sections["present_illness"])

    def test_the_note_is_json_serialisable(self):
        json.dumps(self.note, ensure_ascii=False)

    def test_the_text_rendering_carries_every_heading(self):
        text = self.note["text"]
        for _, title in SECTIONS:
            self.assertIn(f"■ {title}", text)
        self.assertIn(self.note["disclaimer"], text)


class NoMachineOutputTests(unittest.TestCase):
    """Lists must read as prose. A Python repr in a record gets the record ignored."""

    def setUp(self):
        # With a prescription, so the sections most prone to leaking a repr —
        # 治法, 组成, 专家经验 — are actually populated.
        self.sections = {
            s["key"]: s["text"]
            for s in concluded_run(allow_prescription=True).outputs["clinical_note"]["sections"]
        }

    def test_no_python_list_repr_anywhere(self):
        for key, text in self.sections.items():
            with self.subTest(section=key):
                self.assertNotIn("['", text)
                self.assertNotIn("']", text)

    def test_no_python_dict_repr_anywhere(self):
        for key, text in self.sections.items():
            with self.subTest(section=key):
                self.assertNotIn("': '", text)
                self.assertNotIn("{'", text)

    def test_the_treatment_principle_reads_as_prose(self):
        self.assertIn("治法：", self.sections["treatment_plan"])
        self.assertNotIn("[", self.sections["treatment_plan"])

    def test_the_expert_reference_is_a_line_not_a_profile_dump(self):
        """The mined profile is a page of nested data; a record wants two numbers."""
        tcm = self.sections["tcm_diagnosis"]
        self.assertIn("专家经验参照", tcm)
        self.assertIn("例", tcm)
        self.assertLess(len(tcm), 400, tcm)

    def test_a_differential_list_does_not_repeat_the_primary_pattern(self):
        tcm = self.sections["tcm_diagnosis"]
        if "待鉴别证型" in tcm:
            primary = tcm.split("证型：", 1)[1].split("；", 1)[0]
            self.assertNotIn(primary, tcm.split("待鉴别证型：", 1)[1].split("；", 1)[0])


class DoseHandlingTests(unittest.TestCase):
    def test_no_gram_value_appears_when_the_run_produced_none(self):
        sections = {s["key"]: s["text"] for s in concluded_run().outputs["clinical_note"]["sections"]}
        self.assertEqual(sections["prescription"], NOT_COLLECTED)
        self.assertNotIn("克", sections["treatment_plan"])
        self.assertNotIn("g（", sections["treatment_plan"])

    def test_a_draft_is_reproduced_with_its_signature_status(self):
        out = concluded_run(allow_prescription=True)
        self.assertEqual(out.release_status, "draft_for_physician")
        note = out.outputs["clinical_note"]
        sections = {s["key"]: s["text"] for s in note["sections"]}
        self.assertIn("未签名", sections["prescription"])
        self.assertIn("授权范围", sections["prescription"])
        self.assertIn("草案 · 未经医师签名", note["status_label"])
        self.assertIn("未经逐味审核签名前不得作为处方使用", note["disclaimer"])
        self.assertEqual(note["prescription"]["formula_name"],
                         out.outputs["prescription_draft"]["formula_name"])

    def test_the_composition_section_carries_names_without_doses(self):
        out = concluded_run(allow_prescription=True)
        sections = {s["key"]: s["text"] for s in out.outputs["clinical_note"]["sections"]}
        self.assertIn("组成（不含剂量）", sections["treatment_plan"])
        self.assertIn("独活", sections["treatment_plan"])
        self.assertNotIn("9g", sections["treatment_plan"])

    def test_model_narrative_has_its_doses_redacted(self):
        """Applied to what a model wrote, never to the prescription section itself —
        those numbers came from the deterministic pipeline and carry a status."""
        self.assertNotIn("9克", redact_doses("建议独活 9克、桑寄生 15克"))
        self.assertIn("独活", redact_doses("建议独活 9克"))


class ModelAuthoredNoteTests(unittest.TestCase):
    SECTIONS_REPLY = {
        "chief_complaint": "63岁男性，腰痛3月伴右下肢麻木",
        "present_illness": "3个月前久坐后起病，逐渐加重，近期出现右小腿外侧麻木",
        "western_diagnosis": "考虑腰椎间盘突出伴神经根病，需与腰椎管狭窄鉴别",
        "treatment_plan": "先完成神经定位体检；用药方案取决于影像结果",
        "uncertainty": "线上问诊未查体，部分必答项未闭合",
    }

    def model(self, payload):
        class Stub:
            name, model, available = "stub", "stub", True

            def chat(self, messages, tools=None, **kwargs):
                if "clinical_summary" in messages[0]["content"]:
                    return LLMResponse(text=json.dumps(payload, ensure_ascii=False))
                return LLMResponse(text="{}")

        return Stub()

    def test_the_model_writes_the_narrative_sections(self):
        out = concluded_run(llm=self.model(self.SECTIONS_REPLY))
        note = out.outputs["clinical_note"]
        self.assertEqual(note["composed_by"], "llm")
        sections = {s["key"]: s["text"] for s in note["sections"]}
        self.assertEqual(sections["chief_complaint"], self.SECTIONS_REPLY["chief_complaint"])
        self.assertIn("需与腰椎管狭窄鉴别", sections["western_diagnosis"])

    def test_a_section_the_model_left_blank_keeps_what_the_run_knew(self):
        """An empty string from the model must not erase an established finding."""
        payload = {**self.SECTIONS_REPLY, "four_diagnoses": "", "examination": "   "}
        out = concluded_run(llm=self.model(payload))
        sections = {s["key"]: s["text"] for s in out.outputs["clinical_note"]["sections"]}
        self.assertIn("舌暗有瘀点", sections["four_diagnoses"])
        self.assertIn("VAS", sections["examination"])

    def test_a_dose_the_model_writes_is_redacted(self):
        payload = {**self.SECTIONS_REPLY, "treatment_plan": "予独活寄生汤，独活 9克起"}
        out = concluded_run(allow_prescription=True, llm=self.model(payload))
        sections = {s["key"]: s["text"] for s in out.outputs["clinical_note"]["sections"]}
        self.assertNotIn("9克", sections["treatment_plan"])
        self.assertIn("独活寄生汤", sections["treatment_plan"])

    def test_a_model_failure_falls_back_to_the_deterministic_note(self):
        class Broken:
            name, model, available = "broken", "broken", True

            def chat(self, *a, **k):
                raise RuntimeError("upstream 502")

        out = concluded_run(llm=Broken())
        note = out.outputs["clinical_note"]
        self.assertEqual(note["composed_by"], "rule")
        self.assertTrue(note["text"])

    def test_the_prompt_carries_no_run_id(self):
        """A per-run identifier in a prompt makes the call unreplayable: request
        hashes cover content, so the same decision would hash differently each time."""
        seen = []

        class Recorder:
            name, model, available = "rec", "rec", True

            def chat(self, messages, tools=None, **kwargs):
                seen.append(messages[-1]["content"])
                return LLMResponse(text="{}")

        out = concluded_run(llm=Recorder())
        summary_prompts = [p for p in seen if "deterministic_note" in p]
        self.assertTrue(summary_prompts)
        self.assertNotIn(out.run_id, summary_prompts[0])


class DialogueCostTests(unittest.TestCase):
    """The authored note costs a model call, so a dialogue pays for it once."""

    def session(self, calls: list):
        from yaobi_harness.conversation import ConversationSession

        class Counter:
            name, model, available = "counter", "counter", True

            def chat(self, messages, tools=None, **kwargs):
                if "clinical_summary" in messages[0]["content"]:
                    calls.append(messages[0]["content"])
                    return LLMResponse(text=json.dumps({
                        "chief_complaint": "x", "present_illness": "y",
                        "western_diagnosis": "z", "treatment_plan": "w",
                        "uncertainty": "v"}, ensure_ascii=False))
                return LLMResponse(text="{}")

        tools = ToolRegistry(records=corpus(), deid_key=TEST_KEY,
                             authorized_ranges={h: (3.0, 15.0) for h in FORMULA_HERBS})
        return ConversationSession(role="physician",
                                   runner=YaobiGraphRunner(tools, llm=Counter()))

    def test_a_mid_dialogue_turn_gets_the_deterministic_note_only(self):
        calls: list[str] = []
        convo = self.session(calls)
        reply = convo.send("腰痛3个月，久坐加重")
        self.assertTrue(reply.questions, "the interview is still asking")
        note = convo.clinical_note()
        if note is not None:
            self.assertEqual(note["composed_by"], "rule")
        self.assertEqual(calls, [], "no model call while the enquiry is still open")

    def test_a_single_shot_run_is_authored_immediately(self):
        """There is no next turn to wait for, so deferring would mean never."""
        calls: list[str] = []
        out = concluded_run(llm=self.session(calls).llm)
        self.assertEqual(out.outputs["clinical_note"]["composed_by"], "llm")
        self.assertEqual(len(calls), 1)


class RoleScopeTests(unittest.TestCase):
    def test_the_note_reaches_the_physician_view(self):
        from yaobi_harness.render import render

        out = concluded_run()
        self.assertIsNotNone(render(out, "physician")["clinical_note"])

    def test_a_patient_view_does_not_carry_the_operator_note(self):
        """The note is a clinician's artefact: it names evidence ids, the reviewer's
        disagreements and unclosed axes. A patient gets the reply, not the record."""
        from yaobi_harness.render import render

        out = concluded_run()
        self.assertNotIn("clinical_note", render(out, "patient"))


if __name__ == "__main__":
    unittest.main()
