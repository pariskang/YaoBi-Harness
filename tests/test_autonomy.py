"""Tests for model-driven execution and the expert-corpus skill.

Two questions are being pinned here:

* **Is execution actually autonomous?** The model must choose the tools and the
  arguments, self-correct, and have its answer bound to the evidence it
  gathered — not merely have a plan generated for it.
* **Is it still contained?** Tool visibility filtered by skill, output validated
  against a schema, no dose may ever leave a model, and any failure falls back
  to the deterministic body rather than degrading silently.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("YAOBI_DEID_KEY", "unit-test-fixed-key")

import yaml

from yaobi_harness.agent.toolloop import ToolLoop, schema_hint
from yaobi_harness.expert.profile import build_profile, extract_pattern
from yaobi_harness.expert.skillgen import (
    SKILL_ID, audit_for_phi, build_instructions, build_skill_entry, merge_into_manifest, write_skill_file,
)
from yaobi_harness.graph import YaobiGraphRunner
from yaobi_harness.llm.base import LLMResponse, ToolCall
from yaobi_harness.skills.loader import SkillRegistry
from yaobi_harness.state import Budget, ClinicalRunState
from yaobi_harness.tools import CapabilityBroker, ExpertCaseStore, ToolRegistry

TEST_KEY = "unit-test-fixed-key"
MANIFEST = Path(__file__).resolve().parents[1] / "yaobi_harness" / "skills" / "manifest.yaml"
CORE = ["独活", "桑寄生", "杜仲", "牛膝", "当归", "川芎"]


def corpus(n: int = 12) -> list[dict]:
    rows = []
    for i in range(n):
        stasis = i % 2 == 0
        herbs = CORE + (["桃仁", "红花"] if stasis else ["茯苓", "甘草"])
        body = "".join(f",{j}/{h}*1克/9克/用法：无/贴数:7\n" for j, h in enumerate(herbs, 1))
        rows.append({
            "病案号": f"P{i // 3}", "姓名": "张三", "地址": "某区某街道",
            "性别": "男" if i % 2 else "女", "年龄": f"{55 + i}岁",
            "主诉": "腰痛伴下肢放射痛", "现病史": "久坐后加重" + ("，症状加重" if i % 5 == 0 else ""),
            "既往史": "高血压" if i % 3 == 0 else "无",
            "中医诊断": f"腰痹/证型：{'气滞血瘀证' if stasis else '气血痹阻证'}",
            "治疗方法": "中药内服、针灸", "西药": "塞来昔布", "辅助检查": "腰椎MRI",
            "中药": body,
        })
    return rows


def registry_with_corpus(**kwargs) -> ToolRegistry:
    return ToolRegistry(records=corpus(), deid_key=TEST_KEY, **kwargs)


# --------------------------------------------------------------- stub models

class ScriptedModel:
    """Replays a scripted sequence of turns and records what it was shown."""

    name, model, available = "scripted", "scripted", True

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen_tools: list[list[str]] = []
        self.systems: list[str] = []
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        self.seen_tools.append([t.name for t in (tools or [])])
        self.systems.append(messages[0]["content"])
        turn = self.turns.pop(0) if self.turns else LLMResponse(text="{}")
        return turn(messages) if callable(turn) else turn


def tool_turn(name, arguments=None):
    return LLMResponse(tool_calls=[ToolCall(name, arguments or {}, "c1")], prompt_tokens=5, completion_tokens=5)


def answer_turn(payload):
    def build(messages):
        evidence = [
            json.loads(m["content"]).get("evidence_id")
            for m in messages if m.get("role") == "tool" and json.loads(m["content"]).get("evidence_id")
        ]
        return LLMResponse(text=json.dumps({**payload, "citations": evidence}, ensure_ascii=False),
                           prompt_tokens=10, completion_tokens=10)
    return build


def make_loop(model, skill_id="yaobi.tcm_pattern", registry=None, budget=None):
    skills = SkillRegistry.from_file(MANIFEST)
    state = ClinicalRunState("腰痛3月，刺痛固定", role="physician")
    if budget is not None:
        state.budget = budget
    tools = registry or registry_with_corpus()
    broker = CapabilityBroker("physician", "routine", budget=state.budget,
                              skill_registry=skills, active_skill=skill_id)
    loop = ToolLoop(model, tools, broker, state, agent_name="TestAgent",
                    skill_id=skill_id, skill_spec=skills.specs[skill_id])
    return loop, state


PATTERN_ANSWER = {"primary_pattern": "气滞血瘀证", "candidate_patterns": ["气滞血瘀证", "寒湿痹阻证"]}


class ToolLoopTests(unittest.TestCase):
    def test_model_only_sees_tools_its_skill_grants(self):
        loop, _ = make_loop(ScriptedModel([]), "yaobi.tcm_pattern")
        self.assertEqual([s.name for s in loop.allowed_tool_specs()], ["tcm_pattern_knowledge_search"])

        loop, _ = make_loop(ScriptedModel([]), "yaobi.expert_case_reasoning")
        names = {s.name for s in loop.allowed_tool_specs()}
        self.assertIn("expert_practice_profile", names)
        self.assertNotIn("herb_dose_distribution", names)
        self.assertNotIn("physician_review_submit", names)

    def test_forbidden_tools_are_removed_from_the_visible_set(self):
        loop, _ = make_loop(ScriptedModel([]), "yaobi.medication_safety")
        names = {s.name for s in loop.allowed_tool_specs()}
        self.assertNotIn("herb_dose_distribution", names)

    def test_model_chooses_the_tool_and_its_answer_is_bound_to_evidence(self):
        model = ScriptedModel([
            tool_turn("tcm_pattern_knowledge_search", {"text": "腰痛 刺痛固定"}),
            answer_turn(PATTERN_ANSWER),
        ])
        loop, state = make_loop(model)
        result = loop.run("辨证", {"chief_complaint": "腰痛"}, "PatternAssessment")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.mode, "llm_tool_loop")
        self.assertEqual(result.output["primary_pattern"], "气滞血瘀证")
        self.assertEqual(result.citations, result.evidence_ids)
        self.assertEqual(len(state.evidence), 1)

    def test_bad_arguments_are_recoverable_and_never_become_failed_evidence(self):
        """Regression: a malformed model argument blocked the whole run."""
        model = ScriptedModel([
            tool_turn("tcm_pattern_knowledge_search", {}),                       # missing `text`
            tool_turn("tcm_pattern_knowledge_search", {"text": "腰痛"}),          # self-corrected
            answer_turn(PATTERN_ANSWER),
        ])
        loop, state = make_loop(model)
        result = loop.run("辨证", {}, "PatternAssessment")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(state.evidence), 1, "only the successful call should be recorded")
        self.assertTrue(all(e.payload.get("tool_ok") for e in state.evidence.values()))
        self.assertTrue(loop.broker.health.is_healthy("tcm_pattern_knowledge_search"))

    def test_recoverable_error_returns_the_parameter_schema_to_the_model(self):
        captured = {}

        def capture(messages):
            captured["observation"] = json.loads(messages[-1]["content"])
            return LLMResponse(text=json.dumps({**PATTERN_ANSWER, "citations": []}, ensure_ascii=False))

        loop, _ = make_loop(ScriptedModel([tool_turn("tcm_pattern_knowledge_search", {}), capture]))
        loop.run("辨证", {}, "PatternAssessment")
        self.assertTrue(captured["observation"]["recoverable"])
        self.assertIn("properties", captured["observation"]["parameters"])

    def test_unknown_tool_is_handed_back_not_fatal(self):
        model = ScriptedModel([
            tool_turn("definitely_not_a_tool", {}),
            tool_turn("tcm_pattern_knowledge_search", {"text": "腰痛"}),
            answer_turn(PATTERN_ANSWER),
        ])
        loop, state = make_loop(model)
        result = loop.run("辨证", {}, "PatternAssessment")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(state.evidence), 1)

    def test_a_dose_in_the_output_voids_the_whole_answer(self):
        model = ScriptedModel([
            tool_turn("tcm_pattern_knowledge_search", {"text": "腰痛"}),
            answer_turn({"primary_pattern": "气滞血瘀证", "candidate_patterns": ["独活 9克"]}),
        ])
        loop, _ = make_loop(model)
        result = loop.run("辨证", {}, "PatternAssessment")
        self.assertFalse(result.ok)
        self.assertEqual(result.mode, "dose_in_output")

    def test_schema_violation_voids_the_answer(self):
        model = ScriptedModel([
            tool_turn("tcm_pattern_knowledge_search", {"text": "腰痛"}),
            LLMResponse(text='{"随便": "乱写"}'),
        ])
        loop, _ = make_loop(model)
        result = loop.run("辨证", {}, "PatternAssessment")
        self.assertFalse(result.ok)
        self.assertEqual(result.mode, "schema_violation")

    def test_non_json_output_still_never_becomes_an_answer(self):
        """A prose reply gets one reformatting turn, then is rejected outright."""
        model = ScriptedModel([LLMResponse(text="我觉得是气滞血瘀证")])
        loop, _ = make_loop(model)
        result = loop.run("辨证", {}, "PatternAssessment")
        self.assertFalse(result.ok)
        self.assertIsNone(result.output)
        self.assertEqual(result.repairs, 1, "exactly one repair turn is allowed")

    def test_a_prose_reply_is_recovered_when_the_model_complies_on_retry(self):
        """A formatting miss must not discard the evidence the model gathered.

        Falling back wholesale over a missing code-fence is why runs reported
        "已回退确定性逻辑" while the model had actually done the work.
        """
        model = ScriptedModel([
            LLMResponse(text="我觉得是气滞血瘀证，理由如下……"),
            LLMResponse(text=json.dumps({
                "primary_pattern": "气滞血瘀",
                "candidate_patterns": ["寒湿"],
                "citations": [],
            }, ensure_ascii=False)),
        ])
        loop, _ = make_loop(model)
        result = loop.run("辨证", {}, "PatternAssessment")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.output["primary_pattern"], "气滞血瘀")
        self.assertEqual(result.repairs, 1)

    def test_the_repair_turn_shows_the_schema_again(self):
        captured: list[str] = []

        def record(messages):
            captured.append(messages[-1]["content"])
            return LLMResponse(text=json.dumps(
                {"primary_pattern": "气滞血瘀", "candidate_patterns": []}, ensure_ascii=False))

        model = ScriptedModel([LLMResponse(text="散文回答"), record])
        loop, _ = make_loop(model)
        result = loop.run("辨证", {}, "PatternAssessment")
        self.assertTrue(result.ok)
        self.assertTrue(captured)
        self.assertIn("只输出一个 JSON 对象", captured[0])
        self.assertIn("primary_pattern", captured[0])

    def test_a_dose_leak_is_never_given_a_second_chance(self):
        """Re-prompting a dose leak invites a reworded dose leak."""
        model = ScriptedModel([LLMResponse(text=json.dumps(
            {"primary_pattern": "气滞血瘀 当归12克", "candidate_patterns": []}, ensure_ascii=False))])
        loop, _ = make_loop(model)
        result = loop.run("辨证", {}, "PatternAssessment")
        self.assertFalse(result.ok)
        self.assertEqual(result.mode, "dose_in_output")
        self.assertEqual(result.repairs, 0)

    def test_model_error_falls_back(self):
        class Broken:
            name, model, available = "broken", "broken", True

            def chat(self, *a, **k):
                raise RuntimeError("upstream 502")

        loop, state = make_loop(Broken())
        result = loop.run("辨证", {}, "PatternAssessment")
        self.assertFalse(result.ok)
        self.assertEqual(result.mode, "llm_error")
        self.assertTrue(any("回退" in w for w in state.warnings))

    def test_llm_budget_exhaustion_falls_back(self):
        loop, _ = make_loop(ScriptedModel([]), budget=Budget(max_llm_calls=0))
        result = loop.run("辨证", {}, "PatternAssessment")
        self.assertEqual(result.mode, "llm_budget_exhausted")

    def test_loop_is_bounded(self):
        model = ScriptedModel([tool_turn("tcm_pattern_knowledge_search", {"text": "x"})] * 20)
        loop, _ = make_loop(model)
        loop.max_steps = 3
        result = loop.run("辨证", {}, "PatternAssessment")
        self.assertEqual(result.mode, "max_steps_exceeded")
        self.assertLessEqual(model.calls, 3)

    def test_skill_instructions_reach_the_system_prompt(self):
        model = ScriptedModel([answer_turn(PATTERN_ANSWER)])
        loop, _ = make_loop(model)
        loop.run("辨证", {}, "PatternAssessment")
        self.assertIn("primary_pattern 只给一个", model.systems[0])
        self.assertIn("不得输出任何药物克数", model.systems[0])

    def test_large_observations_stay_valid_json(self):
        """Regression: slicing the JSON string handed the model malformed JSON."""
        from yaobi_harness.agent.toolloop import MAX_OBSERVATION_CHARS, serialise_observation

        payload = {
            "evidence_id": "E0004", "ok": True, "summary": "检索",
            "data": {"cases": [{"主诉": "腰痛" * 90, "现病史": "久坐" * 90} for _ in range(60)]},
        }
        text = serialise_observation(payload)
        self.assertLessEqual(len(text), MAX_OBSERVATION_CHARS)
        parsed = json.loads(text)                       # must not raise
        self.assertEqual(parsed["evidence_id"], "E0004")
        self.assertEqual(parsed["data"]["cases"]["count"], 60)

    def test_small_observations_are_passed_through_untouched(self):
        from yaobi_harness.agent.toolloop import serialise_observation

        payload = {"evidence_id": "E1", "ok": True, "data": {"patterns": ["气滞血瘀证"]}}
        self.assertEqual(json.loads(serialise_observation(payload)), payload)

    def test_retrieval_heavy_skill_survives_a_large_corpus(self):
        registry = ToolRegistry(records=corpus(60), deid_key=TEST_KEY)
        model = ScriptedModel([
            tool_turn("similar_case_search", {"query": "腰痛", "limit": 40}),
            answer_turn({"similar": ["专家核心药一致"], "counterexamples": [], "limitation": "单一专家经验"}),
        ])
        loop, _ = make_loop(model, "yaobi.expert_case_reasoning", registry=registry)
        result = loop.run("综合", {}, "ExpertCaseEvidence")
        self.assertTrue(result.ok, result.error)

    def test_schema_hint_lists_required_fields(self):
        hint = schema_hint("PatternAssessment")
        self.assertIn("primary_pattern", hint)
        self.assertIn("citations", hint)


# ------------------------------------------------------------ agent wiring

def scripted_runner(turns_by_agent, registry=None, manifest=None):
    """A model that answers each autonomous agent from a per-agent script."""

    class PerAgent:
        name, model, available = "per-agent", "per-agent", True

        def __init__(self):
            self.used: list[str] = []

        def chat(self, messages, tools=None, **kwargs):
            names = [t.name for t in (tools or [])]
            if not names:
                return LLMResponse(text="{}", prompt_tokens=1, completion_tokens=1)
            agent = messages[0]["content"].split("**")[1]
            self.used.append(agent)
            observations = [json.loads(m["content"]) for m in messages if m.get("role") == "tool"]
            evidence = [o["evidence_id"] for o in observations if o.get("evidence_id")]
            if evidence:
                payload = turns_by_agent.get(agent)
                if payload is None:
                    return LLMResponse(text="{}")
                return LLMResponse(text=json.dumps({**payload, "citations": evidence}, ensure_ascii=False),
                                   prompt_tokens=10, completion_tokens=10)
            # Pick a tool we know how to call correctly, so the stub converges.
            tool = next((n for n in names if n in _KNOWN_ARGS), names[0])
            return tool_turn(tool, _default_args(tool))

    return YaobiGraphRunner(registry or registry_with_corpus(), skill_manifest=manifest, llm=PerAgent())


_KNOWN_ARGS = {
    "clinical_guideline_search", "tcm_pattern_knowledge_search", "similar_case_search",
    "expert_practice_profile", "formula_composition_search", "drug_label_lookup",
}


def _default_args(tool: str) -> dict:
    return {
        "clinical_guideline_search": {"topic": "low back pain"},
        "tcm_pattern_knowledge_search": {"text": "腰痛 刺痛固定"},
        "similar_case_search": {"query": "腰痛"},
        "expert_practice_profile": {"pattern": "气滞血瘀证"},
        "formula_composition_search": {"pattern": "气滞血瘀证"},
        "drug_label_lookup": {"ingredient": "ibuprofen"},
    }.get(tool, {})


AGENT_ANSWERS = {
    "BiomedicalAgent": {"differentials": ["腰椎间盘突出伴神经根病"], "exam_advice": ["直腿抬高试验"]},
    "TCMPatternAgent": PATTERN_ANSWER,
    "ExpertCaseAgent": {"similar": ["同证型 12 例核心药一致"], "counterexamples": [], "limitation": "单一专家经验"},
    "FormulaAgent": {"formula_name": "独活寄生汤加减", "herbs": CORE, "treatment_principle": ["补益肝肾"]},
}


class AutonomousAgentTests(unittest.TestCase):
    def test_agents_run_autonomously_when_the_skill_allows(self):
        runner = scripted_runner(AGENT_ANSWERS)
        out = runner.run(ClinicalRunState("腰痛3月，刺痛固定", role="physician"))
        autonomy = out.outputs["autonomy"]
        for agent in ("BiomedicalAgent", "TCMPatternAgent", "ExpertCaseAgent"):
            with self.subTest(agent=agent):
                self.assertEqual(autonomy[agent]["mode"], "llm_tool_loop")
        self.assertEqual(out.outputs["biomedical"]["_produced_by"], "llm_tool_loop")
        self.assertEqual(out.outputs["biomedical"]["differentials"], ["腰椎间盘突出伴神经根病"])

    def test_without_a_model_the_same_agents_use_the_deterministic_body(self):
        out = YaobiGraphRunner(registry_with_corpus()).run(
            ClinicalRunState("腰痛3月，刺痛固定", role="physician"))
        self.assertEqual(out.outputs["biomedical"]["_produced_by"], "rule_fallback")
        self.assertNotIn("autonomy", out.outputs)

    def test_claims_from_an_autonomous_agent_carry_the_llm_origin(self):
        runner = scripted_runner(AGENT_ANSWERS)
        out = runner.run(ClinicalRunState("腰痛3月，刺痛固定", role="physician"))
        llm_claims = [c for c in out.claims if c.origin == "llm"]
        self.assertTrue(llm_claims)
        self.assertTrue(all(c.evidence_ids for c in llm_claims))

    def test_a_model_proposed_formula_is_still_gated_by_classical_incompatibility(self):
        answers = {**AGENT_ANSWERS, "FormulaAgent": {
            "formula_name": "危险组合", "herbs": ["制川乌", "法半夏", "茯苓"],
            "treatment_principle": ["温经散寒"]}}
        runner = scripted_runner(answers)
        out = runner.run(ClinicalRunState("腰痛3月，怕冷", role="physician"), allow_prescription=True)
        self.assertNotIn("prescription_draft", out.outputs)
        self.assertTrue(any("配伍禁忌" in i for i in out.safety_issues), out.safety_issues)

    def test_dose_generation_is_never_delegated_to_the_model(self):
        skills = SkillRegistry.from_file(MANIFEST)
        self.assertFalse(skills.specs["yaobi.dose_generation"].autonomous)
        self.assertFalse(skills.specs["yaobi.urgent_triage"].autonomous)
        self.assertFalse(skills.specs["yaobi.safety_critic"].autonomous)

    def test_autonomous_run_still_produces_a_deterministic_dose_draft(self):
        registry = ToolRegistry(records=corpus(), deid_key=TEST_KEY,
                                authorized_ranges={h: (3.0, 15.0) for h in CORE})
        runner = scripted_runner(AGENT_ANSWERS, registry=registry)
        state = ClinicalRunState("腰痛3月，刺痛固定", role="physician")
        state.facts.update({
            "special_population": {"pregnancy": False, "age": 63, "renal": "normal", "liver": "normal"},
            "medications_confirmed": True, "allergies_confirmed": True,
        })
        out = runner.run(state, allow_prescription=True)
        self.assertEqual(out.release_status, "draft_for_physician", out.safety_issues)
        draft = out.outputs["prescription_draft"]
        self.assertTrue(all(h["authorized_range_g"] for h in draft["herbs"]))
        self.assertNotIn("_produced_by", draft)


# ------------------------------------------------------------ expert corpus

class ExpertProfileTests(unittest.TestCase):
    def profile(self, **kwargs):
        return build_profile(ExpertCaseStore.from_records(corpus(), deid_key=TEST_KEY).records, **kwargs)

    def test_pattern_extraction(self):
        self.assertEqual(extract_pattern("腰痹/证型：气血痹阻证"), "气血痹阻证")
        self.assertEqual(extract_pattern("腰痛"), "腰痛")
        self.assertEqual(extract_pattern(""), "未标注证型")

    def test_profile_finds_core_herbs_and_habits(self):
        profile = self.profile()
        self.assertEqual(profile.total_cases, 12)
        stasis = profile.pattern_for("气滞血瘀证")
        self.assertIsNotNone(stasis)
        self.assertTrue(stasis.reliable)
        core = {h.herb for h in stasis.core_herbs}
        self.assertTrue(set(CORE).issubset(core), core)
        self.assertTrue(any(t["value"] == "针灸" for t in stasis.treatment_methods))
        self.assertTrue(any(d["value"] == "塞来昔布" for d in stasis.western_drugs))
        self.assertTrue(any(i["value"] == "腰椎MRI" for i in stasis.investigations))

    def test_followup_uses_the_stable_pseudonym(self):
        followup = self.profile().followup
        self.assertGreater(followup["distinct_patients"], 0)
        self.assertGreater(followup["patients_with_followup"], 0)

    def test_min_support_withholds_singletons(self):
        rows = corpus()
        rows[0]["西药"] = "某种只出现一次的药"
        records = ExpertCaseStore.from_records(rows, deid_key=TEST_KEY).records
        drugs = {d["value"] for p in build_profile(records, min_support=2).patterns.values()
                 for d in p.western_drugs}
        self.assertNotIn("某种只出现一次的药", drugs)

    def test_pattern_lookup_tolerates_the_full_diagnosis_string(self):
        profile = self.profile()
        self.assertIsNotNone(profile.pattern_for("腰痹/证型：气滞血瘀证"))
        self.assertIsNone(profile.pattern_for("完全不存在的证"))


class ExpertSkillGenerationTests(unittest.TestCase):
    def profile(self):
        return build_profile(ExpertCaseStore.from_records(corpus(), deid_key=TEST_KEY).records)

    def test_generated_skill_replaces_the_built_in_one_by_id(self):
        self.assertEqual(SKILL_ID, "yaobi.expert_case_reasoning")
        entry = build_skill_entry(self.profile())
        self.assertTrue(entry["autonomous"])
        self.assertIn("expert_practice_profile", entry["allowed_tools"])
        self.assertIn("herb_dose_distribution", entry["forbidden_tools"])

    def test_instructions_contain_no_dose_values(self):
        """Reasoning agents may not emit doses, so they must not be shown any."""
        import re

        instructions = build_instructions(self.profile())
        self.assertEqual(re.findall(r"\d+(?:\.\d+)?\s*(?:克|g\b)", instructions), [])
        self.assertIn("核心药", instructions)

    def test_instructions_contain_no_free_text_sentences(self):
        entry = build_skill_entry(self.profile())
        self.assertEqual(audit_for_phi(entry), [])

    def test_phi_audit_catches_a_leaked_identifier(self):
        entry = build_skill_entry(self.profile())
        entry["instructions"] += "\n- 联系13800138000"
        self.assertTrue(any("phone" in p for p in audit_for_phi(entry)))

    def test_generated_skill_loads_and_merges(self):
        with TemporaryDirectory() as tmp:
            path = write_skill_file(self.profile(), Path(tmp) / "expert.yaml")
            registry = SkillRegistry.from_file(path)
            spec = registry.specs[SKILL_ID]
            self.assertTrue(spec.autonomous)
            self.assertIn("语料规模", spec.instructions)

            manifest = Path(tmp) / "manifest.yaml"
            manifest.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
            entry = yaml.safe_load(path.read_text(encoding="utf-8"))["skills"][0]
            merge_into_manifest(entry, manifest)
            merged = SkillRegistry.from_file(manifest)
        self.assertIn("语料规模", merged.specs[SKILL_ID].instructions)
        self.assertEqual(len([s for s in merged.specs if s == SKILL_ID]), 1)

    def test_expert_profile_tool_withholds_doses_but_dose_tool_keeps_them(self):
        registry = registry_with_corpus()
        profile_data = registry.expert_practice_profile("气滞血瘀证").data
        herbs = profile_data["patterns"][0]["core_herbs"]
        self.assertTrue(herbs)
        self.assertTrue(all("median_g" not in h for h in herbs))

        doses = registry.herb_dose_distribution(["独活"]).data["distributions"]["独活"]
        self.assertEqual(doses["median_g"], 9.0)

    def test_expert_profile_tool_reports_an_empty_corpus_as_a_stub(self):
        result = ToolRegistry(deid_key=TEST_KEY).expert_practice_profile()
        self.assertTrue(result.is_stub)
        self.assertEqual(result.data["total_cases"], 0)

    def test_expert_skill_changes_what_the_model_is_told(self):
        """The whole point: the corpus becomes the agent's instructions."""
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.yaml"
            manifest.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
            merge_into_manifest(build_skill_entry(self.profile()), manifest)
            runner = scripted_runner(AGENT_ANSWERS, manifest=manifest)
            out = runner.run(ClinicalRunState("腰痛3月，刺痛固定", role="physician"))
        self.assertEqual(out.outputs["autonomy"]["ExpertCaseAgent"]["mode"], "llm_tool_loop")
        self.assertIn("ExpertCaseAgent", runner.llm.used)


class SkillCatalogTests(unittest.TestCase):
    def test_catalog_exposes_purpose_and_autonomy_but_not_instructions(self):
        catalog = SkillRegistry.from_file(MANIFEST).catalog("physician")
        entry = next(c for c in catalog if c["skill_id"] == "yaobi.tcm_pattern")
        self.assertTrue(entry["autonomous"])
        self.assertIn("tcm_pattern_knowledge_search", entry["tools"])
        self.assertNotIn("instructions", entry)

    def test_catalog_filters_by_role(self):
        catalog = SkillRegistry.from_file(MANIFEST).catalog("patient")
        self.assertNotIn("yaobi.dose_generation", {c["skill_id"] for c in catalog})

    def test_agent_catalog_tools_never_exceed_their_skill(self):
        """Every catalogued agent's tools must be inside its skill's grant.

        Checked against the *discovered* registry, because a skill may be defined
        in ``manifest.yaml`` or in a ``SKILL.md``; either way the grant is what
        the broker enforces.
        """
        from yaobi_harness.agent.planner import AGENT_CATALOG

        skills = SkillRegistry.discover(MANIFEST)
        for spec in AGENT_CATALOG.values():
            with self.subTest(agent=spec.name):
                skill = skills.specs.get(spec.skill_id)
                self.assertIsNotNone(skill, f"{spec.skill_id} is not declared in any skill source")
                self.assertTrue(
                    set(spec.tools).issubset(set(skill.allowed_tools)),
                    f"{spec.name} requests {sorted(set(spec.tools) - set(skill.allowed_tools))} "
                    f"outside {spec.skill_id}",
                )


class InformalModelDrivesTheWholeRunTests(unittest.TestCase):
    """One test for the claim the whole design rests on.

    Every other test here checks a component. This one asks the end-to-end
    question a user asks: with a model that is competent but writes informally,
    does the model actually plan the graph, pick its own tools, and have its own
    conclusions delivered — or does the run quietly degrade to rules?

    It degraded. A Colab run reported ``planner_mode: rule`` beside a note saying
    the model had answered, and the differentials in the output were the
    deterministic list, not the model's. The stub below writes the way a real
    model writes: a ``steps`` list instead of ``tasks``, ``name``/``id`` instead
    of ``agent``/``task_id``, ``dependencies``/``after`` instead of
    ``depends_on``, wrapped in prose and a code fence, with a trailing comma on
    every answer. All of that used to be discarded.
    """

    TOOL_ARGS = {
        "clinical_guideline_search": {"topic": "low back pain"},
        "tcm_pattern_knowledge_search": {"pattern_hint": "气滞血瘀证"},
        "similar_case_search": {"complaint": "腰痛", "pattern": "气滞血瘀证"},
        "red_flag_evidence_search": {"text": "腰痛3月"},
        "drug_interaction_check": {"medications": ["布洛芬", "华法林"]},
    }
    # Keyed by a required field, because the prompt shows the field list rather
    # than the schema's name — which is what a real model has to go on too.
    ANSWERS = (
        ("differentials", {"differentials": ["腰椎间盘突出伴神经根病"], "exam_advice": ["腰椎MRI"]}),
        ("primary_pattern", {"primary_pattern": "气滞血瘀证", "candidate_patterns": ["寒湿痹阻证"]}),
        ("counterexamples", {"similar": [], "counterexamples": [], "limitation": "样本有限"}),
        ("findings", {"findings": [], "overall": "无重大相互作用"}),
    )
    PLAN = {
        "reasoning": "先筛红旗，再做西医鉴别与辨证",
        "steps": [
            {"id": "S1", "name": "IntakeAgent", "objective": "红旗与信息缺口"},
            {"id": "S2", "name": "BiomedicalAgent", "objective": "西医鉴别",
             "dependencies": ["S1"], "tools": ["clinical_guideline_search"]},
            {"id": "S3", "name": "TCMPatternAgent", "objective": "辨证", "after": ["S1"]},
            {"id": "S4", "name": "ExpertCaseAgent", "objective": "相似病例", "after": ["S3"]},
            {"id": "S5", "name": "MedicationSafetyAgent", "objective": "用药筛查", "after": ["S1"]},
        ],
    }

    class InformalModel:
        name, model, available = "informal", "informal-v1", True

        def __init__(self, outer):
            self.outer = outer
            self.planner_turns = self.tool_turns = self.answer_turns = 0

        def chat(self, messages, tools=None, **kwargs):
            system = messages[0]["content"]
            if "规划器" in system:
                self.planner_turns += 1
                return LLMResponse(
                    text="计划如下：\n```json\n"
                         + json.dumps(self.outer.PLAN, ensure_ascii=False) + "\n```\n以上。",
                    completion_tokens=40)
            names = [t.name for t in (tools or [])]
            if names and not any(m.get("role") == "tool" for m in messages):
                self.tool_turns += 1
                return LLMResponse(
                    tool_calls=[ToolCall(names[0], self.outer.TOOL_ARGS.get(names[0], {}), "c1")],
                    completion_tokens=8)
            self.answer_turns += 1
            body = next((dict(v) for key, v in self.outer.ANSWERS if key in system), None)
            if body is None:
                return LLMResponse(text="{}", completion_tokens=2)
            body["citations"] = [
                eid for m in messages if m.get("role") == "tool"
                for eid in [json.loads(m["content"]).get("evidence_id")] if eid
            ]
            # Trailing comma: a syntax slip, not an ambiguity.
            return LLMResponse(text=json.dumps(body, ensure_ascii=False)[:-1] + ",}",
                               completion_tokens=20)

    def setUp(self):
        self.model = self.InformalModel(self)
        state = ClinicalRunState("腰痛3月，久坐加重，右下肢麻木，无大小便异常", role="physician")
        state.facts.update({"medications": ["布洛芬 0.3g bid", "华法林 3mg qd"]})
        state.budget = Budget(max_llm_calls=80, max_tool_calls=80)
        self.out = YaobiGraphRunner(llm=self.model).run(state)

    def test_the_model_planned_the_graph(self):
        self.assertEqual(self.out.planner_mode, "llm", self.out.outputs["plan"]["note"])
        self.assertEqual(self.out.outputs["plan"]["note"], "llm_plan_accepted")
        self.assertEqual([t.task_id for t in self.out.tasks],
                         ["S1", "S2", "S3", "S4", "S5", "SAFETY"])

    def test_the_dependency_structure_survived_the_aliases(self):
        """`dependencies` and `after` must become `depends_on`, not be dropped —
        losing them flattens the graph while still looking like a success."""
        depends = {t.task_id: t.depends_on for t in self.out.tasks}
        self.assertEqual(depends["S2"], ["S1"])
        self.assertEqual(depends["S3"], ["S1"])
        self.assertEqual(depends["S4"], ["S3"])

    def test_every_autonomous_agent_ran_the_tool_loop(self):
        autonomy = self.out.outputs.get("autonomy", {})
        self.assertTrue(autonomy, "no agent ran autonomously")
        for agent, info in autonomy.items():
            with self.subTest(agent=agent):
                self.assertEqual(info["mode"], "llm_tool_loop", info.get("error"))
                self.assertEqual(info["repairs"], 0, "a trailing comma must not cost a repair turn")
                self.assertTrue([s for s in info["steps"] if s.get("tool")],
                                "the model chose no tool")

    def test_the_delivered_conclusion_is_the_models_own(self):
        """The point of the whole exercise: the answer came from the model, not
        from the deterministic fallback list."""
        self.assertEqual(self.out.outputs["biomedical"]["differentials"],
                         ["腰椎间盘突出伴神经根病"])
        self.assertEqual(self.out.outputs["tcm_pattern"]["primary_pattern"], "气滞血瘀证")

    def test_the_control_plane_still_ran(self):
        """Model-driven is not model-governed: the safety node and the release
        state machine are unchanged by any of this."""
        self.assertIn("SAFETY", [t.task_id for t in self.out.tasks])
        self.assertEqual([t.status for t in self.out.tasks if t.task_id == "SAFETY"], ["ok"])
        self.assertIn(self.out.release_status,
                      ("needs_examination", "needs_more_information", "treatment_advice_only"))
        self.assertNotIn("prescription_draft", self.out.outputs)


if __name__ == "__main__":
    unittest.main()
