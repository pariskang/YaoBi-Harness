"""Regression suite.

Every test below either pins a safety invariant or reproduces a defect found in
the V0.0 skeleton. Tests are written for ``unittest`` so they run with either
``python -m unittest discover`` or ``pytest``.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("YAOBI_DEID_KEY", "unit-test-fixed-key")

from yaobi_harness.agent.planner import PlannerAgent, rule_plan, validate_plan
from yaobi_harness.graph import YaobiGraphRunner, resume_run
from yaobi_harness.llm.base import LLMResponse, NullLLMClient, extract_json
from yaobi_harness.llm.factory import build_client
from yaobi_harness.render import render
from yaobi_harness.safety import incompatibility as incompat
from yaobi_harness.safety import red_flags
from yaobi_harness.skills.loader import SkillManifestError, SkillRegistry
from yaobi_harness.state import Budget, ClinicalRunState, Task
from yaobi_harness.tools import (
    CapabilityBroker, DeidentificationKeyError, ExpertCaseStore, ToolRegistry, parse_herbs,
)

TEST_KEY = "unit-test-fixed-key"
MANIFEST = Path(__file__).resolve().parents[1] / "yaobi_harness" / "skills" / "manifest.yaml"

RAW_RECORD = {
    "就诊序号": "12725633", "医师工号": "007", "医师姓名": "沈钦荣", "科室代码": "1166",
    "姓名": "张三", "性别": "男", "年龄": "63岁", "病案号": "50512983", "地址": "某区某街道",
    "主诉": "右腰部疼痛10天", "现病史": "久坐后疼痛明显，无双下肢麻木疼痛。舌略暗。",
    "中医诊断": "腰痹/证型：气血痹阻证", "西医诊断": "腰痛",
    "中药": "1/独活*1克/10克/用法：无/贴数:7\n,2/盐杜仲*1克/12克/用法：无/贴数:7\n,3/细辛*1克/3克/用法：无/贴数:7",
}

FORMULA_HERBS = ["独活", "桑寄生", "杜仲", "牛膝", "当归", "川芎", "白芍", "熟地黄", "党参", "茯苓", "甘草"]
BLOOD_STASIS_EXTRA = ["桃仁", "红花", "延胡索"]
SAFE_RANGES = {h: (3.0, 15.0) for h in FORMULA_HERBS + BLOOD_STASIS_EXTRA}
TIGHT_RANGES = {h: (3.0, 9.0) for h in FORMULA_HERBS + BLOOD_STASIS_EXTRA}


def case_with_doses(index: int, dose: float, herbs=None, pattern: str = "腰痹/证型：气滞血瘀证") -> dict:
    herbs = herbs or FORMULA_HERBS + BLOOD_STASIS_EXTRA
    body = "".join(f",{i}/{h}*1克/{dose}克/用法：无/贴数:7\n" for i, h in enumerate(herbs, 1))
    return {"病案号": f"C{index}", "年龄": "63岁", "主诉": "腰痛", "中医诊断": pattern, "中药": body}


def physician_state(complaint: str = "腰痛3月，刺痛固定，久坐加重") -> ClinicalRunState:
    state = ClinicalRunState(complaint, role="physician")
    state.facts.update(
        {
            "special_population": {"pregnancy": False, "age": 63, "renal": "normal", "liver": "normal"},
            "medications_confirmed": True,
            "allergies_confirmed": True,
        }
    )
    return state


# --------------------------------------------------------------------- privacy

class DeidentificationTests(unittest.TestCase):
    def test_case_search_never_returns_direct_identifiers(self):
        out = ToolRegistry(records=[RAW_RECORD], deid_key=TEST_KEY).case_store.search("腰痛 久坐", 1)[0]
        blob = str(out)
        for identifier in ("张三", "50512983", "某区某街道", "007", "沈钦荣"):
            self.assertNotIn(identifier, blob)
        self.assertTrue(out["research_patient_id"].startswith("YP"))

    def test_missing_deid_key_fails_loudly_instead_of_randomising(self):
        """Regression: a random fallback key silently broke longitudinal lookup."""
        saved = os.environ.pop("YAOBI_DEID_KEY", None)
        try:
            with self.assertRaises(DeidentificationKeyError):
                ExpertCaseStore.from_records([RAW_RECORD])
        finally:
            if saved is not None:
                os.environ["YAOBI_DEID_KEY"] = saved

    def test_research_id_is_stable_across_stores(self):
        first = ExpertCaseStore.from_records([RAW_RECORD], deid_key=TEST_KEY).records[0]["research_patient_id"]
        second = ExpertCaseStore.from_records([RAW_RECORD], deid_key=TEST_KEY).records[0]["research_patient_id"]
        self.assertEqual(first, second)

    def test_patient_timeline_search_finds_repeat_visits(self):
        rows = [dict(RAW_RECORD), dict(RAW_RECORD, 就诊序号="12725634")]
        registry = ToolRegistry(records=rows, deid_key=TEST_KEY)
        research_id = registry.case_store.records[0]["research_patient_id"]
        broker = CapabilityBroker("physician", "routine")
        result = registry.call(broker, "patient_timeline_search", research_patient_id=research_id)
        self.assertTrue(result.ok)
        self.assertEqual(result.data["visit_count"], 2)

    def test_dlp_redacts_ids_phones_and_dates_in_free_text(self):
        row = dict(RAW_RECORD, 现病史="联系13800138000，身份证110101199001011234，2023-05-06就诊")
        record = ExpertCaseStore.from_records([row], deid_key=TEST_KEY).records[0]
        self.assertNotIn("13800138000", record["现病史"])
        self.assertNotIn("110101199001011234", record["现病史"])
        self.assertNotIn("2023-05-06", record["现病史"])

    def test_herb_dose_parser_matches_star_slash_format(self):
        herbs = parse_herbs(RAW_RECORD["中药"])
        self.assertEqual({h["herb_name"]: h["dose_g"] for h in herbs}, {"独活": 10.0, "盐杜仲": 12.0, "细辛": 3.0})


# -------------------------------------------------------------------- triage

class RedFlagScreeningTests(unittest.TestCase):
    def test_history_prefix_no_longer_suppresses_a_live_emergency(self):
        """Regression P0-3: ``既往体健，现突发胸痛`` used to screen as routine."""
        for text in (
            "既往体健，现突发胸痛、大汗、呼吸困难",
            "去年做过腰椎手术，今天突然不能排尿、会阴麻木",
            "多年前有腰痛史，现在双腿越来越无力",
            "无高血压病史，但今日出现胸痛",
        ):
            with self.subTest(text=text):
                self.assertTrue(red_flags.screen(text).urgent, text)

    def test_colloquial_symptom_wording_is_covered(self):
        for text in ("腰痛数月，近日出现小便解不出来", "腰痛，屁股和大腿根发麻，尿憋不住", "石膏后剧痛，肢端发凉苍白"):
            with self.subTest(text=text):
                self.assertTrue(red_flags.screen(text).urgent, text)

    def test_negation_family_history_and_hypotheticals_stay_routine(self):
        for text in (
            "腰痛，无发热、无外伤、无大小便失禁、无会阴麻木",
            "父亲患癌，本人只是久坐腰酸",
            "如果以后胸痛怎么办，目前无不适",
            "去年曾跌倒且已经痊愈",
            "慢性骨质疏松，多年无新发症状",
        ):
            with self.subTest(text=text):
                self.assertFalse(red_flags.screen(text).urgent, text)

    def test_isolated_night_pain_is_soft_not_urgent(self):
        result = red_flags.screen("单纯夜间腰痛")
        self.assertFalse(result.urgent)
        self.assertTrue(result.soft_hits)

    def test_night_pain_with_corroboration_escalates(self):
        self.assertTrue(red_flags.screen("夜间腰痛，伴发热和体重下降").urgent)

    def test_llm_signals_can_only_add_never_clear(self):
        base = red_flags.screen("腰痛，无发热")
        merged = red_flags.merge_llm_hits(base, [{"signal": "cauda_equina", "term": "解手费劲"}])
        self.assertTrue(merged.urgent)
        untouched = red_flags.screen("突发胸痛")
        self.assertTrue(red_flags.merge_llm_hits(untouched, []).urgent)

    def test_urgent_plan_text_has_no_python_repr_leaking_to_the_patient(self):
        out = YaobiGraphRunner().run(ClinicalRunState("突发腰痛伴尿潴留和会阴麻木", role="patient"))
        judgement = out.outputs["urgent_action_plan"]["risk_judgement"]
        self.assertNotIn("[", judgement)
        self.assertNotIn("cauda_equina", judgement)
        self.assertIn("马尾", judgement)

    def test_urgent_run_withholds_prescription_and_returns_action_plan(self):
        out = YaobiGraphRunner().run(ClinicalRunState("突发腰痛伴尿潴留和会阴麻木", role="patient"), allow_prescription=True)
        self.assertEqual(out.risk_mode, "urgent")
        self.assertEqual(out.release_status, "urgent_action_plan")
        self.assertNotIn("prescription_draft", out.outputs)
        self.assertIn("immediate_action", out.outputs["urgent_action_plan"])


# ------------------------------------------------------------------ dose safety

class DoseSafetyTests(unittest.TestCase):
    def test_dose_outside_authorized_range_is_blocked(self):
        """Regression P0-1: 30 g doses were drafted against a 3-9 g range."""
        registry = ToolRegistry(
            records=[case_with_doses(i, 30.0) for i in range(5)],
            authorized_ranges=TIGHT_RANGES,
            deid_key=TEST_KEY,
        )
        out = YaobiGraphRunner(registry).run(physician_state(), allow_prescription=True)
        self.assertNotIn("prescription_draft", out.outputs)
        self.assertNotEqual(out.release_status, "draft_for_physician")
        self.assertTrue(any("超出授权" in issue or "异常值" in issue for issue in out.safety_issues), out.safety_issues)

    def test_in_range_doses_with_enough_samples_produce_a_draft(self):
        registry = ToolRegistry(
            records=[case_with_doses(i, 9.0) for i in range(6)],
            authorized_ranges=SAFE_RANGES,
            deid_key=TEST_KEY,
        )
        out = YaobiGraphRunner(registry).run(physician_state(), allow_prescription=True)
        self.assertEqual(out.release_status, "draft_for_physician", out.safety_issues)
        draft = out.outputs["prescription_draft"]
        self.assertTrue(draft["requires_physician_approval"])
        self.assertTrue(draft["prescription_hash"])
        for herb in draft["herbs"]:
            low, high = herb["authorized_range_g"]
            self.assertLessEqual(low, herb["dose_value"])
            self.assertLessEqual(herb["dose_value"], high)

    def test_pharmacopeia_check_compares_the_proposed_dose(self):
        registry = ToolRegistry(authorized_ranges={"独活": (3.0, 9.0)}, deid_key=TEST_KEY)
        inside = registry.pharmacopeia_check(["独活"], doses={"独活": 6.0})
        outside = registry.pharmacopeia_check(["独活"], doses={"独活": 30.0})
        self.assertTrue(inside.data["checked"][0]["dose_within_range"])
        self.assertFalse(outside.data["checked"][0]["dose_within_range"])
        self.assertFalse(outside.data["pass"])

    def test_missing_authorized_range_blocks_even_with_many_samples(self):
        registry = ToolRegistry(records=[case_with_doses(i, 9.0) for i in range(6)], deid_key=TEST_KEY)
        out = YaobiGraphRunner(registry).run(physician_state(), allow_prescription=True)
        self.assertNotIn("prescription_draft", out.outputs)
        self.assertTrue(any("授权药典范围" in issue for issue in out.safety_issues))

    def test_single_outlier_case_cannot_create_a_draft(self):
        registry = ToolRegistry(records=[case_with_doses(1, 999.0)], deid_key=TEST_KEY)
        out = YaobiGraphRunner(registry).run(physician_state("腰痛3月，久坐加重"), allow_prescription=True)
        self.assertNotIn("prescription_draft", out.outputs)
        self.assertEqual(out.release_status, "treatment_advice_only")

    def test_unconfirmed_medication_history_blocks_the_draft(self):
        state = physician_state()
        state.facts["medications_confirmed"] = False
        registry = ToolRegistry(
            records=[case_with_doses(i, 9.0) for i in range(6)], authorized_ranges=SAFE_RANGES, deid_key=TEST_KEY
        )
        out = YaobiGraphRunner(registry).run(state, allow_prescription=True)
        self.assertNotIn("prescription_draft", out.outputs)


class CombinationSafetyTests(unittest.TestCase):
    def test_eighteen_antagonisms_are_detected(self):
        violations = incompat.check_combination(["制川乌", "法半夏", "茯苓"])
        self.assertTrue(violations)
        self.assertEqual(violations[0]["rule"], "十八反")

    def test_nineteen_incompatibilities_are_detected(self):
        self.assertTrue(incompat.check_combination(["丁香", "郁金"]))
        self.assertTrue(incompat.check_combination(["肉桂", "赤石脂"]))

    def test_licorice_and_kansui_are_detected_through_aliases(self):
        self.assertTrue(incompat.check_combination(["炙甘草", "醋甘遂"]))

    def test_safe_formula_reports_no_violation(self):
        self.assertEqual(incompat.check_combination(FORMULA_HERBS), [])

    def test_pregnancy_contraindications_are_detected(self):
        self.assertTrue(incompat.check_pregnancy(["桃仁", "茯苓"]))
        self.assertEqual(incompat.check_pregnancy(["茯苓", "白芍"]), [])

    def test_interaction_check_surfaces_combination_violations(self):
        result = ToolRegistry(deid_key=TEST_KEY).interaction_check(
            ["制川乌", "法半夏"], medications_confirmed=True, allergies_confirmed=True
        )
        self.assertFalse(result.data["pass"])
        self.assertTrue(result.data["combination_violations"])

    def test_risk_herb_still_requires_special_review(self):
        registry = ToolRegistry(authorized_ranges={"细辛": (1.0, 3.0)}, deid_key=TEST_KEY)
        checked = registry.pharmacopeia_check(["细辛"], doses={"细辛": 3.0}).data["checked"][0]
        self.assertFalse(checked["ok"])
        self.assertIn("risk_herb_requires_special_review", checked["risk_flags"])


# ------------------------------------------------------------- control plane

class CapabilityAndBudgetTests(unittest.TestCase):
    def test_denied_call_does_not_consume_budget(self):
        """Regression P1-4: denials used to drain the budget before any check."""
        budget = Budget(max_tool_calls=10)
        broker = CapabilityBroker("patient", "routine", budget=budget)
        for _ in range(3):
            allowed, reason = broker.allow("herb_dose_distribution")
            self.assertFalse(allowed)
            self.assertEqual(reason, "patient_role_forbids_formula_or_dose_tools")
        self.assertEqual(budget.used_tool_calls, 0)

    def test_executed_call_consumes_exactly_one_unit(self):
        budget = Budget(max_tool_calls=10)
        broker = CapabilityBroker("physician", "routine", budget=budget)
        ToolRegistry(deid_key=TEST_KEY).call(broker, "red_flag_evidence_search", text="腰痛")
        self.assertEqual(budget.used_tool_calls, 1)

    def test_zero_budget_fails_closed_without_executing_tools(self):
        state = ClinicalRunState("腰痛3月", role="physician")
        state.budget = Budget(max_tool_calls=0)
        out = YaobiGraphRunner().run(state)
        self.assertEqual(out.release_status, "failed_closed")
        self.assertEqual(out.budget.used_tool_calls, 0)

    def test_repeated_failures_open_the_circuit_breaker(self):
        registry = ToolRegistry(failing_tools={"clinical_guideline_search"}, deid_key=TEST_KEY)
        state = ClinicalRunState("腰痛3月，久坐加重", role="physician")
        broker = CapabilityBroker("physician", "routine", budget=state.budget)
        registry.call(broker, "clinical_guideline_search", topic="x")
        self.assertFalse(broker.health.is_healthy("clinical_guideline_search"))
        allowed, reason = broker.allow("clinical_guideline_search")
        self.assertFalse(allowed)
        self.assertEqual(reason, "tool_unhealthy_circuit_open")

    def test_critical_tool_failure_fails_closed(self):
        out = YaobiGraphRunner(ToolRegistry(failing_tools={"clinical_guideline_search"}, deid_key=TEST_KEY)).run(
            ClinicalRunState("腰痛3月，久坐加重", role="physician")
        )
        self.assertEqual(out.release_status, "failed_closed")
        self.assertTrue(any("关键工具失败" in x for x in out.safety_issues))

    def test_patient_cannot_reach_formula_or_dose_tools(self):
        out = YaobiGraphRunner().run(ClinicalRunState("腰痛3月，久坐加重", role="patient"), allow_prescription=True)
        self.assertNotIn("formula", out.outputs)
        self.assertNotIn("prescription_draft", out.outputs)


class SkillPolicyTests(unittest.TestCase):
    def test_manifest_enforces_forbidden_tools(self):
        registry = SkillRegistry.from_file(MANIFEST)
        ok, problems = registry.enforce("yaobi.urgent_triage", "patient", ["red_flag_evidence_search", "herb_dose_distribution"])
        self.assertFalse(ok)
        self.assertTrue(any("forbidden" in p for p in problems))

    def test_empty_allowlist_means_no_tools(self):
        registry = SkillRegistry.from_file(MANIFEST)
        ok, _ = registry.enforce("yaobi.safety_critic", "physician", ["red_flag_evidence_search"])
        self.assertFalse(ok)

    def test_unknown_skill_is_denied_not_ignored(self):
        registry = SkillRegistry.from_file(MANIFEST)
        ok, problems = registry.enforce("yaobi.does_not_exist", "physician", ["red_flag_evidence_search"])
        self.assertFalse(ok)
        self.assertIn("not found", problems[0])

    def test_agent_without_a_declared_skill_gets_no_tool_rights(self):
        """Regression P1-7: an unmapped agent used to bypass skill policy entirely."""
        registry = SkillRegistry.from_file(MANIFEST)
        broker = CapabilityBroker("physician", "routine", skill_registry=registry, active_skill=None)
        allowed, reason = broker.allow("red_flag_evidence_search")
        self.assertFalse(allowed)
        self.assertIn("no_active_skill_declared", reason)

    def test_malformed_manifest_raises_instead_of_degrading(self):
        with self.assertRaises(SkillManifestError):
            SkillRegistry.from_document({"skills": [{"skill_id": "a", "allowed_tools": "not-a-list"}]})
        with self.assertRaises(SkillManifestError):
            SkillRegistry.from_document({"not_skills": []})
        with self.assertRaises(SkillManifestError):
            SkillRegistry.from_document({"skills": [{"skill_id": "a", "bogus_key": 1}]})

    def test_runner_fails_closed_when_manifest_omits_a_skill(self, ):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.yaml"
            manifest.write_text("skills:\n  - skill_id: yaobi.intake\n    version: 3.0.0\n    allowed_tools: []\n", encoding="utf-8")
            out = YaobiGraphRunner(skill_manifest=manifest).run(ClinicalRunState("腰痛", role="patient"))
        self.assertEqual(out.release_status, "failed_closed")
        self.assertTrue(any("skill_policy_denied" in x for x in out.safety_issues))


# ------------------------------------------------------------------- the graph

class GraphControlTests(unittest.TestCase):
    def test_critic_runs_even_when_every_clinical_task_is_skipped(self):
        """Regression P0-2: the critic was dependency-gated out of the patient path."""
        out = YaobiGraphRunner().run(ClinicalRunState("腰痛3月，久坐加重", role="patient"))
        self.assertIn("CriticAgent", [t.agent for t in out.traces])
        self.assertIn("safety_audit", out.outputs)
        self.assertIn("citation_guard", out.outputs["safety_audit"]["checks_run"])

    def test_critic_runs_after_a_failed_closed_run(self):
        out = YaobiGraphRunner(ToolRegistry(failing_tools={"red_flag_evidence_search"}, deid_key=TEST_KEY)).run(
            ClinicalRunState("腰痛", role="patient")
        )
        self.assertEqual(out.release_status, "failed_closed")
        self.assertIn("safety_audit", out.outputs)

    def test_critic_independently_rejects_a_tampered_out_of_range_draft(self):
        state = physician_state()
        state.outputs["prescription_draft"] = {
            "formula_name": "x",
            "herbs": [{"herb_name": "独活", "dose_value": 30.0, "authorized_range_g": [3.0, 9.0]}],
            "requires_physician_approval": True,
        }
        state.release_status = "draft_for_physician"
        from yaobi_harness.agent.agents import CriticAgent

        CriticAgent().run(state)
        self.assertEqual(state.release_status, "blocked")
        self.assertTrue(any("独立复核" in issue for issue in state.safety_issues))

    def test_repair_loop_is_bounded_and_terminates(self):
        """A critic that never stops complaining must not loop forever."""

        class NeverSatisfiedCritic:
            name, skill_id = "CriticAgent", "yaobi.safety_critic"
            calls = 0

            def run(self, state, tools=None, broker=None):
                NeverSatisfiedCritic.calls += 1
                state.outputs["safety_audit"] = {
                    "checks_run": ["stub"],
                    "issues": [],
                    "repair_requests": [{"agent": "BiomedicalAgent", "reason": "always_repair"}],
                }
                return state

        runner = YaobiGraphRunner()
        runner.agents["CriticAgent"] = NeverSatisfiedCritic()
        state = ClinicalRunState("腰痛3月，久坐加重", role="patient")
        state.budget = Budget(max_loops=2, max_tool_calls=64)
        out = runner.run(state)
        self.assertEqual(out.budget.loop_counts.get("repair"), 2)
        self.assertTrue(any("最大修复轮次" in w for w in out.warnings))

    def test_repair_request_reruns_the_named_agent_and_its_downstream(self):
        state = ClinicalRunState("腰痛3月，久坐加重", role="patient")
        state.tasks = [
            Task("A", "BiomedicalAgent", "a", status="ok"),
            Task("B", "TCMPatternAgent", "b", status="ok"),
            Task("SAFETY", "CriticAgent", "c", status="ok"),
        ]
        changed = YaobiGraphRunner()._reset_for_repair(state, [{"agent": "BiomedicalAgent", "reason": "r"}])
        self.assertTrue(changed)
        self.assertEqual(state.task_by_id("A").status, "repair_requested")
        self.assertEqual(state.task_by_id("B").status, "pending")
        self.assertEqual(state.task_by_id("SAFETY").status, "ok")

    def test_checkpoint_round_trip_restores_state(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = YaobiGraphRunner(checkpoint_dir=tmp).run(ClinicalRunState("腰痛3月，久坐加重", role="patient"))
            latest = Path(tmp) / f"{out.run_id}.latest.json"
            self.assertTrue(latest.exists())
            restored = YaobiGraphRunner.load_checkpoint(latest)
        self.assertEqual(restored.run_id, out.run_id)
        self.assertEqual(restored.release_status, out.release_status)
        self.assertEqual(len(restored.tasks), len(out.tasks))
        self.assertEqual(len(restored.evidence), len(out.evidence))

    def test_resume_continues_a_checkpointed_run(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            first = YaobiGraphRunner(checkpoint_dir=tmp).run(ClinicalRunState("腰痛3月，久坐加重", role="patient"))
            resumed = resume_run(Path(tmp) / f"{first.run_id}.latest.json")
        self.assertEqual(resumed.run_id, first.run_id)
        self.assertIn("safety_audit", resumed.outputs)

    def test_resume_with_new_facts_unblocks_the_prescriptive_path(self):
        """Resuming is only meaningful if newly supplied facts change the outcome."""
        import tempfile

        registry = ToolRegistry(
            records=[case_with_doses(i, 9.0) for i in range(6)], authorized_ranges=SAFE_RANGES, deid_key=TEST_KEY
        )
        with tempfile.TemporaryDirectory() as tmp:
            incomplete = ClinicalRunState("腰痛3月，刺痛固定，久坐加重", role="physician")
            first = YaobiGraphRunner(registry, checkpoint_dir=tmp).run(incomplete, allow_prescription=True)
            self.assertNotIn("prescription_draft", first.outputs)

            state = YaobiGraphRunner.load_checkpoint(Path(tmp) / f"{first.run_id}.latest.json")
            state.facts.update({
                "special_population": {"pregnancy": False, "age": 63, "renal": "normal", "liver": "normal"},
                "medications_confirmed": True,
                "allergies_confirmed": True,
            })
            second = YaobiGraphRunner(registry, checkpoint_dir=tmp).run(state, allow_prescription=True, resumed=True)
        self.assertEqual(second.release_status, "draft_for_physician", second.safety_issues)

    def test_resume_never_reopens_a_failed_closed_run(self):
        state = ClinicalRunState("腰痛", role="physician")
        state.fail_closed("injected")
        state.tasks = [Task("A", "BiomedicalAgent", "a", status="skipped_dependency")]
        YaobiGraphRunner()._reopen_for_resume(state)
        self.assertEqual(state.task_by_id("A").status, "skipped_dependency")

    def test_physician_review_closes_the_loop(self):
        registry = ToolRegistry(
            records=[case_with_doses(i, 9.0) for i in range(6)], authorized_ranges=SAFE_RANGES, deid_key=TEST_KEY
        )
        drafted = YaobiGraphRunner(registry).run(physician_state(), allow_prescription=True)
        draft = drafted.outputs["prescription_draft"]

        approving = physician_state()
        approving.facts["physician_review"] = {
            "physician_id": "D1",
            "signature": "sig",
            "approvals": {h["herb_name"]: True for h in draft["herbs"]},
        }
        approved = YaobiGraphRunner(registry).run(approving, allow_prescription=True)
        self.assertEqual(approved.release_status, "approved_by_physician")
        self.assertTrue(approved.outputs["physician_review"]["approved"])

    def test_physician_review_rejects_incomplete_risk_herb(self):
        result = ToolRegistry(deid_key=TEST_KEY).physician_review_submit(
            {"prescription_hash": "abc", "herbs": [{"herb_name": "附片"}]},
            {"附片": True}, physician_id="D1", signature="sig",
        )
        self.assertFalse(result.ok)
        self.assertTrue(any("risk_herb" in p or "missing_dose" in p for p in result.data["problems"]))

    def test_stub_guideline_is_not_recorded_as_guideline_grade(self):
        """Regression P1-6: a placeholder source was laundered into guideline evidence."""
        out = YaobiGraphRunner().run(ClinicalRunState("腰痛3月，久坐加重", role="physician"))
        levels = {e.source: e.level for e in out.evidence.values()}
        self.assertEqual(levels["clinical_guideline_search"], "stub_not_for_clinical_use")


# ------------------------------------------------------------------- planning

class PlannerTests(unittest.TestCase):
    def test_rule_plan_validates(self):
        state = ClinicalRunState("腰痛", role="physician")
        ok, problems = validate_plan(rule_plan(state), state)
        self.assertTrue(ok, problems)

    def test_plan_with_unknown_agent_is_rejected(self):
        state = ClinicalRunState("腰痛", role="physician")
        ok, problems = validate_plan([Task("X1", "EvilAgent", "exfiltrate")], state)
        self.assertFalse(ok)
        self.assertTrue(any("unknown agent" in p for p in problems))

    def test_plan_requesting_a_tool_outside_the_skill_is_rejected(self):
        state = ClinicalRunState("腰痛", role="physician")
        ok, problems = validate_plan([Task("X1", "BiomedicalAgent", "x", ["herb_dose_distribution"])], state)
        self.assertFalse(ok)
        self.assertTrue(any("outside its skill" in p for p in problems))

    def test_urgent_plan_may_not_contain_prescriptive_agents(self):
        state = ClinicalRunState("腰痛", role="physician")
        state.risk_mode = "urgent"
        ok, problems = validate_plan([Task("X1", "DoseAgent", "x")], state)
        self.assertFalse(ok)
        self.assertTrue(any("urgent" in p for p in problems))

    def test_cyclic_plan_is_rejected(self):
        state = ClinicalRunState("腰痛", role="physician")
        tasks = [Task("A", "BiomedicalAgent", "a", [], ["B"]), Task("B", "TCMPatternAgent", "b", [], ["A"])]
        ok, problems = validate_plan(tasks, state)
        self.assertFalse(ok)
        self.assertTrue(any("cycle" in p for p in problems))

    def test_llm_plan_is_used_when_valid(self):
        class StubLLM:
            name, model, available = "stub", "stub", True

            def chat(self, messages, **kwargs):
                return LLMResponse(text=json.dumps({"tasks": [
                    {"task_id": "P1", "agent": "BiomedicalAgent", "objective": "先做西医鉴别",
                     "required_tools": ["clinical_guideline_search"]},
                    {"task_id": "P2", "agent": "TCMPatternAgent", "objective": "再辨证",
                     "required_tools": ["tcm_pattern_knowledge_search"], "depends_on": ["P1"]},
                ]}), prompt_tokens=10, completion_tokens=20)

        state = ClinicalRunState("腰痛3月", role="physician")
        PlannerAgent(StubLLM()).run(state)
        self.assertEqual(state.planner_mode, "llm")
        self.assertEqual([t.agent for t in state.tasks], ["BiomedicalAgent", "TCMPatternAgent", "CriticAgent"])
        self.assertEqual(state.budget.used_llm_calls, 1)
        self.assertEqual(state.budget.used_llm_tokens, 30)

    def test_malicious_llm_plan_falls_back_to_rules(self):
        class EvilLLM:
            name, model, available = "evil", "evil", True

            def chat(self, messages, **kwargs):
                return LLMResponse(text=json.dumps({"tasks": [
                    {"task_id": "P1", "agent": "DoseAgent", "objective": "直接开方",
                     "required_tools": ["physician_review_submit", "similar_case_search"]},
                ]}))

        state = ClinicalRunState("突发胸痛、大汗", role="patient")
        state.risk_mode = "urgent"
        PlannerAgent(EvilLLM()).run(state)
        self.assertEqual(state.planner_mode, "rule")
        self.assertNotIn("DoseAgent", [t.agent for t in state.tasks])
        self.assertTrue(any("驳回" in w for w in state.warnings))

    def test_llm_failure_falls_back_to_rules(self):
        class BrokenLLM:
            name, model, available = "broken", "broken", True

            def chat(self, messages, **kwargs):
                raise RuntimeError("upstream 502")

        state = ClinicalRunState("腰痛3月", role="physician")
        PlannerAgent(BrokenLLM()).run(state)
        self.assertEqual(state.planner_mode, "rule")
        self.assertTrue(state.tasks)

    def test_critic_task_is_always_appended(self):
        state = ClinicalRunState("腰痛3月", role="physician")
        PlannerAgent(None).run(state)
        self.assertEqual(state.tasks[-1].agent, "CriticAgent")


# ------------------------------------------------------------------------ llm

class LLMLayerTests(unittest.TestCase):
    def test_no_provider_configured_yields_null_client(self):
        client = build_client("none")
        self.assertFalse(client.available)
        self.assertEqual(client.chat([]).text, "")

    def test_unknown_provider_raises(self):
        from yaobi_harness.llm.base import LLMError

        with self.assertRaises(LLMError):
            build_client("not-a-provider")

    def test_each_supported_provider_builds_with_credentials(self):
        azure = build_client("azure", api_key="k", model="dep", base_url="https://example.openai.azure.com")
        self.assertIn("/openai/deployments/dep/chat/completions", azure.endpoint())
        self.assertEqual(azure.headers()["api-key"], "k")

        poe = build_client("poe", api_key="k", model="Claude-Sonnet-4.5")
        self.assertEqual(poe.endpoint(), "https://api.poe.com/v1/chat/completions")
        self.assertTrue(poe.headers()["Authorization"].startswith("Bearer "))

        minimax = build_client("minimax", api_key="k", model="MiniMax-Text-01", group_id="g1")
        self.assertIn("/text/chatcompletion_v2", minimax.endpoint())
        self.assertIn("GroupId=g1", minimax.endpoint())

        litellm = build_client("litellm", api_key="k", model="gpt-4o", base_url="http://gateway:4000")
        self.assertEqual(litellm.endpoint(), "http://gateway:4000/v1/chat/completions")

    def test_missing_credentials_raise_rather_than_silently_downgrading(self):
        from yaobi_harness.llm.base import LLMError

        with self.assertRaises(LLMError):
            build_client("poe", api_key="")

    def test_response_parsing_handles_tool_calls_and_fenced_json(self):
        client = build_client("poe", api_key="k", model="m")
        parsed = client.parse_response({
            "choices": [{"message": {"content": "```json\n{\"a\": 1}\n```",
                                      "tool_calls": [{"function": {"name": "t", "arguments": "{\"x\": 2}"}}]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        })
        self.assertEqual(parsed.json(), {"a": 1})
        self.assertEqual(parsed.tool_calls[0].name, "t")
        self.assertEqual(parsed.tool_calls[0].arguments, {"x": 2})
        self.assertEqual(parsed.total_tokens, 12)

    def test_extract_json_tolerates_prose(self):
        self.assertEqual(extract_json('结果如下 {"ok": true} 完毕'), {"ok": True})
        self.assertIsNone(extract_json("no json here"))

    def test_llm_budget_is_enforced(self):
        state = ClinicalRunState("腰痛", role="physician")
        state.budget = Budget(max_llm_calls=1)
        self.assertTrue(state.budget.reserve_llm())
        self.assertFalse(state.budget.reserve_llm())

    def test_runner_defaults_to_deterministic_null_client(self):
        runner = YaobiGraphRunner()
        self.assertIsInstance(runner.llm, NullLLMClient)


# --------------------------------------------------------------------- output

class RenderingTests(unittest.TestCase):
    def test_patient_view_hides_other_patients_records_and_evidence_ledger(self):
        registry = ToolRegistry(records=[RAW_RECORD], deid_key=TEST_KEY)
        out = YaobiGraphRunner(registry).run(ClinicalRunState("腰痛3月，久坐加重", role="patient"))
        self.assertTrue(out.outputs["expert_cases"]["similar"], "fixture should retrieve a case")
        view = render(out, "patient")
        blob = json.dumps(view, ensure_ascii=False)
        self.assertNotIn("research_patient_id", blob)
        self.assertNotIn("evidence_ledger", view)
        self.assertIn("disclaimer", view)

    def test_physician_view_exposes_ledger_and_audit(self):
        out = YaobiGraphRunner().run(ClinicalRunState("腰痛3月，久坐加重", role="physician"))
        view = render(out, "physician")
        self.assertIn("evidence_ledger", view)
        self.assertIn("safety_audit", view)
        self.assertTrue(any(item["level"] == "stub_not_for_clinical_use" for item in view["evidence_ledger"]))

    def test_researcher_view_returns_aggregates_only(self):
        registry = ToolRegistry(records=[RAW_RECORD], deid_key=TEST_KEY)
        out = YaobiGraphRunner(registry).run(ClinicalRunState("腰痛3月，久坐加重", role="researcher"))
        view = render(out, "researcher")
        self.assertIn("counts", view)
        self.assertNotIn("expert_cases", view)

    def test_debug_view_returns_full_state(self):
        out = YaobiGraphRunner().run(ClinicalRunState("腰痛", role="patient"))
        self.assertIn("evidence", render(out, debug=True))


if __name__ == "__main__":
    unittest.main()
