"""Tests for consult subagents and skill discovery.

Two invariants carry most of the weight:

* **No subagent is prescriptive.** Whatever a persona or a site's ``SKILL.md``
  claims, the formula, dose and signature tools stay unreachable from a panel
  member.
* **Synthesis is conservative, not democratic.** One member seeing an emergency
  sets the panel's urgency, and four members disagreeing does not downgrade it.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from yaobi_harness.agent.panel import (
    DEFAULT_PANEL, MAX_DEPTH, PERSONAS, URGENCY_ORDER, ConsultOpinion,
    ConsultPanel, ConsultSubagent, PanelResult, choose_panel,
)
from yaobi_harness.llm.base import LLMResponse
from yaobi_harness.skills.discovery import (
    SkillFileError, parse_skill_markdown, split_frontmatter,
)
from yaobi_harness.skills.loader import SkillManifestError, SkillRegistry
from yaobi_harness.state import Budget, ClinicalRunState
from yaobi_harness.tools import ToolRegistry

MANIFEST = Path(__file__).parent.parent / "yaobi_harness" / "skills" / "manifest.yaml"
PRESCRIPTIVE_TOOLS = {"formula_composition_search", "herb_dose_distribution", "physician_review_submit"}


class ScriptedLLM:
    """Returns one JSON body per call, so each member gets its own opinion."""

    name = "scripted"
    model = "scripted-1"
    available = True

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls = 0

    def chat(self, messages, *, tools=None, temperature=0.0, max_tokens=1024, response_format_json=False):
        self.calls += 1
        payload = self.payloads.pop(0) if self.payloads else {}
        return LLMResponse(text=json.dumps(payload, ensure_ascii=False),
                           prompt_tokens=5, completion_tokens=5)


def opinion(urgency="routine", **kwargs):
    return {
        "urgency": urgency,
        "key_findings": kwargs.get("key_findings", ["所见"]),
        "concerns": kwargs.get("concerns", []),
        "recommend_next": kwargs.get("recommend_next", ["下一步"]),
        "questions_for_patient": kwargs.get("questions_for_patient", []),
        "dissent": kwargs.get("dissent", ""),
        "evidence_note": kwargs.get("evidence_note", "依据规则包"),
    }


class SkillDiscoveryTests(unittest.TestCase):
    def test_frontmatter_and_body_split(self):
        front, body = split_frontmatter("---\nskill-id: a.b\nversion: 1\n---\n# Title\n\ntext")
        self.assertEqual(front["skill-id"], "a.b")
        self.assertTrue(body.startswith("# Title"))

    def test_a_body_only_file_takes_its_id_from_the_directory(self):
        entry = parse_skill_markdown("just prose", default_skill_id="my_skill")
        self.assertEqual(entry["skill_id"], "my_skill")
        self.assertEqual(entry["instructions"], "just prose")

    def test_an_unclosed_fence_is_an_error_not_silent_prose(self):
        """Silently treating it as prose would drop every policy field in it."""
        with self.assertRaises(SkillFileError):
            split_frontmatter("---\nskill-id: a.b\nallowed-tools: [x]\n\nno closing fence")

    def test_invalid_yaml_frontmatter_is_an_error(self):
        with self.assertRaises(SkillFileError):
            split_frontmatter("---\n: : :\n\t bad\n---\nbody")

    def test_a_file_with_no_id_and_no_directory_name_is_an_error(self):
        with self.assertRaises(SkillFileError):
            parse_skill_markdown("---\nversion: 1\n---\nbody")

    def test_kebab_keys_are_normalised(self):
        entry = parse_skill_markdown(
            "---\nskill-id: a.b\nwhen-to-use: 'x'\nallowed-tools: 'p, q'\n"
            "consult-mode: screening\n---\nbody")
        self.assertEqual(entry["when_to_use"], "x")
        self.assertEqual(entry["allowed_tools"], ["p", "q"])
        self.assertEqual(entry["consult_mode"], "screening")

    def test_tools_accept_a_list_or_a_string(self):
        as_list = parse_skill_markdown("---\nskill-id: a\nallowed-tools:\n  - p\n  - q\n---\nb")
        as_string = parse_skill_markdown("---\nskill-id: a\nallowed-tools: p q\n---\nb")
        self.assertEqual(as_list["allowed_tools"], as_string["allowed_tools"])

    def test_the_body_wins_over_a_frontmatter_instructions_key(self):
        entry = parse_skill_markdown("---\nskill-id: a\ninstructions: short\n---\nthe real procedure")
        self.assertEqual(entry["instructions"], "the real procedure")

    def test_discovery_layers_skill_files_over_the_manifest(self):
        registry = SkillRegistry.discover(MANIFEST)
        manifest_only = SkillRegistry.from_file(MANIFEST)
        self.assertGreater(len(registry.specs), len(manifest_only.specs))
        for skill_id in ("yaobi.interview", "yaobi.vision_read",
                         "yaobi.consult_panel", "yaobi.osteoporosis_risk"):
            self.assertIn(skill_id, registry.specs)

    def test_shipped_skill_files_carry_substantial_procedure_text(self):
        registry = SkillRegistry.discover(MANIFEST)
        for skill_id in ("yaobi.interview", "yaobi.vision_read", "yaobi.consult_panel"):
            with self.subTest(skill=skill_id):
                self.assertGreater(len(registry.specs[skill_id].instructions), 1200)

    def test_a_site_directory_overrides_a_shipped_skill(self):
        with TemporaryDirectory() as tmp:
            site = Path(tmp) / "interview"
            site.mkdir()
            (site / "SKILL.md").write_text(
                "---\nskill-id: yaobi.interview\nversion: 9.9.9\n"
                "allowed-tools: interview_axis_lookup\n---\n本院自定问诊流程",
                encoding="utf-8",
            )
            registry = SkillRegistry.discover(MANIFEST, extra_roots=[tmp])
            spec = registry.specs["yaobi.interview"]
            self.assertEqual(spec.version, "9.9.9")
            self.assertEqual(spec.instructions, "本院自定问诊流程")

    def test_an_unknown_consult_mode_is_rejected_rather_than_ignored(self):
        """A typo must not become a silently wider grant."""
        with self.assertRaises(SkillManifestError):
            SkillRegistry.from_entries([{
                "skill_id": "a.b", "version": "1", "consult_mode": "unrestricted"}])

    def test_source_records_where_a_skill_came_from(self):
        registry = SkillRegistry.discover(MANIFEST)
        self.assertTrue(registry.specs["yaobi.interview"].source.endswith("SKILL.md"))
        self.assertIn("manifest", registry.specs["yaobi.dose_generation"].source)

    def test_catalog_exposes_the_source_but_not_the_instructions(self):
        entry = next(c for c in SkillRegistry.discover(MANIFEST).catalog()
                     if c["skill_id"] == "yaobi.interview")
        self.assertEqual(entry["source"], "SKILL.md")
        self.assertNotIn("instructions", entry)


class ConsultModeTests(unittest.TestCase):
    def setUp(self):
        self.registry = SkillRegistry.discover(MANIFEST)
        self.spec = self.registry.specs["yaobi.consult_panel"]

    def test_narrowing_only_never_widening(self):
        for mode in ("evidence_only", "screening", "advisory"):
            with self.subTest(mode=mode):
                effective = set(self.spec.effective_tools(mode))
                self.assertTrue(effective.issubset(set(self.spec.allowed_tools)))

    def test_no_mode_reaches_a_prescriptive_tool(self):
        for mode in ("evidence_only", "screening", "advisory", ""):
            with self.subTest(mode=mode or "default"):
                self.assertFalse(PRESCRIPTIVE_TOOLS & set(self.spec.effective_tools(mode)))

    def test_evidence_only_is_narrower_than_screening(self):
        self.assertLess(len(self.spec.effective_tools("evidence_only")),
                        len(self.spec.effective_tools("screening")))

    def test_a_persona_cannot_grant_a_prescriptive_tool_through_its_mode(self):
        spec = SkillRegistry.from_entries([{
            "skill_id": "rogue", "version": "1",
            "allowed_tools": ["herb_dose_distribution", "clinical_guideline_search"],
            "consult_mode": "advisory",
        }]).specs["rogue"]
        self.assertNotIn("herb_dose_distribution", spec.effective_tools())

    def test_forbidden_tools_are_removed_before_the_mode_applies(self):
        spec = SkillRegistry.from_entries([{
            "skill_id": "x", "version": "1",
            "allowed_tools": ["clinical_guideline_search", "drug_label_lookup"],
            "forbidden_tools": ["drug_label_lookup"],
        }]).specs["x"]
        self.assertNotIn("drug_label_lookup", spec.effective_tools("screening"))


class PanelCompositionTests(unittest.TestCase):
    def test_every_persona_is_well_formed(self):
        for name, profile in PERSONAS.items():
            with self.subTest(persona=name):
                self.assertTrue(profile.get("label"))
                self.assertGreater(len(profile.get("instructions", "")), 80)
                self.assertIn(profile.get("consult_mode"), ("evidence_only", "screening", "advisory"))

    def test_the_pharmacist_is_always_convened(self):
        """Medication safety must not depend on a model noticing a drug list."""
        self.assertIn("clinical_pharmacist", DEFAULT_PANEL)
        state = ClinicalRunState(complaint="腰痛", role="patient")
        self.assertIn("clinical_pharmacist", choose_panel(state))

    def test_radicular_wording_adds_the_pain_specialist(self):
        state = ClinicalRunState(complaint="腰痛伴右腿放射麻木", role="patient")
        self.assertIn("pain_specialist", choose_panel(state))

    def test_four_diagnoses_add_the_tcm_consultant(self):
        state = ClinicalRunState(complaint="腰痛", role="patient")
        state.facts["four_diagnoses"] = "舌淡苔白"
        self.assertIn("tcm_orthopedist", choose_panel(state))

    def test_chronic_wording_adds_rehab(self):
        state = ClinicalRunState(complaint="腰痛反复3年", role="patient")
        self.assertIn("rehab_specialist", choose_panel(state))

    def test_depth_is_capped_at_one(self):
        self.assertEqual(MAX_DEPTH, 1)
        with self.assertRaises(ValueError):
            ConsultSubagent("ortho_attending", None, depth=2)

    def test_an_unknown_persona_is_rejected(self):
        with self.assertRaises(ValueError):
            ConsultSubagent("chiropractor", None)


class SynthesisTests(unittest.TestCase):
    """Conservative-first is the point: safety is not put to a vote."""

    def test_the_highest_urgency_wins_against_a_majority(self):
        result = PanelResult(opinions=[
            ConsultOpinion("a", "A", urgency="routine"),
            ConsultOpinion("b", "B", urgency="routine"),
            ConsultOpinion("c", "C", urgency="routine"),
            ConsultOpinion("d", "D", urgency="emergency", concerns=["马尾风险"]),
        ])
        synthesised = ConsultPanel.synthesise(result)
        self.assertEqual(synthesised.urgency, "emergency")
        self.assertIn("马尾风险", synthesised.concerns)

    def test_disagreement_is_reported_not_resolved_away(self):
        result = ConsultPanel.synthesise(PanelResult(opinions=[
            ConsultOpinion("a", "A", urgency="urgent"),
            ConsultOpinion("b", "B", urgency="routine"),
        ]))
        self.assertIn("1/2", result.agreement)
        self.assertTrue(any("分歧" in d for d in result.dissents))

    def test_unanimity_adds_no_dissent_note(self):
        result = ConsultPanel.synthesise(PanelResult(opinions=[
            ConsultOpinion("a", "A", urgency="routine"),
            ConsultOpinion("b", "B", urgency="routine"),
        ]))
        self.assertEqual(result.urgency, "routine")
        self.assertFalse(result.dissents)

    def test_concerns_are_unioned_across_members(self):
        result = ConsultPanel.synthesise(PanelResult(opinions=[
            ConsultOpinion("a", "A", concerns=["出血风险"]),
            ConsultOpinion("b", "B", concerns=["肾损伤风险", "出血风险"]),
        ]))
        self.assertEqual(sorted(result.concerns), ["出血风险", "肾损伤风险"])

    def test_a_failed_member_does_not_kill_the_panel(self):
        result = ConsultPanel.synthesise(PanelResult(opinions=[
            ConsultOpinion("a", "A", urgency="urgent"),
            ConsultOpinion("b", "B", ok=False, error="timeout"),
        ]))
        self.assertEqual(result.mode, "panel")
        self.assertEqual(result.urgency, "urgent")

    def test_all_members_failing_is_reported_as_such(self):
        result = ConsultPanel.synthesise(PanelResult(opinions=[
            ConsultOpinion("a", "A", ok=False, error="x"),
            ConsultOpinion("b", "B", ok=False, error="y"),
        ]))
        self.assertEqual(result.mode, "all_members_failed")

    def test_an_unknown_urgency_string_clamps_to_routine(self):
        from yaobi_harness.agent.panel import _clamp_urgency

        self.assertEqual(_clamp_urgency("catastrophic"), "routine")
        self.assertEqual(_clamp_urgency(None), "routine")
        for level in URGENCY_ORDER:
            self.assertEqual(_clamp_urgency(level), level)


class PanelRunTests(unittest.TestCase):
    def _state(self, complaint="腰痛3月，右下肢麻木"):
        state = ClinicalRunState(complaint=complaint, role="physician")
        state.facts.update({"medications": ["布洛芬", "华法林"], "conditions": ["elderly"]})
        state.budget = Budget()
        return state

    def test_a_panel_runs_and_records_each_member(self):
        registry = SkillRegistry.discover(MANIFEST)
        llm = ScriptedLLM([opinion("routine"), opinion("urgent", concerns=["出血风险"])])
        result = ConsultPanel(llm).run(
            self._state(), ToolRegistry(), registry, personas=["ortho_attending", "clinical_pharmacist"])
        self.assertEqual(result.mode, "panel")
        self.assertEqual(len(result.opinions), 2)
        self.assertEqual(result.urgency, "urgent")

    def test_without_a_model_the_panel_reports_itself_unavailable(self):
        result = ConsultPanel(None).run(self._state(), ToolRegistry(), SkillRegistry.discover(MANIFEST))
        self.assertEqual(result.mode, "llm_unavailable")
        self.assertEqual(result.opinions, [])

    def test_member_budgets_are_carved_from_the_parent(self):
        state = self._state()
        registry = SkillRegistry.discover(MANIFEST)
        llm = ScriptedLLM([opinion(), opinion(), opinion(), opinion(), opinion()])
        before = state.budget.used_llm_calls
        ConsultPanel(llm).run(state, ToolRegistry(), registry,
                              personas=["ortho_attending", "clinical_pharmacist"])
        # The chargeback keeps the parent's totals honest rather than hiding
        # subagent spend from the run's own ceilings.
        self.assertGreater(state.budget.used_llm_calls, before)

    def test_a_member_cannot_reach_a_tool_outside_its_consult_mode(self):
        from yaobi_harness.agent.panel import _member_broker

        registry = SkillRegistry.discover(MANIFEST)
        state = self._state()
        broker = _member_broker(state, registry, "yaobi.consult_panel", ("clinical_guideline_search",))
        allowed, reason = broker.allow("drug_interaction_check")
        self.assertFalse(allowed)
        self.assertIn("consult_mode_denied", reason)
        self.assertTrue(broker.allow("clinical_guideline_search")[0])

    def test_the_run_role_and_risk_mode_still_apply_inside_a_member(self):
        """The member broker only subtracts; it never bypasses the outer checks."""
        from yaobi_harness.agent.panel import _member_broker

        registry = SkillRegistry.discover(MANIFEST)
        state = ClinicalRunState(complaint="突发胸痛", role="patient")
        state.risk_mode = "urgent"
        state.budget = Budget()
        broker = _member_broker(state, registry, "yaobi.consult_panel",
                               ("clinical_guideline_search", "herb_dose_distribution"))
        self.assertFalse(broker.allow("herb_dose_distribution")[0])

    def test_a_member_is_never_shown_a_signature_or_a_draft(self):
        member = ConsultSubagent("ortho_attending", ScriptedLLM([]))
        state = self._state()
        state.facts["physician_review"] = {"signature": "SECRET", "approved": True}
        context = member._context(state)
        self.assertNotIn("physician_review", context["facts"])
        self.assertNotIn("SECRET", json.dumps(context, ensure_ascii=False))

    def test_a_panel_may_raise_urgency_but_a_routine_verdict_changes_nothing(self):
        from yaobi_harness.agent.agents import ConsultPanelAgent
        from yaobi_harness.tools import CapabilityBroker

        registry = SkillRegistry.discover(MANIFEST)
        state = ClinicalRunState(complaint="腰痛3月", role="physician")
        state.risk_mode = "urgent"  # already urgent by rule
        state.budget = Budget()
        broker = CapabilityBroker("physician", "urgent", budget=state.budget,
                                  skill_registry=registry, active_skill="yaobi.consult_panel")
        agent = ConsultPanelAgent(ScriptedLLM([opinion("routine"), opinion("routine")]))
        agent.run(state, ToolRegistry(), broker)
        self.assertEqual(state.risk_mode, "urgent", "a panel must not de-escalate a rule-based hit")

    def test_a_panel_urgency_escalates_a_routine_run(self):
        from yaobi_harness.agent.agents import ConsultPanelAgent
        from yaobi_harness.tools import CapabilityBroker

        registry = SkillRegistry.discover(MANIFEST)
        state = self._state()
        broker = CapabilityBroker("physician", "routine", budget=state.budget,
                                  skill_registry=registry, active_skill="yaobi.consult_panel")
        agent = ConsultPanelAgent(ScriptedLLM([opinion("emergency", concerns=["疑马尾"]), opinion("routine")]))
        agent.run(state, ToolRegistry(), broker)
        self.assertEqual(state.risk_mode, "urgent")
        self.assertTrue(any("上调紧急度" in w for w in state.warnings))


class PanelPlanTests(unittest.TestCase):
    def test_the_panel_is_off_by_default(self):
        from yaobi_harness.agent.planner import rule_plan

        state = ClinicalRunState(complaint="腰痛3月", role="physician")
        agents = {t.agent for t in rule_plan(state)}
        self.assertNotIn("ConsultPanelAgent", agents)

    def test_enabling_it_schedules_it_after_the_interview(self):
        from yaobi_harness.agent.planner import rule_plan

        state = ClinicalRunState(complaint="腰痛3月", role="physician")
        state.enable_panel = True
        tasks = rule_plan(state)
        panel = next(t for t in tasks if t.agent == "ConsultPanelAgent")
        self.assertIn("T3", panel.depends_on)

    def test_the_vision_node_is_scheduled_only_when_images_are_attached(self):
        from yaobi_harness.agent.planner import rule_plan

        state = ClinicalRunState(complaint="腰痛3月", role="physician")
        self.assertNotIn("VisionAgent", {t.agent for t in rule_plan(state)})
        state.images = [{"kind": "radiograph", "ref": "x.png", "deidentified": True}]
        self.assertIn("VisionAgent", {t.agent for t in rule_plan(state)})

    def test_the_interview_runs_on_the_urgent_path_too(self):
        from yaobi_harness.agent.planner import rule_plan

        state = ClinicalRunState(complaint="突发胸痛", role="patient")
        state.risk_mode = "urgent"
        self.assertIn("InterviewAgent", {t.agent for t in rule_plan(state)})


if __name__ == "__main__":
    unittest.main()
