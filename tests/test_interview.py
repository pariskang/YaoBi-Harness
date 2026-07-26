"""Tests for model-driven history taking.

The properties worth pinning here are the containment ones: a model may word and
order the enquiry, but it cannot skip a required axis, smuggle advice into a
question, emit a dose, or declare itself finished while a red-flag axis is open.
"""

from __future__ import annotations

import unittest

from yaobi_harness.conversation import ConversationSession, rule_extract
from yaobi_harness.interview import axes as ax
from yaobi_harness.interview.adequacy import (
    ACHIEVED, BLOCKED, CAP_REACHED, NOT_ACHIEVED, STALLED, AdequacyJudge,
)
from yaobi_harness.interview.loop import InterviewLoop
from yaobi_harness.llm.base import LLMResponse, ToolCall
from yaobi_harness.state import Budget

RED_FLAGS_ANSWERED = {
    "bowel_bladder": "否认", "neuro_symptoms": "否认",
    "fever_trauma_tumor": "否认", "night_pain": "否认", "limb_vascular": "否认",
}
CORE_ANSWERED = {
    "onset": "3个月", "pain_location": "腰", "radiation": "无",
    "medications_confirmed": True, "allergies_confirmed": True,
    "age": 63, "pregnancy": False, "renal": "normal", "liver": "normal",
}


class FakeLLM:
    """Returns a scripted sequence of responses; records the prompts it saw."""

    name = "fake"
    model = "fake-1"
    available = True

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, tools=None, temperature=0.0, max_tokens=1024, response_format_json=False):
        self.calls.append(messages)
        return self.responses.pop(0) if self.responses else LLMResponse(text="")


def ask_call(questions, complete=False, reasoning=""):
    return LLMResponse(tool_calls=[ToolCall(
        "ask_patient",
        {"questions": questions, "interview_complete": complete, "reasoning": reasoning},
        "c1",
    )])


class AxisTaxonomyTests(unittest.TestCase):
    def test_every_tier_is_populated_and_ids_are_unique(self):
        ids = [a.axis_id for a in ax.AXES]
        self.assertEqual(len(ids), len(set(ids)))
        for tier in ax.TIERS:
            self.assertTrue([a for a in ax.AXES if a.tier == tier], f"tier {tier} is empty")

    def test_ten_questions_song_is_actually_covered(self):
        """十问歌 is a claim the code has to back up, not a docstring flourish."""
        song = {a.axis_id for a in ax.AXES if a.tradition == "十问歌"}
        for expected in (
            "cold_heat", "sweating", "head_body", "bowel_urine_tcm", "appetite_diet",
            "chest_abdomen", "hearing_thirst", "past_history_cause", "medication_history",
            "menstruation",
        ):
            self.assertIn(expected, song)

    def test_every_axis_declares_probes_and_a_rationale(self):
        for axis in ax.AXES:
            with self.subTest(axis=axis.axis_id):
                self.assertTrue(axis.probes, "an axis with no probes cannot be asked deterministically")
                self.assertTrue(axis.rationale, "an axis with no rationale cannot explain itself")
                self.assertTrue(axis.closes, "an axis that closes no fact can never be satisfied")
                self.assertIn(axis.tier, ax.TIERS)

    def test_axis_needs_all_of_its_facts_not_just_one(self):
        axis = ax.AXES_BY_ID["special_population"]
        self.assertFalse(axis.satisfied({"age": 63}))
        self.assertTrue(axis.satisfied(
            {"age": 63, "pregnancy": False, "renal": "normal", "liver": "normal"}))

    def test_special_population_is_read_from_the_nested_dict_too(self):
        facts = {"special_population": {"age": 63, "pregnancy": False, "renal": "n", "liver": "n"}}
        self.assertTrue(ax.AXES_BY_ID["special_population"].satisfied(facts))

    def test_false_counts_as_an_answer(self):
        """``pregnancy: False`` is an answer; a falsy check would call it missing."""
        self.assertTrue(ax.AXES_BY_ID["special_population"].satisfied(
            {"age": 1, "pregnancy": False, "renal": "n", "liver": "n"}))

    def test_menstruation_axis_is_scoped_to_who_can_answer_it(self):
        relevant = lambda facts: any(  # noqa: E731
            a.axis_id == "menstruation" for a in ax.relevant_axes(facts, "腰痛"))
        self.assertTrue(relevant({"sex": "女", "age": 35}))
        self.assertFalse(relevant({"sex": "男", "age": 35}))

    def test_fragility_axis_appears_for_older_patients(self):
        applies = lambda facts, text: any(  # noqa: E731
            a.axis_id == "bone_fragility" for a in ax.relevant_axes(facts, text))
        self.assertTrue(applies({"age": 68}, "腰痛"))
        self.assertTrue(applies({}, "腰痛，医生说有骨质疏松"))
        self.assertFalse(applies({"age": 28}, "腰痛"))

    def test_urgent_mode_requires_only_red_flags(self):
        required = ax.required_open_axes({}, "突发胸痛", risk_mode="urgent")
        self.assertTrue(required)
        self.assertEqual({a.tier for a in required}, {"RED_FLAG"})

    def test_prescriptive_runs_additionally_require_four_diagnoses(self):
        facts = {**RED_FLAGS_ANSWERED, **CORE_ANSWERED}
        self.assertFalse(ax.required_open_axes(facts, "腰痛3个月", role="physician"))
        prescriptive = ax.required_open_axes(facts, "腰痛3个月", role="physician", prescriptive=True)
        self.assertIn("tongue_pulse", {a.axis_id for a in prescriptive})

    def test_plan_puts_required_axes_first(self):
        plan = ax.plan_next({}, "腰痛3个月", limit=3)
        self.assertTrue(plan.axis_ids)
        for axis_id in plan.axis_ids:
            self.assertEqual(ax.AXES_BY_ID[axis_id].tier, "RED_FLAG")

    def test_coverage_rises_as_facts_arrive(self):
        empty = ax.coverage({}, "腰痛3个月")["ratio"]
        partial = ax.coverage(RED_FLAGS_ANSWERED, "腰痛3个月")["ratio"]
        self.assertEqual(empty, 0.0)
        self.assertGreater(partial, empty)

    def test_a_broken_predicate_keeps_the_axis_rather_than_hiding_it(self):
        """Failing safe here means *showing* an axis, never silently dropping it."""
        axis = ax.Axis("boom", "爆", "RED_FLAG", "t", closes=("x",), probes=("?",),
                       applies_when=lambda facts, complaint: 1 / 0)
        self.assertTrue(axis.relevant({}, "腰痛"))


class AdequacyJudgeTests(unittest.TestCase):
    def test_open_required_axes_block(self):
        verdict = AdequacyJudge().judge({}, "腰痛3个月")
        self.assertEqual(verdict.verdict, NOT_ACHIEVED)
        self.assertTrue(verdict.blocking_axes)
        self.assertFalse(verdict.may_proceed)

    def test_closed_axes_reach_achieved(self):
        verdict = AdequacyJudge().judge({**RED_FLAGS_ANSWERED, **CORE_ANSWERED}, "腰痛3个月")
        self.assertEqual(verdict.verdict, ACHIEVED)
        self.assertTrue(verdict.may_proceed)
        self.assertFalse(verdict.deficit)

    def test_three_identical_rounds_with_no_blockers_is_a_stall(self):
        """A gap the patient keeps not answering stops the loop rather than spinning."""
        judge = AdequacyJudge(stall_threshold=3)
        # A verifier that keeps naming the same non-required gap: nothing blocks
        # release, but no progress is being made either.
        judge._ask_model = lambda *a, **k: (["tongue_pulse"], "还缺舌脉", [])  # type: ignore[method-assign]
        facts = {**RED_FLAGS_ANSWERED, **CORE_ANSWERED}
        verdicts = [judge.judge(facts, "腰痛3个月", rounds_used=i).verdict for i in range(3)]
        self.assertEqual(verdicts[:2], [NOT_ACHIEVED, NOT_ACHIEVED])
        self.assertEqual(verdicts[2], STALLED)
        self.assertTrue(judge.judge(facts, "腰痛3个月", rounds_used=3).may_proceed)

    def test_two_identical_rounds_are_not_yet_a_stall(self):
        """Patients answer one thing at a time; two rounds is one bad exchange."""
        judge = AdequacyJudge()
        judge.judge({}, "腰痛3个月")
        verdict = judge.judge({}, "腰痛3个月")
        self.assertEqual(verdict.verdict, NOT_ACHIEVED)

    def test_repeated_rounds_with_open_required_axes_end_in_blocked(self):
        judge = AdequacyJudge()
        for _ in range(3):
            verdict = judge.judge({}, "腰痛3个月")
        self.assertEqual(verdict.verdict, BLOCKED)
        self.assertFalse(verdict.may_proceed)
        self.assertTrue(verdict.blocking_axes)

    def test_round_cap_with_no_blockers_proceeds_with_a_recorded_deficit(self):
        judge = AdequacyJudge(max_rounds=1, stall_threshold=99)
        facts = {**RED_FLAGS_ANSWERED, **CORE_ANSWERED, "conditions": []}
        verdict = judge.judge(facts, "腰痛3个月", rounds_used=5)
        # Everything required is closed, so the cap has nothing to fire on.
        self.assertEqual(verdict.verdict, ACHIEVED)

    def test_cap_reached_when_optional_gaps_remain(self):
        judge = AdequacyJudge(max_rounds=2, stall_threshold=99)
        judge._ask_model = lambda *a, **k: (["tongue_pulse"], "还缺舌脉", [])  # type: ignore[method-assign]
        verdict = judge.judge({**RED_FLAGS_ANSWERED, **CORE_ANSWERED}, "腰痛3个月", rounds_used=9)
        self.assertEqual(verdict.verdict, CAP_REACHED)
        self.assertTrue(verdict.may_proceed)
        self.assertTrue(verdict.deficit)

    def test_model_may_add_gaps_but_never_remove_a_required_one(self):
        judge = AdequacyJudge()
        judge._ask_model = lambda *a, **k: ([], "看起来够了", [])  # type: ignore[method-assign]
        verdict = judge.judge({}, "腰痛3个月")
        self.assertNotEqual(verdict.verdict, ACHIEVED)
        self.assertTrue(verdict.blocking_axes)

    def test_model_named_gaps_are_unioned_in(self):
        judge = AdequacyJudge()
        judge._ask_model = lambda *a, **k: (["tongue_pulse"], "缺舌脉", [])  # type: ignore[method-assign]
        verdict = judge.judge({**RED_FLAGS_ANSWERED, **CORE_ANSWERED}, "腰痛3个月")
        self.assertIn("tongue_pulse", verdict.missing_axes)
        self.assertEqual(verdict.verdict, NOT_ACHIEVED)

    def test_unknown_axis_ids_from_the_model_are_dropped(self):
        judge = AdequacyJudge(FakeLLM([LLMResponse(
            text='{"adequate": false, "missing_axes": ["not_a_real_axis"], "reason": "x"}')]))
        verdict = judge.judge({**RED_FLAGS_ANSWERED, **CORE_ANSWERED}, "腰痛", budget=Budget())
        self.assertNotIn("not_a_real_axis", verdict.missing_axes)

    def test_a_verifier_outage_falls_back_to_the_rule_verdict(self):
        class Boom:
            name, model, available = "boom", "b", True

            def chat(self, *a, **k):
                raise RuntimeError("down")

        verdict = AdequacyJudge(Boom()).judge({}, "腰痛3个月", budget=Budget())
        self.assertEqual(verdict.judged_by, "rule")
        self.assertTrue(verdict.blocking_axes)

    def test_signature_stall_ignores_an_empty_gap_set(self):
        judge = AdequacyJudge()
        judge.history = [frozenset(), frozenset(), frozenset()]
        self.assertFalse(judge._stalled(frozenset()))

    def test_the_verifier_never_sees_a_signature_or_a_dose(self):
        llm = FakeLLM([LLMResponse(text='{"adequate": true, "missing_axes": []}')])
        AdequacyJudge(llm).judge(
            {"physician_review": {"signature": "S", "approved": True},
             "prescription_draft": {"herbs": [{"herb": "当归", "grams": 12}]},
             **RED_FLAGS_ANSWERED, **CORE_ANSWERED},
            "腰痛", budget=Budget(),
        )
        blob = str(llm.calls)
        self.assertNotIn("physician_review", blob)
        self.assertNotIn("grams", blob)


class InterviewLoopTests(unittest.TestCase):
    def test_without_a_model_the_probe_bank_still_asks_real_questions(self):
        loop = InterviewLoop()
        result = loop.next_round({}, "腰痛3个月")
        self.assertEqual(result.composer, "probe_bank")
        self.assertTrue(result.questions)
        for question in result.questions:
            self.assertIn(question.axis_id, ax.AXES_BY_ID)
            self.assertTrue(question.question)

    def test_red_flags_come_first(self):
        result = InterviewLoop().next_round({}, "腰痛3个月")
        self.assertEqual(result.questions[0].tier, "RED_FLAG")

    def test_probes_are_not_repeated_across_rounds(self):
        loop = InterviewLoop()
        first = {q.question for q in loop.next_round({}, "腰痛3个月").questions}
        second = {q.question for q in loop.next_round({}, "腰痛3个月").questions}
        self.assertFalse(first & second)

    def test_model_questions_are_accepted_when_well_formed(self):
        llm = FakeLLM([ask_call([
            {"axis_id": "cauda_equina", "question": "这几天小便还顺畅吗？", "why": "排除马尾"},
            {"axis_id": "progressive_neuro", "question": "腿劲儿有变化吗？"},
        ])])
        result = InterviewLoop(llm, judge=AdequacyJudge()).next_round({}, "腰痛", budget=Budget())
        self.assertEqual(result.composer, "llm")
        self.assertIn("这几天小便还顺畅吗？", [q.question for q in result.questions])
        self.assertEqual([q.origin for q in result.questions][0], "llm")

    def test_a_question_carrying_a_dose_is_rejected(self):
        llm = FakeLLM([ask_call([
            {"axis_id": "cauda_equina", "question": "要不要先吃布洛芬 0.3g？"},
        ])])
        result = InterviewLoop(llm, judge=AdequacyJudge()).next_round({}, "腰痛", budget=Budget())
        self.assertTrue(any("剂量" in r for r in result.rejected))
        self.assertNotIn("要不要先吃布洛芬 0.3g？", [q.question for q in result.questions])

    def test_a_question_carrying_treatment_advice_is_rejected(self):
        llm = FakeLLM([ask_call([
            {"axis_id": "cauda_equina", "question": "建议你服用止痛药，能接受吗？"},
        ])])
        result = InterviewLoop(llm, judge=AdequacyJudge()).next_round({}, "腰痛", budget=Budget())
        self.assertTrue(any("建议" in r for r in result.rejected))

    def test_an_unknown_axis_is_rejected(self):
        llm = FakeLLM([ask_call([{"axis_id": "astrology", "question": "你什么星座？"}])])
        result = InterviewLoop(llm, judge=AdequacyJudge()).next_round({}, "腰痛", budget=Budget())
        self.assertTrue(any("未知问诊轴" in r for r in result.rejected))
        self.assertNotIn("你什么星座？", [q.question for q in result.questions])

    def test_a_skipped_required_axis_is_added_back_from_the_probe_bank(self):
        """The model chooses wording; the rules choose scope."""
        llm = FakeLLM([ask_call([{"axis_id": "sleep", "question": "睡得好吗？"}])])
        result = InterviewLoop(llm, judge=AdequacyJudge()).next_round({}, "腰痛", budget=Budget())
        asked = {q.axis_id for q in result.questions}
        self.assertIn("cauda_equina", asked)
        self.assertTrue(any("必答轴被模型遗漏" in r for r in result.rejected))

    def test_model_claiming_completion_does_not_end_the_interview(self):
        llm = FakeLLM([ask_call([{"axis_id": "cauda_equina", "question": "小便正常吗？"}], complete=True)])
        result = InterviewLoop(llm, judge=AdequacyJudge()).next_round({}, "腰痛", budget=Budget())
        self.assertTrue(result.model_claimed_complete)
        self.assertEqual(result.verdict.verdict, NOT_ACHIEVED)
        self.assertTrue(result.questions, "a claimed-complete interview still has required axes open")

    def test_a_satisfied_interview_asks_nothing(self):
        result = InterviewLoop().next_round({**RED_FLAGS_ANSWERED, **CORE_ANSWERED}, "腰痛3个月")
        self.assertEqual(result.verdict.verdict, ACHIEVED)
        self.assertEqual(result.questions, [])

    def test_json_reply_is_accepted_when_the_gateway_drops_tool_calls(self):
        llm = FakeLLM([LLMResponse(text='{"questions": ['
                                        '{"axis_id": "cauda_equina", "question": "小便顺畅吗？"}]}')])
        result = InterviewLoop(llm, judge=AdequacyJudge()).next_round({}, "腰痛", budget=Budget())
        self.assertEqual(result.composer, "llm")

    def test_free_prose_is_not_accepted_as_a_round(self):
        llm = FakeLLM([LLMResponse(text="我觉得应该问问他睡得好不好。")])
        result = InterviewLoop(llm, judge=AdequacyJudge()).next_round({}, "腰痛", budget=Budget())
        self.assertEqual(result.composer, "probe_bank")

    def test_exhausted_budget_degrades_to_the_probe_bank(self):
        budget = Budget(max_llm_calls=0)
        result = InterviewLoop(FakeLLM([]), judge=AdequacyJudge()).next_round({}, "腰痛", budget=budget)
        self.assertEqual(result.composer, "probe_bank")
        self.assertTrue(result.questions)

    def test_round_count_is_capped_per_question(self):
        llm = FakeLLM([ask_call([
            {"axis_id": axis_id, "question": f"问题{index}"}
            for index, axis_id in enumerate(list(ax.AXES_BY_ID)[:8])
        ])])
        loop = InterviewLoop(llm, judge=AdequacyJudge())
        result = loop.next_round({}, "腰痛", budget=Budget())
        self.assertLessEqual(len(result.questions), loop.max_questions)

    def test_summary_reports_coverage_and_the_verdict(self):
        loop = InterviewLoop()
        loop.next_round({}, "腰痛3个月")
        summary = loop.summary({}, "腰痛3个月")
        self.assertEqual(summary["rounds_used"], 1)
        self.assertIn("coverage", summary)
        self.assertEqual(summary["verdict"]["verdict"], NOT_ACHIEVED)


class DenialExtractionTests(unittest.TestCase):
    """A denial is an answer. Missing that is what made the interview loop forever."""

    def test_denials_close_their_axis(self):
        facts = rule_extract("大小便正常，没有发烧盗汗，腿没有越来越无力，晚上不会痛醒")
        self.assertEqual(facts["bowel_bladder"], "否认")
        self.assertEqual(facts["neuro_symptoms"], "否认")
        self.assertEqual(facts["fever_trauma_tumor"], "否认")
        self.assertEqual(facts["night_pain"], "否认")

    def test_denials_do_not_read_as_reports(self):
        facts = rule_extract("大小便正常，没有发烧")
        self.assertNotEqual(facts.get("bowel_bladder"), "报告")
        self.assertNotEqual(facts.get("fever_trauma_tumor"), "报告")

    def test_reports_win_over_denials_elsewhere_in_the_message(self):
        facts = rule_extract("腿不麻但越来越无力")
        self.assertEqual(facts["neuro_symptoms"], "报告")

    def test_a_real_symptom_is_still_reported(self):
        facts = rule_extract("这两天突然尿不出来，会阴发麻")
        self.assertEqual(facts["bowel_bladder"], "报告")

    def test_third_party_history_answers_nothing(self):
        self.assertEqual(rule_extract("我父亲有肿瘤"), {})

    def test_past_tumour_history_is_a_positive_answer(self):
        """History suppresses a red *flag* but answers the history *question*."""
        facts = rule_extract("以前查出过肿瘤")
        self.assertEqual(facts["fever_trauma_tumor"], "既往报告")

    def test_fused_denials_are_understood(self):
        facts = rule_extract("腿不麻，晚上不痛，腿不肿")
        self.assertEqual(facts["neuro_symptoms"], "否认")
        self.assertEqual(facts["night_pain"], "否认")
        self.assertEqual(facts["limb_vascular"], "否认")

    def test_a_bare_denial_answers_only_what_was_asked(self):
        facts = rule_extract("都正常", asked_axes=("cauda_equina",))
        self.assertEqual(facts.get("bowel_bladder"), "否认")
        self.assertNotIn("fever_trauma_tumor", facts)

    def test_specialty_answers_are_extracted(self):
        facts = rule_extract("走两百米就得停，早上僵十分钟左右")
        self.assertIn("walking_tolerance", facts)
        self.assertIn("morning_stiffness", facts)

    def test_onset_and_site_are_extracted_from_a_narrative(self):
        facts = rule_extract("腰痛3个月，久坐加重")
        self.assertEqual(facts["onset"], "3个月")
        self.assertIn("腰", facts["pain_location"])


class ConversationInterviewTests(unittest.TestCase):
    def test_coverage_climbs_across_turns(self):
        session = ConversationSession(role="patient")
        ratios = []
        for message in (
            "腰痛3个月，久坐加重",
            "大小便正常，没有发烧，腿不麻，晚上不痛",
            "63岁，没怀孕，肝肾功能正常，在吃布洛芬和华法林，没有过敏",
        ):
            ratios.append(session.send(message).interview["coverage_ratio"])
        self.assertEqual(ratios, sorted(ratios))
        self.assertGreater(ratios[-1], ratios[0])

    def test_questions_carry_their_axis_and_tier(self):
        reply = ConversationSession(role="patient").send("腰痛3个月")
        self.assertTrue(reply.structured_questions)
        for question in reply.structured_questions:
            self.assertIn(question["tier"], ax.TIERS)
            self.assertTrue(question["label"])

    def test_the_loop_is_shared_so_rounds_accumulate(self):
        session = ConversationSession(role="patient")
        session.send("腰痛3个月")
        session.send("久坐加重")
        self.assertEqual(session.interview.rounds_used, 2)

    def test_a_cooperative_denial_does_not_trigger_a_false_emergency(self):
        """The whole feature is worthless if answering "no" calls an ambulance."""
        session = ConversationSession(role="patient")
        session.send("腰痛3个月，久坐加重")
        reply = session.send("大小便正常，没有发烧盗汗，腿没有越来越无力，晚上不会痛醒")
        self.assertNotEqual(reply.risk_mode, "urgent")
        self.assertNotEqual(reply.release_status, "urgent_action_plan")

    def test_a_real_red_flag_still_escalates_mid_conversation(self):
        session = ConversationSession(role="patient")
        session.send("腰痛3个月，久坐加重")
        session.send("大小便正常，没有发烧，腿不麻")
        reply = session.send("这两天突然尿不出来，会阴发麻")
        self.assertEqual(reply.risk_mode, "urgent")
        self.assertEqual(reply.release_status, "urgent_action_plan")
        self.assertTrue(reply.escalated)

    def test_an_achieved_interview_stops_asking(self):
        session = ConversationSession(role="patient")
        for message in (
            "腰痛3个月，久坐加重",
            "大小便正常，没有发烧盗汗，腿不麻，晚上不痛，腿不肿",
            "63岁，没怀孕，肝肾功能正常，在吃布洛芬，没有过敏",
            "痛会往右腿后侧窜到小腿",
        ):
            reply = session.send(message)
        self.assertEqual(reply.interview["verdict"], ACHIEVED)
        self.assertEqual(reply.questions, [])

    def test_new_axis_facts_are_on_the_extraction_allowlist(self):
        from yaobi_harness.conversation import EXTRACTABLE_FACTS

        for axis in ax.AXES:
            for key in axis.closes:
                with self.subTest(axis=axis.axis_id, key=key):
                    self.assertIn(
                        key, EXTRACTABLE_FACTS,
                        f"{axis.axis_id} closes {key!r}, which no message can ever establish",
                    )

    def test_physician_review_is_still_not_extractable(self):
        from yaobi_harness.conversation import EXTRACTABLE_FACTS

        self.assertNotIn("physician_review", EXTRACTABLE_FACTS)


class InterviewNodeTests(unittest.TestCase):
    def test_the_interview_runs_as_a_graph_node(self):
        from yaobi_harness.graph import YaobiGraphRunner
        from yaobi_harness.state import ClinicalRunState

        state = ClinicalRunState(complaint="腰痛3个月，久坐加重", role="patient")
        out = YaobiGraphRunner().run(state)
        interview = out.outputs["interview"]
        self.assertIn("coverage", interview)
        self.assertIn("verdict", interview)
        self.assertTrue(out.open_questions)

    def test_four_diagnoses_are_required_only_when_a_dose_may_follow(self):
        from yaobi_harness.graph import YaobiGraphRunner
        from yaobi_harness.state import ClinicalRunState

        runner = YaobiGraphRunner()
        blocking = {}
        for allow in (False, True):
            state = ClinicalRunState(complaint="腰痛3月，久坐加重", role="physician")
            out = runner.run(state, allow_prescription=allow)
            blocking[allow] = out.outputs["interview"]["verdict"]["blocking_axes"]
        self.assertNotIn("tongue_pulse", blocking[False])
        self.assertIn("tongue_pulse", blocking[True])

    def test_a_blocked_interview_records_a_safety_issue(self):
        from yaobi_harness.graph import YaobiGraphRunner
        from yaobi_harness.interview.loop import InterviewLoop
        from yaobi_harness.state import ClinicalRunState

        loop = InterviewLoop()
        runner = YaobiGraphRunner(interview_loop=loop)
        for _ in range(3):
            state = ClinicalRunState(complaint="腰痛3个月", role="patient")
            out = runner.run(state)
        self.assertEqual(out.outputs["interview"]["verdict"]["verdict"], BLOCKED)
        self.assertTrue(any("blocked" in issue for issue in out.safety_issues))


class ScreeningRegressionTests(unittest.TestCase):
    """The colloquial-denial fix must not weaken true-positive screening."""

    def test_spoken_denial_no_longer_reads_as_a_positive(self):
        from yaobi_harness.safety.red_flags import screen

        self.assertFalse(screen("大小便正常，晚上不会痛醒").hits)

    def test_inability_is_not_read_as_a_denial(self):
        from yaobi_harness.safety.red_flags import screen

        self.assertTrue(screen("现在不会排尿了").hits)

    def test_a_denied_corroborator_no_longer_promotes_a_soft_signal(self):
        from yaobi_harness.safety.red_flags import screen

        result = screen("腰痛，夜间痛，没有发热，没有外伤，没有体重下降")
        self.assertFalse(result.hits, "a denial must not corroborate the thing it denies")
        self.assertTrue(result.soft_hits)

    def test_a_present_corroborator_still_promotes(self):
        from yaobi_harness.safety.red_flags import screen

        self.assertTrue(screen("腰痛，夜间痛醒，最近体重下降").hits)

    def test_contrast_still_preserves_the_live_symptom(self):
        from yaobi_harness.safety.red_flags import screen

        self.assertTrue(screen("无高血压病史但今日胸痛").hits)

    def test_present_tense_still_vetoes_history(self):
        from yaobi_harness.safety.red_flags import screen

        self.assertTrue(screen("既往体健，现突发胸痛、大汗").hits)


if __name__ == "__main__":
    unittest.main()
