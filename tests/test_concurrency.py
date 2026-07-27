"""Tests for concurrent consult members and the isolation that makes it safe.

Parallelism is the easy half. The property under test here is that a concurrent
panel produces the **same audit trail** as a sequential one: same evidence ids,
same citations, same budget accounting — because an audit trail that depends on
which HTTP response arrived first is not an audit trail.
"""

from __future__ import annotations

import json
import random
import threading
import time
import unittest
from pathlib import Path

from yaobi_harness.agent.panel import ConsultPanel, _carve_budget, _default_concurrency
from yaobi_harness.agent.scope import MemberScope, merge_scopes
from yaobi_harness.llm.base import LLMResponse, ToolCall
from yaobi_harness.skills.loader import SkillRegistry
from yaobi_harness.state import Budget, ClinicalRunState, EvidenceLevel
from yaobi_harness.tools import ToolHealth, ToolRegistry

MANIFEST = Path(__file__).parent.parent / "yaobi_harness" / "skills" / "manifest.yaml"
ALL_PERSONAS = ["ortho_attending", "pain_specialist", "rehab_specialist",
                "tcm_orthopedist", "clinical_pharmacist"]

OPINION = {
    "urgency": "routine", "key_findings": ["所见"], "concerns": [],
    "recommend_next": ["下一步"], "questions_for_patient": [], "dissent": "",
    "evidence_note": "依据规则包",
}


def case() -> ClinicalRunState:
    state = ClinicalRunState(complaint="腰痛3月，右下肢麻木", role="physician")
    state.facts.update({"medications": ["布洛芬", "华法林"], "conditions": ["elderly"]})
    state.budget = Budget()
    return state


class ToolThenAnswer:
    """Calls one tool, then answers. Jitter makes completion order vary."""

    name, model, available = "jitter", "j-1", True

    def __init__(self, jitter: tuple[float, float] = (0.005, 0.05)) -> None:
        self.jitter = jitter
        self._turns: dict[str, int] = {}
        self._lock = threading.Lock()

    def chat(self, messages, *, tools=None, temperature=0.0, max_tokens=1024, response_format_json=False):
        key = messages[0]["content"][:48]
        with self._lock:
            turn = self._turns.get(key, 0)
            self._turns[key] = turn + 1
        time.sleep(random.uniform(*self.jitter))
        if turn == 0:
            return LLMResponse(
                tool_calls=[ToolCall("clinical_guideline_search", {"topic": "low back pain"}, "c1")],
                prompt_tokens=5, completion_tokens=5,
            )
        return LLMResponse(text=json.dumps(OPINION, ensure_ascii=False), prompt_tokens=5, completion_tokens=5)


class Slow:
    name, model, available = "slow", "s-1", True

    def __init__(self, delay: float = 0.12) -> None:
        self.delay = delay

    def chat(self, messages, **kwargs):
        time.sleep(self.delay)
        return LLMResponse(text=json.dumps(OPINION, ensure_ascii=False), prompt_tokens=5, completion_tokens=5)


class MemberScopeTests(unittest.TestCase):
    def test_reads_fall_through_to_the_parent(self):
        parent = case()
        scope = MemberScope(parent, Budget(), label="骨科")
        self.assertEqual(scope.complaint, parent.complaint)
        self.assertEqual(scope.role, parent.role)
        self.assertEqual(scope.run_id, parent.run_id)

    def test_facts_are_deep_copied_so_siblings_cannot_interfere(self):
        parent = case()
        parent.facts["nested"] = {"a": 1}
        first = MemberScope(parent, Budget(), label="A")
        second = MemberScope(parent, Budget(), label="B")
        first.facts["nested"]["a"] = 99
        self.assertEqual(second.facts["nested"]["a"], 1)
        self.assertEqual(parent.facts["nested"]["a"], 1)

    def test_evidence_stays_private_until_merged(self):
        parent = case()
        scope = MemberScope(parent, Budget(), label="A")
        scope.add_evidence(EvidenceLevel.TOOL.value, "t", "s")
        self.assertEqual(len(scope.evidence), 1)
        self.assertEqual(len(parent.evidence), 0)

    def test_a_member_cannot_escalate_the_run_itself(self):
        """Escalation is the panel's decision, made after synthesis."""
        scope = MemberScope(case(), Budget(), label="A")
        with self.assertRaises(PermissionError):
            scope.risk_mode = "urgent"

    def test_a_member_cannot_set_the_release_status(self):
        scope = MemberScope(case(), Budget(), label="A")
        with self.assertRaises(PermissionError):
            scope.release_status = "approved_by_physician"

    def test_a_member_reads_the_real_risk_posture(self):
        parent = case()
        parent.risk_mode = "urgent"
        self.assertEqual(MemberScope(parent, Budget(), label="A").risk_mode, "urgent")

    def test_fail_closed_becomes_a_recorded_issue_not_a_failed_run(self):
        """One specialist's tool outage must not fail the whole consultation."""
        parent = case()
        scope = MemberScope(parent, Budget(), label="药师")
        scope.fail_closed("工具不可用")
        self.assertEqual(parent.release_status, "needs_more_information")
        self.assertTrue(any("药师" in issue for issue in scope.safety_issues))

    def test_merge_allocates_ids_in_roster_order(self):
        parent = case()
        first = MemberScope(parent, Budget(), label="A")
        second = MemberScope(parent, Budget(), label="B")
        second.add_evidence(EvidenceLevel.TOOL.value, "tb", "from B")
        first.add_evidence(EvidenceLevel.TOOL.value, "ta", "from A")

        reports = merge_scopes(parent, [first, second])
        self.assertEqual([e.summary for e in parent.evidence.values()], ["from A", "from B"])
        self.assertEqual(reports[0].member, "A")
        self.assertEqual(list(reports[0].evidence_remap.values()), ["E0001"])
        self.assertEqual(list(reports[1].evidence_remap.values()), ["E0002"])

    def test_merge_rewrites_claim_and_trace_citations(self):
        parent = case()
        scope = MemberScope(parent, Budget(), label="A")
        local = scope.add_evidence(EvidenceLevel.TOOL.value, "t", "s")
        scope.add_claim("differential", "腰椎管狭窄", [local])
        scope.trace("A", "act", evidence_ids=[local])

        merge_scopes(parent, [scope])
        new_id = next(iter(parent.evidence))
        self.assertEqual(parent.claims[0].evidence_ids, [new_id])
        self.assertEqual(parent.traces[0].evidence_ids, [new_id])

    def test_merge_preserves_evidence_grade_and_failure(self):
        parent = case()
        scope = MemberScope(parent, Budget(), label="A")
        scope.add_evidence(EvidenceLevel.GUIDELINE.value, "t", "ok")
        scope.add_evidence(EvidenceLevel.GUIDELINE.value, "t", "bad", ok=False, error="boom")
        merge_scopes(parent, [scope])
        levels = [e.level for e in parent.evidence.values()]
        self.assertEqual(levels, [EvidenceLevel.GUIDELINE.value, EvidenceLevel.FAILED.value])

    def test_merge_carries_warnings_and_issues(self):
        parent = case()
        scope = MemberScope(parent, Budget(), label="A")
        scope.warn("注意")
        scope.fail_closed("坏了")
        merge_scopes(parent, [scope])
        self.assertIn("注意", parent.warnings)
        self.assertTrue(any("坏了" in issue for issue in parent.safety_issues))


class BudgetConcurrencyTests(unittest.TestCase):
    def test_reserve_llm_never_exceeds_the_ceiling_under_contention(self):
        """Without the lock, threads both pass the check and both spend."""
        budget = Budget(max_llm_calls=50)
        granted = []
        lock = threading.Lock()

        def worker():
            for _ in range(30):
                if budget.reserve_llm():
                    with lock:
                        granted.append(1)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(granted), 50)
        self.assertEqual(budget.used_llm_calls, 50)

    def test_charge_tool_counts_every_call_under_contention(self):
        budget = Budget(max_tool_calls=10_000)

        def worker():
            for _ in range(200):
                budget.charge_tool()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(budget.used_tool_calls, 1600)

    def test_refund_returns_a_reservation(self):
        budget = Budget()
        budget.reserve_llm()
        budget.refund_llm()
        self.assertEqual(budget.used_llm_calls, 0)

    def test_refund_never_goes_negative(self):
        budget = Budget()
        budget.refund_llm()
        self.assertEqual(budget.used_llm_calls, 0)

    def test_a_slice_refunds_when_the_parent_refuses(self):
        """A partial acquisition would slowly drain a ceiling nobody spent."""
        parent = Budget(max_llm_calls=0)
        chargeback = _carve_budget(parent, 2)
        self.assertFalse(chargeback.reserve_llm())
        self.assertEqual(chargeback._slice.used_llm_calls, 0)

    def test_the_lock_is_not_serialised_into_a_checkpoint(self):
        from dataclasses import asdict

        self.assertNotIn("_lock", asdict(Budget()))
        state = case()
        rebuilt = ClinicalRunState.from_dict(state.to_dict())
        self.assertTrue(rebuilt.budget.reserve_llm())


class ToolHealthConcurrencyTests(unittest.TestCase):
    def test_two_failures_open_the_circuit_even_when_concurrent(self):
        health = ToolHealth(failure_threshold=2)
        barrier = threading.Barrier(2)

        def worker():
            barrier.wait()
            health.record_failure("t")

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertFalse(health.is_healthy("t"))
        self.assertEqual(health.consecutive_failures["t"], 2)


class PanelConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.registry = SkillRegistry.discover(MANIFEST)

    def _run(self, workers, seed=1, llm=None):
        random.seed(seed)
        state = case()
        panel = ConsultPanel(llm or ToolThenAnswer(), concurrency=workers)
        result = panel.run(state, ToolRegistry(), self.registry, personas=ALL_PERSONAS)
        return state, result

    def test_all_members_produce_an_opinion(self):
        _, result = self._run(5)
        self.assertEqual(result.mode, "panel")
        self.assertEqual(len(result.opinions), 5)
        self.assertTrue(all(o.ok for o in result.opinions))
        self.assertEqual(result.concurrency, 5)

    def test_opinions_come_back_in_convened_order(self):
        _, result = self._run(5)
        self.assertEqual([o.persona for o in result.opinions], ALL_PERSONAS)

    def test_the_evidence_ledger_is_identical_to_the_sequential_run(self):
        sequential_state, sequential = self._run(1, seed=1)
        concurrent_state, concurrent = self._run(5, seed=2)

        def summarise(state):
            return [(eid, e.source, e.level) for eid, e in state.evidence.items()]

        self.assertEqual(summarise(sequential_state), summarise(concurrent_state))
        self.assertEqual([(o.persona, tuple(o.evidence_ids)) for o in sequential.opinions],
                         [(o.persona, tuple(o.evidence_ids)) for o in concurrent.opinions])

    def test_the_ledger_is_stable_across_differently_jittered_runs(self):
        """Different completion orders must still give one ledger."""
        ledgers = []
        for seed in range(4):
            state, _ = self._run(5, seed=seed + 10)
            ledgers.append([(eid, e.source) for eid, e in state.evidence.items()])
        self.assertEqual(len(set(map(tuple, ledgers))), 1, f"ledgers diverged: {ledgers}")

    def test_budget_accounting_matches_the_sequential_run(self):
        sequential_state, _ = self._run(1, seed=1)
        concurrent_state, _ = self._run(5, seed=2)
        self.assertEqual(sequential_state.budget.used_llm_calls, concurrent_state.budget.used_llm_calls)
        self.assertEqual(sequential_state.budget.used_tool_calls, concurrent_state.budget.used_tool_calls)

    def test_concurrency_actually_overlaps_the_waiting(self):
        start = time.time()
        self._run(5, llm=Slow(0.12))
        concurrent = time.time() - start

        start = time.time()
        self._run(1, llm=Slow(0.12))
        sequential = time.time() - start

        self.assertLess(concurrent, sequential * 0.6,
                        f"concurrent {concurrent:.2f}s vs sequential {sequential:.2f}s")

    def test_one_failing_member_does_not_take_down_the_panel(self):
        class OneBad(ToolThenAnswer):
            def chat(self, messages, **kwargs):
                # The persona system prompt opens with the member's own label.
                if messages[0]["content"].startswith("你是骨科多学科会诊中的**临床药师"):
                    raise RuntimeError("member down")
                return super().chat(messages, **kwargs)

        _, result = self._run(5, llm=OneBad())
        self.assertEqual(result.mode, "panel")
        failed = [o for o in result.opinions if not o.ok]
        self.assertEqual(len(failed), 1)
        self.assertEqual(len(result.opinions), 5)

    def test_the_merge_record_is_reported(self):
        _, result = self._run(5)
        self.assertEqual(len(result.merge), 5)
        self.assertTrue(all("evidence_remap" in entry for entry in result.merge))

    def test_a_member_cannot_escalate_the_run_mid_flight(self):
        state, result = self._run(5)
        self.assertEqual(state.risk_mode, "routine")

    def test_the_sequential_path_reports_concurrency_of_one(self):
        _, result = self._run(1)
        self.assertEqual(result.concurrency, 1)
        self.assertEqual(result.merge, [], "the sequential path writes directly, so nothing is merged")

    def test_a_single_member_panel_needs_no_threads(self):
        state = case()
        result = ConsultPanel(ToolThenAnswer(), concurrency=5).run(
            state, ToolRegistry(), self.registry, personas=["ortho_attending"])
        self.assertEqual(result.concurrency, 1)


class ConsoleConcurrencyTests(unittest.TestCase):
    """The console is a threading server; its shared state must survive that."""

    def _service(self):
        from yaobi_harness.ui.server import ConsoleService

        return ConsoleService()

    def test_session_creation_and_eviction_survive_a_burst(self):
        """``pop(next(iter(...)))`` raises if another thread inserts mid-iteration."""
        from yaobi_harness.ui.server import MAX_SESSIONS

        service = self._service()
        errors: list[str] = []
        lock = threading.Lock()

        def worker():
            try:
                service.chat({"message": "腰痛3个月", "role": "patient"})
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=worker) for _ in range(MAX_SESSIONS + 12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertLessEqual(len(service.sessions), MAX_SESSIONS)

    def test_concurrent_sessions_do_not_mix_their_facts(self):
        service = self._service()
        first = service.chat({"message": "腰痛3个月，久坐加重", "role": "patient"})
        second = service.chat({"message": "颈痛伴手麻2周", "role": "patient"})
        self.assertNotEqual(first["session_id"], second["session_id"])
        follow = service.chat({"message": "63岁", "role": "patient", "session_id": first["session_id"]})
        self.assertEqual(follow["reply"]["known_facts"].get("age"), 63)
        other = service.chat({"message": "还有别的吗", "role": "patient", "session_id": second["session_id"]})
        self.assertIsNone(other["reply"]["known_facts"].get("age"))

    def test_the_accept_queue_is_deep_enough_for_a_burst(self):
        """A default backlog of 5 makes a queued request look like a crash."""
        from yaobi_harness.ui.server import _ConsoleServer

        self.assertGreaterEqual(_ConsoleServer.request_queue_size, 64)
        self.assertTrue(_ConsoleServer.daemon_threads)


class SharedRegistryTests(unittest.TestCase):
    def test_the_expert_profile_is_mined_once_under_contention(self):
        """Five members racing a cold cache must not each mine the whole corpus."""
        records = [
            {"性别": "女", "年龄": "63", "中医诊断": "腰痹病 寒湿型",
             "中药": "独活 1克/9克/用法:水煎服", "主诉": "腰痛"}
            for _ in range(6)
        ]
        registry = ToolRegistry(records=records, deid_key="test-key")
        built = []
        real = registry.expert_profile

        import yaobi_harness.expert.profile as profile_module

        original = profile_module.build_profile

        def counting(*args, **kwargs):
            built.append(1)
            return original(*args, **kwargs)

        profile_module.build_profile = counting
        try:
            threads = [threading.Thread(target=real) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            profile_module.build_profile = original
        self.assertEqual(len(built), 1, f"mined {len(built)} times")


class ConcurrencyConfigTests(unittest.TestCase):
    def test_the_environment_can_force_sequential(self):
        import os

        original = os.environ.get("YAOBI_PANEL_CONCURRENCY")
        try:
            os.environ["YAOBI_PANEL_CONCURRENCY"] = "1"
            self.assertEqual(_default_concurrency(), 1)
            os.environ["YAOBI_PANEL_CONCURRENCY"] = "99"
            self.assertLessEqual(_default_concurrency(), 8)
            os.environ["YAOBI_PANEL_CONCURRENCY"] = "not-a-number"
            self.assertGreaterEqual(_default_concurrency(), 1)
        finally:
            if original is None:
                os.environ.pop("YAOBI_PANEL_CONCURRENCY", None)
            else:
                os.environ["YAOBI_PANEL_CONCURRENCY"] = original

    def test_an_explicit_setting_reaches_the_panel_agent(self):
        """Per-caller, not per-process: the console serves several operators."""
        from yaobi_harness.graph import YaobiGraphRunner

        runner = YaobiGraphRunner(panel_concurrency=2)
        self.assertEqual(runner.panel_concurrency, 2)
        self.assertEqual(runner.agents["ConsultPanelAgent"].concurrency, 2)

    def test_the_journal_records_the_effective_setting_not_a_placeholder(self):
        """A replay is only faithful against a matching concurrency, so the meta
        line has to carry the number actually used — never an unresolved None."""
        import tempfile

        from yaobi_harness.graph import YaobiGraphRunner
        from yaobi_harness.journal import Journal

        with tempfile.TemporaryDirectory() as tmp:
            journal = Journal(Path(tmp) / "run.jsonl", mode="record")
            YaobiGraphRunner(journal=journal, panel_concurrency=3)
            self.assertEqual(journal.meta["panel_concurrency"], 3)
            journal_default = Journal(Path(tmp) / "default.jsonl", mode="record")
            YaobiGraphRunner(journal=journal_default)
            self.assertIsInstance(journal_default.meta["panel_concurrency"], int)


class WaveSchedulingTests(unittest.TestCase):
    """Independent graph tasks run together, and it must not show in the result.

    On a routine plan, 西医鉴别 / 辨证 / 病例检索 / 用药安全 read nothing of each
    other's output — eight sequential model calls that had no reason to be
    sequential. Collapsing them is worth several minutes of a patient's wait with
    a reasoning model. It is only worth it if the ledger is unchanged, which is
    what these tests are for.
    """

    ANSWERS = {
        "BiomedicalAgent": {"differentials": ["腰椎间盘突出"], "exam_advice": ["直腿抬高"]},
        "TCMPatternAgent": {"primary_pattern": "气滞血瘀证", "evidence": ["刺痛固定"],
                            "differential_patterns": ["寒湿痹阻证"], "counter_evidence_needed": ["舌脉"]},
        "ExpertCaseAgent": {"similar": ["同证型12例"], "counterexamples": [], "limitation": "单一专家经验"},
    }

    class Jittery:
        """Random latency, so an ordering race shows up rather than hiding."""

        name, model, available = "jittery", "j", True

        def __init__(self, answers):
            self.answers = answers
            self.spans: list[tuple[str, float, float]] = []
            self.lock = threading.Lock()

        def chat(self, messages, **kwargs):
            import sys

            started = time.monotonic()
            frame, agent = sys._getframe(), "-"
            while frame:
                owner = frame.f_locals.get("self")
                if owner is not None and type(owner).__name__.endswith("Agent"):
                    agent = type(owner).__name__
                    break
                frame = frame.f_back
            time.sleep(random.uniform(0, 0.02))
            with self.lock:
                self.spans.append((agent, started, time.monotonic()))
            payload = dict(self.answers.get(agent) or {})
            payload.setdefault("triage", "routine")
            payload.setdefault("adequate", True)
            payload.setdefault("workup_now", True)
            return LLMResponse(text=json.dumps(payload, ensure_ascii=False), model="j")

    def _run(self, concurrency, seed):
        from yaobi_harness.graph import YaobiGraphRunner

        random.seed(seed)
        llm = self.Jittery(self.ANSWERS)
        runner = YaobiGraphRunner(llm=llm, task_concurrency=concurrency)
        state = ClinicalRunState("腰痛3月，刺痛固定，夜间不痛醒，大小便正常", role="physician")
        runner.run(state, allow_prescription=True)
        return state, llm

    @staticmethod
    def _ledger(state):
        return {
            "release_status": state.release_status,
            "risk_mode": state.risk_mode,
            "tasks": [(t.task_id, t.agent, t.status) for t in state.tasks],
            "evidence": [f"{i}|{e.source}|{e.level}" for i, e in state.evidence.items()],
            "claims": [(c.kind, c.text, tuple(c.evidence_ids)) for c in state.claims],
            "traces": [(t.agent, t.action) for t in state.traces],
            "safety_issues": sorted(state.safety_issues),
        }

    def test_a_parallel_run_produces_the_same_ledger_as_a_sequential_one(self):
        baseline = self._ledger(self._run(1, 0)[0])
        for seed in range(6):
            with self.subTest(seed=seed):
                self.assertEqual(self._ledger(self._run(4, seed)[0]), baseline)

    def test_the_wave_actually_overlaps(self):
        """Without this the test above would pass on a scheduler that never
        parallelised anything."""
        _, llm = self._run(4, 11)
        wave = [s for s in llm.spans if s[0] in ("BiomedicalAgent", "TCMPatternAgent", "ExpertCaseAgent")]
        self.assertTrue(wave, "the workup did not run")
        # Each agent makes several calls, so "all overlap" is the wrong claim.
        # The real one is that at some instant more than one workup agent was in
        # flight — a sequential scheduler can never produce that.
        edges = sorted([(s[1], 1) for s in wave] + [(s[2], -1) for s in wave])
        peak, live = 0, 0
        for _, delta in edges:
            live += delta
            peak = max(peak, live)
        self.assertGreater(peak, 1, "no two workup calls were ever in flight together")

    def test_every_agent_in_a_wave_is_recorded_in_the_shared_autonomy_ledger(self):
        """Output dicts merge key-by-key. Replacing them wholesale kept only the
        last member of the wave, so the run reported two of its four agents as
        never having run autonomously — a false audit trail, not a slow one."""
        state, _ = self._run(4, 3)
        recorded = set(state.outputs.get("autonomy") or {})
        self.assertLessEqual({"BiomedicalAgent", "TCMPatternAgent", "ExpertCaseAgent"}, recorded)

    def test_a_journalled_run_is_never_parallel(self):
        """A journal is an ordered sequence of calls. Recording concurrently
        writes an order that depends on network timing, and replaying it
        concurrently consumes an order the recording never had."""
        import tempfile

        from yaobi_harness.graph import YaobiGraphRunner
        from yaobi_harness.journal import Journal

        with tempfile.TemporaryDirectory() as tmp:
            journal = Journal(Path(tmp) / "run.jsonl", mode="record")
            runner = YaobiGraphRunner(journal=journal, task_concurrency=8)
            self.assertEqual(runner.task_concurrency, 1)

    def test_a_task_that_raises_fails_the_run_rather_than_hanging_the_wave(self):
        from yaobi_harness.graph import YaobiGraphRunner

        runner = YaobiGraphRunner(llm=self.Jittery(self.ANSWERS), task_concurrency=4)

        def explode(state, tools, broker):
            raise RuntimeError("模拟子体崩溃")

        runner.agents["TCMPatternAgent"].run = explode
        state = ClinicalRunState("腰痛3月，刺痛固定，夜间不痛醒，大小便正常", role="patient")
        runner.run(state)
        self.assertEqual(state.release_status, "failed_closed")
        self.assertTrue(any("模拟子体崩溃" in issue for issue in state.safety_issues))


class TaskScopeTests(unittest.TestCase):
    """A graph task may write the release status; a panel member may not."""

    def setUp(self):
        self.parent = ClinicalRunState("腰痛", role="patient")
        self.parent.outputs["autonomy"] = {"IntakeAgent": {"mode": "llm"}}

    def scope(self, label="T1"):
        from yaobi_harness.agent.scope import TaskScope

        return TaskScope(self.parent, Budget(), label=label)

    def test_a_status_write_is_recorded_and_replayed_in_task_order(self):
        from yaobi_harness.agent.scope import merge_task_scopes

        first, second = self.scope("N1"), self.scope("N2")
        first.release_status = "needs_examination"
        second.release_status = "treatment_advice_only"
        merge_task_scopes(self.parent, [first, second])
        self.assertEqual(self.parent.release_status, "treatment_advice_only",
                         "later task in plan order wins, whichever finished first")

    def test_a_task_may_not_rewrite_the_plan_it_is_being_scheduled_from(self):
        """Silently dropping the write at merge would leave a task looking as
        though it had rewritten the plan when it had not."""
        for field_name in ("tasks", "planner_mode"):
            with self.subTest(field=field_name), self.assertRaises(PermissionError):
                setattr(self.scope("N1"), field_name, [])

    def test_a_member_scope_still_refuses_a_status_write(self):
        with self.assertRaises(PermissionError):
            MemberScope(self.parent, Budget(), label="M1").release_status = "approved_by_physician"

    def test_output_dicts_merge_rather_than_replace(self):
        from yaobi_harness.agent.scope import merge_task_scopes

        first, second = self.scope("N1"), self.scope("N2")
        first.outputs["autonomy"]["BiomedicalAgent"] = {"mode": "llm_tool_loop"}
        second.outputs["autonomy"]["TCMPatternAgent"] = {"mode": "llm_tool_loop"}
        merge_task_scopes(self.parent, [first, second])
        self.assertEqual(set(self.parent.outputs["autonomy"]),
                         {"IntakeAgent", "BiomedicalAgent", "TCMPatternAgent"})

    def test_gaps_and_questions_a_task_added_reach_the_parent(self):
        from yaobi_harness.agent.scope import merge_task_scopes

        scope = self.scope("N7")
        scope.missing_information.append("当前用药清单")
        scope.open_questions.append("目前在吃什么药？")
        scope.note("并行任务的备注也要合并回去")
        merge_task_scopes(self.parent, [scope])
        self.assertIn("当前用药清单", self.parent.missing_information)
        self.assertIn("目前在吃什么药？", self.parent.open_questions)
        self.assertIn("并行任务的备注也要合并回去", self.parent.notes)

    def test_a_task_may_fail_the_run_closed(self):
        from yaobi_harness.agent.scope import merge_task_scopes

        scope = self.scope("N1")
        scope.fail_closed("关键工具失败")
        merge_task_scopes(self.parent, [scope])
        self.assertEqual(self.parent.release_status, "failed_closed")
        self.assertIn("关键工具失败", self.parent.safety_issues)


if __name__ == "__main__":
    unittest.main()
