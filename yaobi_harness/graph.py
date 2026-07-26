"""The run loop: plan → act → observe → repair, with a terminal safety gate.

Guarantees this runner provides regardless of the plan it is given:

* the safety critic runs on **every** path, including failed and skipped ones —
  it is never a dependency-gated task;
* every tool call is brokered against a skill that must exist (fail-closed);
* every agent output is validated against its declared schema;
* state is checkpointed after each node and can be resumed from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import schemas
from .agent.agents import (
    BiomedicalAgent, ConsultPanelAgent, CriticAgent, DoseAgent, ExpertCaseAgent,
    FormulaAgent, IntakeAgent, InterviewAgent, MedicationSafetyAgent,
    OsteoporosisAgent, PhysicianReviewAgent, SummaryAgent, TCMPatternAgent, TimelineAgent,
    UrgentCareAgent, UrgentPlannerAgent, VisionAgent,
)
from .agent.planner import AGENT_CATALOG, PlannerAgent
from .llm.base import NullLLMClient
from .llm.factory import describe_client
from .skills.loader import SkillRegistry
from .state import ClinicalRunState, Task
from .tools import CapabilityBroker, ToolHealth, ToolRegistry

#: Agents that may only run for a physician who explicitly opted in.
PRESCRIPTIVE_AGENTS = {"FormulaAgent", "DoseAgent", "PhysicianReviewAgent"}


def _panel_concurrency() -> int:
    from .agent.panel import _default_concurrency

    return _default_concurrency()


class YaobiGraphRunner:
    """Task-driven state graph runner with bounded self-repair loops."""

    def __init__(
        self,
        tools: ToolRegistry | None = None,
        checkpoint_dir: str | Path | None = None,
        skill_manifest: str | Path | None = None,
        llm: Any | None = None,
        skill_dirs: list[str | Path] | None = None,
        interview_loop: Any | None = None,
        journal: Any | None = None,
        panel_concurrency: int | None = None,
    ) -> None:
        self.tools = tools or ToolRegistry()
        # Resolved here rather than inside the panel, because the journal's meta
        # line records the effective value and a replay is only faithful against
        # a matching one: an unresolved ``None`` in the meta would tell a future
        # replayer nothing.
        self.panel_concurrency = panel_concurrency if panel_concurrency else _panel_concurrency()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        manifest = Path(skill_manifest) if skill_manifest else Path(__file__).parent / "skills" / "manifest.yaml"
        # ``discover`` layers every ``SKILL.md`` over the manifest, so the rich
        # procedure files are policy too, not just prose the model happens to read.
        self.skill_registry = SkillRegistry.discover(manifest, extra_roots=skill_dirs)
        #: Optional call journal. Recording captures every tool and model call;
        #: replaying feeds them back so a past decision can be re-derived offline.
        self.journal = journal
        base_llm = llm or NullLLMClient()
        if self.journal is not None:
            from .journal import JournaledLLM

            self.llm = JournaledLLM(
                base_llm, self.journal,
                # A replay with no configured model must be served entirely from
                # the journal; running out is an error, not a silent degradation
                # to the deterministic path, which would hide the shortfall.
                offline=self.journal.mode == "replay" and not getattr(base_llm, "available", False),
            )
        else:
            self.llm = base_llm
        if self.journal is not None and self.journal.mode == "record":
            # Written before the first entry so a replay can adopt the recorded
            # model name; the name is part of every request's content address, so
            # without it an offline replay diverges on the name alone.
            self.journal.write_meta({
                "llm_model": getattr(base_llm, "model", "none"),
                "llm_provider": getattr(base_llm, "name", "none"),
                "llm_available": bool(getattr(base_llm, "available", False)),
                "panel_concurrency": self.panel_concurrency,
            })
        self.health = ToolHealth()
        self.agents = {
            "TimelineAgent": TimelineAgent(self.llm),
            "IntakeAgent": IntakeAgent(self.llm),
            # The interview keeps one loop across every node and every dialogue
            # turn, so its round history — and therefore stall detection — is
            # continuous rather than reset on each call.
            "InterviewAgent": InterviewAgent(self.llm, loop=interview_loop),
            "VisionAgent": VisionAgent(self.llm),
            "ConsultPanelAgent": ConsultPanelAgent(self.llm, concurrency=self.panel_concurrency),
            "OsteoporosisAgent": OsteoporosisAgent(self.llm),
            "UrgentPlannerAgent": UrgentPlannerAgent(self.llm),
            "UrgentCareAgent": UrgentCareAgent(self.llm),
            "BiomedicalAgent": BiomedicalAgent(self.llm),
            "TCMPatternAgent": TCMPatternAgent(self.llm),
            "ExpertCaseAgent": ExpertCaseAgent(self.llm),
            "MedicationSafetyAgent": MedicationSafetyAgent(self.llm),
            "FormulaAgent": FormulaAgent(self.llm),
            "DoseAgent": DoseAgent(self.llm),
            "PhysicianReviewAgent": PhysicianReviewAgent(self.llm),
            "CriticAgent": CriticAgent(self.llm),
            "SummaryAgent": SummaryAgent(self.llm),
            # Registered for skill/output-contract lookup only; the planner is
            # not in AGENT_CATALOG so it can never be scheduled as a task.
            "PlannerAgent": PlannerAgent(self.llm, self.skill_registry),
        }

    # ----------------------------------------------------------------- plumbing
    def _broker(self, state: ClinicalRunState, agent_name: str) -> CapabilityBroker:
        agent = self.agents.get(agent_name)
        return CapabilityBroker(
            state.role,
            state.risk_mode,
            budget=state.budget,
            skill_registry=self.skill_registry,
            active_skill=getattr(agent, "skill_id", "") or None,
            health=self.health,
            journal=self.journal,
        )

    def _checkpoint(self, state: ClinicalRunState, node: str) -> None:
        if not self.checkpoint_dir:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = state.to_dict()
        payload["_checkpoint"] = {"node": node, "loop_index": state.loop_index}
        (self.checkpoint_dir / f"{state.run_id}.{node}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.checkpoint_dir / f"{state.run_id}.latest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def load_checkpoint(path: str | Path) -> ClinicalRunState:
        """Rebuild a state written by :meth:`_checkpoint` so a run can resume."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload.pop("_checkpoint", None)
        return ClinicalRunState.from_dict(payload)

    def _deps_ok(self, state: ClinicalRunState, task: Task) -> bool:
        status = {t.task_id: t.status for t in state.tasks}
        return all(status.get(dep) == "ok" for dep in task.depends_on)

    def _validate_output(self, state: ClinicalRunState, agent_name: str) -> None:
        """Enforce the skill's ``hard_requirements`` and ``output_schema``."""
        if self.skill_registry is None:
            return
        skill_id = getattr(self.agents.get(agent_name), "skill_id", "")
        spec = self.skill_registry.specs.get(skill_id)
        if spec is None or not spec.output_key or spec.output_key not in state.outputs:
            return
        payload = state.outputs[spec.output_key]
        ok, problems = self.skill_registry.validate_output(skill_id, payload)
        schema_ok, schema_problems = schemas.validate(spec.output_schema, payload)
        for problem in (problems if not ok else []) + (schema_problems if not schema_ok else []):
            state.safety_issues.append(f"输出契约校验失败: {problem}")

    # -------------------------------------------------------------------- nodes
    def _bootstrap(self, state: ClinicalRunState) -> None:
        """Intake runs before planning: the plan depends on the risk mode."""
        self.agents["IntakeAgent"].run(state, self.tools, self._broker(state, "IntakeAgent"))
        self._validate_output(state, "IntakeAgent")
        self._checkpoint(state, "IntakeAgent")

    def _plan(self, state: ClinicalRunState) -> None:
        PlannerAgent(self.llm, self.skill_registry).run(state)
        self._validate_output(state, "PlannerAgent")
        self._checkpoint(state, "PlannerAgent")

    def _execute_tasks(self, state: ClinicalRunState, allow_prescription: bool) -> None:
        for task in state.tasks:
            if task.status not in ("pending", "repair_requested"):
                continue
            if task.agent == "CriticAgent":
                continue  # the critic is a terminal node, never a graph task
            if task.agent == "IntakeAgent" and state.loop_index == 0:
                task.status = "ok"
                continue
            if task.agent in PRESCRIPTIVE_AGENTS and not (state.role == "physician" and allow_prescription):
                task.status = "skipped_not_authorized"
                continue
            if not self._deps_ok(state, task):
                task.status = "skipped_dependency"
                continue
            agent = self.agents.get(task.agent)
            if agent is None or task.agent not in AGENT_CATALOG:
                task.status = "failed"
                state.fail_closed(f"未实现或未登记的任务Agent: {task.agent}")
                return

            agent.run(state, self.tools, self._broker(state, task.agent))
            self._validate_output(state, task.agent)
            task.status = "failed" if state.release_status == "failed_closed" else "ok"
            self._checkpoint(state, task.agent)
            if state.release_status == "failed_closed":
                return
            if state.risk_mode == "urgent" and task.agent == "UrgentCareAgent":
                return  # urgent plan is terminal; nothing downstream may run

    def _critic(self, state: ClinicalRunState) -> list[dict[str, str]]:
        self.agents["CriticAgent"].run(state, self.tools, self._broker(state, "CriticAgent"))
        self._validate_output(state, "CriticAgent")
        for task in state.tasks:
            if task.agent == "CriticAgent":
                task.status = "ok"
        self._checkpoint(state, "CriticAgent")
        return state.outputs.get("safety_audit", {}).get("repair_requests", [])

    def _reset_for_repair(self, state: ClinicalRunState, repairs: list[dict[str, str]]) -> bool:
        """Mark the tasks a critic finding asked to redo. Returns True if any.

        Everything downstream of the earliest repaired task is invalidated too,
        because it was derived from output that is now being recomputed.
        """
        reasons = {r.get("agent"): r.get("reason", "") for r in repairs if r.get("agent")}
        indexes = [i for i, t in enumerate(state.tasks) if t.agent in reasons]
        if not indexes:
            return False
        first = min(indexes)
        for index, task in enumerate(state.tasks):
            if task.agent == "CriticAgent" or index < first:
                continue
            if task.agent in reasons:
                task.status = "repair_requested"
                task.repair_reason = reasons[task.agent]
            elif task.status in ("ok", "skipped_dependency"):
                task.status = "pending"
        return True

    def _finalize(self, state: ClinicalRunState) -> None:
        # Repair loops re-run agents, so the same finding can be recorded twice.
        state.safety_issues = list(dict.fromkeys(state.safety_issues))
        self._backfill_questions(state)
        if state.release_status == "needs_more_information":
            has_soft = bool(state.outputs.get("intake", {}).get("screening", {}).get("soft_hits"))
            critical_gaps = [m for m in state.missing_information if m in {"神经症状", "大小便/会阴感觉", "发热外伤肿瘤史"}]
            if has_soft:
                state.release_status = "needs_examination"
            elif critical_gaps:
                state.release_status = "needs_more_information"
            elif state.outputs.get("biomedical"):
                state.release_status = "treatment_advice_only"
        state.outputs["run_meta"] = {
            "planner_mode": state.planner_mode,
            "llm": describe_client(self.llm),
            "loops_used": state.loop_index + 1,
            "budget": {
                "tool_calls": f"{state.budget.used_tool_calls}/{state.budget.max_tool_calls}",
                "llm_calls": f"{state.budget.used_llm_calls}/{state.budget.max_llm_calls}",
                "llm_tokens": f"{state.budget.used_llm_tokens}/{state.budget.max_llm_tokens}",
            },
            "tool_health": self.health.snapshot(),
            "knowledge": self._knowledge_meta(),
            "journal": self.journal.summary() if self.journal is not None else {"mode": "off"},
        }
        self._summarise(state)

    def _summarise(self, state: ClinicalRunState) -> None:
        """Write the clinical note, last of all.

        Deliberately *after* the release status is settled above rather than as a
        graph task: ``SummaryAgent`` skips a run that has not concluded, and a task
        scheduled mid-graph would always see ``needs_more_information`` and skip
        every time. Keeping it out of ``AGENT_CATALOG`` also means an LLM plan
        cannot schedule it somewhere it would be useless.
        """
        try:
            self.agents["SummaryAgent"].run(state, self.tools, self._broker(state, "SummaryAgent"))
        except Exception as exc:  # noqa: BLE001 - a note is an artefact, never a gate
            state.warn(f"病历摘要生成失败（不影响本次结论）: {type(exc).__name__}: {exc}")

    @staticmethod
    def _backfill_questions(state: ClinicalRunState) -> None:
        """Guarantee a run that still needs information actually asks for it.

        InterviewAgent normally owns questioning, but an LLM-proposed plan may
        legitimately omit it, and a run can fail closed before it executes. In
        either case a patient who is told "需要更多信息" without being asked
        anything has been given nothing to act on, so the axis probe bank fills in.
        """
        if state.open_questions or state.release_status == "urgent_action_plan":
            return
        # An interview that ran and was satisfied deliberately produced no
        # questions. Backfilling then would re-open an enquiry the judge just
        # closed, and the patient would be asked filler after answering everything.
        verdict = ((state.outputs.get("interview") or {}).get("verdict") or {}).get("verdict")
        if verdict in ("achieved", "stalled", "cap_reached"):
            return
        from .interview.axes import plan_next

        plan = plan_next(
            state.facts, state.complaint, role=state.role,
            risk_mode=state.risk_mode, limit=3,
        )
        questions = [
            plan.suggested_probes[axis_id][0]
            for axis_id in plan.axis_ids
            if plan.suggested_probes.get(axis_id)
        ]
        allowed = state.budget.reserve_questions(len(questions))
        state.open_questions = questions[: allowed or len(questions)]

    def _knowledge_meta(self) -> dict[str, Any]:
        """Which external sources backed this run, and under which licence policy."""
        store = getattr(self.tools, "knowledge", None)
        if store is None:
            return {
                "configured": False,
                "note": "未配置授权知识库；指南与药典证据为占位数据，仅内置规则包可用",
            }
        return {
            "configured": True,
            "enabled_sources": store.enabled_sources(),
            "policy": store.policy.to_dict(),
        }

    # ---------------------------------------------------------------------- run
    def _reopen_for_resume(self, state: ClinicalRunState) -> None:
        """Re-open unfinished work so a resumed run can use newly supplied facts.

        Resuming exists so the answers to a run's open questions can move it
        forward, which only works if the tasks those answers unblock are
        runnable again. Completed tasks stay completed, and a run that failed
        closed stays failed closed — resume is not an escape hatch from a
        safety decision.
        """
        if state.release_status == "failed_closed":
            return
        reopenable = {"skipped_dependency", "skipped_not_authorized", "repair_requested", "failed", "pending"}
        for task in state.tasks:
            if task.agent == "CriticAgent":
                continue
            if task.status in reopenable:
                task.status = "pending"
            elif task.agent in PRESCRIPTIVE_AGENTS and state.release_status in {
                "treatment_advice_only", "needs_more_information", "needs_examination",
                "insufficient_evidence", "blocked", "draft_for_physician",
            }:
                # New facts (special population, medication history, physician
                # signature) can legitimately change a prescriptive outcome.
                task.status = "pending"
        state.outputs.pop("safety_audit", None)

    def run(self, state: ClinicalRunState, allow_prescription: bool = False, resumed: bool = False) -> ClinicalRunState:
        # Record the permission on the state so agents can read it. The interview
        # needs it: 四诊 completeness is required before a dose-bearing draft, and
        # demanding it of every physician run — including ones that never asked
        # for a prescription — reports a blocking deficit that does not exist.
        state.allow_prescription = bool(allow_prescription)
        try:
            if not resumed:
                self._bootstrap(state)
            else:
                self._reopen_for_resume(state)
            if state.release_status != "failed_closed":
                if not state.tasks:
                    self._plan(state)
                self._run_loops(state, allow_prescription)
        except Exception as exc:  # noqa: BLE001 - an unexpected crash must still fail closed
            state.fail_closed(f"运行时异常，已故障关闭: {type(exc).__name__}: {exc}")
        finally:
            # The safety critic is unconditional: it must observe every run,
            # including ones that failed closed or skipped every clinical task.
            if "safety_audit" not in state.outputs:
                self._critic(state)
            self._check_replay_fidelity(state)
            self._finalize(state)
        return state

    def _check_replay_fidelity(self, state: ClinicalRunState) -> None:
        """A replay that did not reproduce must not look like one that did.

        Agents wrap model calls in broad ``except Exception`` so they can fall
        back to a deterministic path, which silently converts a replay divergence
        into an ordinary "fell back to rules" warning. The journal latches
        divergences precisely so this check can see them, and the run fails closed:
        an audit replay whose calls did not match the recording has answered a
        different question than the one asked of it.
        """
        journal = self.journal
        if journal is None or journal.mode != "replay":
            return
        if journal.diverged:
            first = journal.divergences[0]
            where = (
                f"同一调用 {first['issued']} 但参数不同"
                if first.get("differs_by") == "arguments"
                else f"记录的是 {first['recorded']}，本次发出的是 {first['issued']}"
            )
            state.fail_closed(
                f"重放偏离：日志第 {first['seq']} 条 —— {where}。"
                f"病例、代码或模型已变化，本次重放不能作为对原决策的复核。"
            )
        elif journal.live_after_exhaustion:
            state.warn(
                f"重放日志已耗尽，其后 {journal.live_after_exhaustion} 次调用走了实时链路；"
                f"本次结果不是纯离线重放"
            )

    def _run_loops(self, state: ClinicalRunState, allow_prescription: bool) -> None:
        while True:
            self._execute_tasks(state, allow_prescription)
            if state.release_status == "failed_closed":
                return
            repairs = self._critic(state)
            if not repairs or not state.budget.can_loop("repair"):
                if repairs:
                    state.warn(f"已达最大修复轮次({state.budget.max_loops})，遗留问题交由人工处理")
                return
            state.budget.count_loop("repair")
            state.loop_index += 1
            if not self._reset_for_repair(state, repairs):
                return
            # Clear the previous audit so the critic re-runs on the repaired state.
            state.outputs.pop("safety_audit", None)


def resume_run(
    checkpoint: str | Path,
    tools: ToolRegistry | None = None,
    allow_prescription: bool = False,
    **runner_kwargs: Any,
) -> ClinicalRunState:
    """Resume a checkpointed run from disk and drive it to completion."""
    state = YaobiGraphRunner.load_checkpoint(checkpoint)
    runner = YaobiGraphRunner(tools=tools, **runner_kwargs)
    return runner.run(state, allow_prescription=allow_prescription, resumed=True)
