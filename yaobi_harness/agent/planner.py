"""Planning: a validated task graph, produced by rules or by an LLM.

The planner is the one place where "autonomy" lives, so it is also the place
that needs the tightest containment. An LLM-proposed plan is treated as an
*untrusted proposal*: it is checked against the agent catalogue, the skill
allowlists and the risk-mode rules, and any violation discards the proposal and
falls back to the deterministic plan. The model can therefore reorder, prune or
extend the investigation, but it cannot invent an agent, reach a tool its skill
forbids, or route around urgent-mode prescription bans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..state import ClinicalRunState, Task
from ..tools import TOOL_NAMES


@dataclass(frozen=True)
class AgentSpec:
    """What an agent is allowed to be asked to do."""

    name: str
    skill_id: str
    description: str
    tools: tuple[str, ...] = ()
    roles: tuple[str, ...] = ("patient", "physician", "researcher")
    risk_modes: tuple[str, ...] = ("routine", "urgent")
    prescriptive: bool = False


AGENT_CATALOG: dict[str, AgentSpec] = {
    "TimelineAgent": AgentSpec("TimelineAgent", "yaobi.timeline", "整理病史时间线与既往就诊轨迹", ("patient_timeline_search",)),
    "IntakeAgent": AgentSpec("IntakeAgent", "yaobi.intake", "红旗筛查与信息缺口识别", ("red_flag_evidence_search",)),
    "UrgentPlannerAgent": AgentSpec("UrgentPlannerAgent", "yaobi.urgent_triage", "急症假设与追问排序", (), risk_modes=("urgent",)),
    "UrgentCareAgent": AgentSpec(
        "UrgentCareAgent", "yaobi.urgent_triage", "急症鉴别、即时行动与转运建议",
        ("emergency_resource_lookup", "clinical_guideline_search"), risk_modes=("urgent",),
    ),
    "BiomedicalAgent": AgentSpec("BiomedicalAgent", "yaobi.biomedical_differential", "西医鉴别诊断与查体/影像建议", ("clinical_guideline_search",)),
    "TCMPatternAgent": AgentSpec("TCMPatternAgent", "yaobi.tcm_pattern", "中医辨证与反证需求", ("tcm_pattern_knowledge_search",)),
    "ExpertCaseAgent": AgentSpec("ExpertCaseAgent", "yaobi.expert_case_reasoning", "相似病例与反例检索", ("similar_case_search", "counterexample_case_search")),
    "MedicationSafetyAgent": AgentSpec(
        "MedicationSafetyAgent", "yaobi.medication_safety", "现有西药相互作用、禁忌与围术期风险筛查",
        ("drug_interaction_check", "drug_label_lookup", "drug_normalize"),
    ),
    "FormulaAgent": AgentSpec(
        "FormulaAgent", "yaobi.formula_design", "候选治法与方剂组成", ("formula_composition_search",),
        roles=("physician",), risk_modes=("routine",), prescriptive=True,
    ),
    "DoseAgent": AgentSpec(
        "DoseAgent", "yaobi.dose_generation", "逐味剂量生成与放行前安全校验",
        ("herb_dose_distribution", "pharmacopeia_check", "interaction_check", "special_population_check"),
        roles=("physician",), risk_modes=("routine",), prescriptive=True,
    ),
    "PhysicianReviewAgent": AgentSpec(
        "PhysicianReviewAgent", "yaobi.physician_review", "提交医师逐味审核并记录签名放行",
        ("physician_review_submit",), roles=("physician",), risk_modes=("routine",), prescriptive=True,
    ),
    "CriticAgent": AgentSpec("CriticAgent", "yaobi.safety_critic", "安全审查、引用校验与修复请求", ()),
}

#: Agents that never call tools; they still need a skill for auditability.
NO_TOOL_AGENTS = {"UrgentPlannerAgent", "CriticAgent"}

MAX_PLAN_TASKS = 14


def rule_plan(state: ClinicalRunState) -> list[Task]:
    """The deterministic fallback plan. Always valid by construction."""
    tasks = [
        Task("T1", "TimelineAgent", "标准化病历与时间线", ["patient_timeline_search"]),
        Task("T2", "IntakeAgent", "识别信息缺口和红旗", ["red_flag_evidence_search"]),
    ]
    if state.risk_mode == "urgent":
        tasks += [
            Task("U1", "UrgentPlannerAgent", "急症假设、追问与资源预算"),
            Task("U2", "UrgentCareAgent", "急症鉴别、即时行动与转运", ["emergency_resource_lookup", "clinical_guideline_search"], ["U1"]),
        ]
    else:
        tasks += [
            Task("N1", "BiomedicalAgent", "西医鉴别", ["clinical_guideline_search"]),
            Task("N2", "TCMPatternAgent", "辨证论治", ["tcm_pattern_knowledge_search"]),
            Task("N3", "ExpertCaseAgent", "相似与反例病例", ["similar_case_search", "counterexample_case_search"]),
            Task("N7", "MedicationSafetyAgent", "现有用药相互作用与禁忌筛查", ["drug_interaction_check"]),
            Task("N4", "FormulaAgent", "候选治法方剂", ["formula_composition_search"], ["N1", "N2", "N3"]),
            Task(
                "N5", "DoseAgent", "逐味剂量与安全校验",
                ["herb_dose_distribution", "pharmacopeia_check", "interaction_check", "special_population_check"],
                ["N4"],
            ),
            Task("N6", "PhysicianReviewAgent", "医师逐味审核", ["physician_review_submit"], ["N5"]),
        ]
    return tasks


def validate_plan(tasks: list[Task], state: ClinicalRunState) -> tuple[bool, list[str]]:
    """Check a proposed plan against the catalogue and the run's risk posture."""
    problems: list[str] = []
    if not tasks:
        return False, ["plan is empty"]
    if len(tasks) > MAX_PLAN_TASKS:
        problems.append(f"plan exceeds {MAX_PLAN_TASKS} tasks")

    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        problems.append("duplicate task ids")

    for task in tasks:
        spec = AGENT_CATALOG.get(task.agent)
        if spec is None:
            problems.append(f"unknown agent {task.agent!r}")
            continue
        if state.risk_mode not in spec.risk_modes:
            problems.append(f"{task.agent} is not permitted in risk_mode={state.risk_mode}")
        if state.role not in spec.roles:
            problems.append(f"{task.agent} is not permitted for role={state.role}")
        unknown_tools = [t for t in task.required_tools if t not in TOOL_NAMES]
        if unknown_tools:
            problems.append(f"{task.agent} requests unknown tools {unknown_tools}")
        outside = [t for t in task.required_tools if t not in spec.tools]
        if outside:
            problems.append(f"{task.agent} requests tools outside its skill: {outside}")
        missing_deps = [d for d in task.depends_on if d not in ids]
        if missing_deps:
            problems.append(f"{task.task_id} depends on unknown tasks {missing_deps}")

    if _has_cycle(tasks):
        problems.append("plan dependency graph contains a cycle")
    if state.risk_mode == "urgent" and any(AGENT_CATALOG[t.agent].prescriptive for t in tasks if t.agent in AGENT_CATALOG):
        problems.append("urgent plan must not contain prescriptive agents")
    return not problems, problems


def _has_cycle(tasks: list[Task]) -> bool:
    graph = {t.task_id: [d for d in t.depends_on] for t in tasks}
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        if state.get(node) == 1:
            return True
        if state.get(node) == 2:
            return False
        state[node] = 1
        for dep in graph.get(node, []):
            if dep in graph and visit(dep):
                return True
        state[node] = 2
        return False

    return any(visit(node) for node in graph)


PLANNER_SYSTEM_PROMPT = """你是骨科临床决策系统的规划器。你只能从给定的 Agent 目录中选择任务，
不能发明新的 Agent 或工具。你的输出会被规则层逐条校验，任何越权提案都会被整体丢弃并回退到默认计划。

硬约束：
1. 急症模式(urgent)下禁止任何处方/剂量类 Agent。
2. 每个任务的 required_tools 必须是该 Agent 目录里声明的工具子集。
3. depends_on 只能引用本次计划中已存在的 task_id，且不得构成环。
4. 任务总数不超过 {max_tasks}。
5. 安全审查节点由系统强制追加，你不需要也不应该省略其它必要的证据收集步骤。

只输出 JSON：{{"reasoning": "一句话说明取舍", "tasks": [
  {{"task_id": "P1", "agent": "AgentName", "objective": "本任务目标", "required_tools": [...], "depends_on": [...]}}
]}}"""


def build_planner_prompt(state: ClinicalRunState) -> list[dict[str, str]]:
    catalog = [
        {
            "agent": spec.name,
            "does": spec.description,
            "tools": list(spec.tools),
            "roles": list(spec.roles),
            "risk_modes": list(spec.risk_modes),
        }
        for spec in AGENT_CATALOG.values()
        if state.role in spec.roles and state.risk_mode in spec.risk_modes
    ]
    screening = state.outputs.get("intake", {}).get("screening", {})
    context = {
        "chief_complaint": state.complaint,
        "role": state.role,
        "risk_mode": state.risk_mode,
        "red_flag_hits": screening.get("hits", []),
        "soft_signals": screening.get("soft_hits", []),
        "missing_information": state.missing_information,
        "known_facts": {k: v for k, v in state.facts.items() if k != "raw"},
        "agent_catalog": catalog,
        "loop_index": state.loop_index,
    }
    import json

    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT.format(max_tasks=MAX_PLAN_TASKS)},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]


def parse_plan(payload: Any) -> list[Task]:
    """Turn a model JSON payload into Tasks; malformed entries are dropped."""
    if not isinstance(payload, dict):
        return []
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        return []
    tasks: list[Task] = []
    for index, item in enumerate(raw_tasks):
        if not isinstance(item, dict) or not item.get("agent"):
            continue
        tasks.append(
            Task(
                task_id=str(item.get("task_id") or f"L{index + 1}"),
                agent=str(item["agent"]),
                objective=str(item.get("objective") or AGENT_CATALOG.get(str(item["agent"]), AgentSpec("", "", "")).description),
                required_tools=[str(t) for t in (item.get("required_tools") or []) if isinstance(t, str)],
                depends_on=[str(d) for d in (item.get("depends_on") or []) if isinstance(d, str)],
                origin="llm",
            )
        )
    return tasks


class PlannerAgent:
    """Produces the task graph, preferring a validated LLM plan when available."""

    name = "PlannerAgent"
    skill_id = "yaobi.planning"

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def run(self, state: ClinicalRunState) -> ClinicalRunState:
        fallback = rule_plan(state)
        tasks, mode, note = fallback, "rule", "llm_not_configured"

        if self.llm is not None and getattr(self.llm, "available", False):
            proposal, note = self._propose(state)
            if proposal:
                ok, problems = validate_plan(proposal, state)
                if ok:
                    tasks, mode = proposal, "llm"
                    note = "llm_plan_accepted"
                else:
                    note = "llm_plan_rejected:" + "; ".join(problems[:3])
                    state.warn(f"LLM 规划被规则层驳回，已回退默认计划: {problems[:3]}")

        tasks = self._append_critic(tasks)
        state.tasks = tasks
        state.planner_mode = mode
        state.outputs["plan"] = {
            "planner_mode": mode,
            "note": note,
            "tasks": [
                {"task_id": t.task_id, "agent": t.agent, "objective": t.objective,
                 "required_tools": t.required_tools, "depends_on": t.depends_on, "origin": t.origin}
                for t in tasks
            ],
        }
        state.trace("PlannerAgent", "plan", state.complaint, f"{mode}: 生成{len(tasks)}个任务 ({note})")
        return state

    def _propose(self, state: ClinicalRunState) -> tuple[list[Task], str]:
        from ..llm.base import LLMError

        if not state.budget.reserve_llm():
            return [], "llm_budget_exhausted"
        try:
            response = self.llm.chat(build_planner_prompt(state), temperature=0.0, max_tokens=1200, response_format_json=True)
        except (LLMError, Exception) as exc:  # noqa: BLE001 - a failed planner must never fail the run
            state.warn(f"LLM 规划调用失败，已回退默认计划: {exc!r}")
            return [], f"llm_error:{type(exc).__name__}"
        state.budget.charge_llm_tokens(response.total_tokens)
        return parse_plan(response.json({})), "llm_responded"

    @staticmethod
    def _append_critic(tasks: list[Task]) -> list[Task]:
        """The safety critic is appended by the system, never by the planner."""
        tasks = [t for t in tasks if t.agent != "CriticAgent"]
        return tasks + [Task("SAFETY", "CriticAgent", "安全审查、引用校验与修复请求", [], [])]
