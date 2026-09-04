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
    "InterviewAgent": AgentSpec(
        "InterviewAgent", "yaobi.interview", "自主追问：十问歌与骨科专科问诊，逐轮追问至病史充分",
        ("interview_axis_lookup", "red_flag_evidence_search"),
    ),
    "ConsultPanelAgent": AgentSpec(
        "ConsultPanelAgent", "yaobi.consult_panel", "多学科会诊子体（骨科/疼痛/康复/中医骨伤/药师），最保守优先合议",
        (
            "clinical_guideline_search", "tcm_pattern_knowledge_search", "similar_case_search",
            "counterexample_case_search", "expert_practice_profile", "patient_timeline_search",
            "drug_label_lookup", "drug_normalize", "drug_interaction_check", "interaction_check",
            "special_population_check", "red_flag_evidence_search", "interview_axis_lookup",
            "medical_image_read",
        ),
    ),
    "VisionAgent": AgentSpec(
        "VisionAgent", "yaobi.vision_read", "临床图片判读（影像翻拍/舌象/体态/患肢外观/报告转录）",
        ("medical_image_read", "red_flag_evidence_search", "interview_axis_lookup"),
    ),
    "OsteoporosisAgent": AgentSpec(
        "OsteoporosisAgent", "yaobi.osteoporosis_risk", "骨质疏松与脆性骨折风险评估、跌倒风险与用药前置条件",
        ("clinical_guideline_search", "drug_label_lookup", "drug_interaction_check",
         "special_population_check", "interview_axis_lookup", "medical_image_read"),
    ),
    "UrgentPlannerAgent": AgentSpec("UrgentPlannerAgent", "yaobi.urgent_triage", "急症假设与追问排序", (), risk_modes=("urgent",)),
    "UrgentCareAgent": AgentSpec(
        "UrgentCareAgent", "yaobi.urgent_triage", "急症鉴别、即时行动与转运建议",
        ("emergency_resource_lookup", "clinical_guideline_search"), risk_modes=("urgent",),
    ),
    "BiomedicalAgent": AgentSpec(
        "BiomedicalAgent", "yaobi.biomedical_differential", "西医鉴别诊断与查体/影像建议",
        ("clinical_guideline_search", "drug_label_lookup"),
    ),
    "TCMPatternAgent": AgentSpec("TCMPatternAgent", "yaobi.tcm_pattern", "中医辨证与反证需求", ("tcm_pattern_knowledge_search",)),
    "ExpertCaseAgent": AgentSpec(
        "ExpertCaseAgent", "yaobi.expert_case_reasoning", "专家经验画像、相似病例与反例的综合推理",
        ("similar_case_search", "counterexample_case_search", "expert_practice_profile", "patient_timeline_search"),
    ),
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
    if state.images:
        # Before the interview, not after it. The read depends only on intake,
        # and its ``suggest_ask`` exists to steer the questioning — scheduled
        # after ``InterviewAgent`` it arrived one full turn late, so the model
        # composed its questions blind to a film it already had.
        tasks.append(Task("T4", "VisionAgent", "判读随诊图片（非诊断）", ["medical_image_read"], ["T2"]))
    # The interview runs on every path, urgent included: an emergency still
    # needs its cauda-equina questions asked, just fewer of everything else.
    tasks.append(Task("T3", "InterviewAgent", "自主追问，评估病史充分性", ["interview_axis_lookup"], ["T2"]))
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
        # The panel is opt-in rather than default: five subagents multiply the
        # token cost of a run several-fold, and that is the operator's call to
        # make, not a default to discover on the bill. An LLM planner may still
        # schedule it on its own — it is in the catalogue — which is exactly the
        # autonomy this design is for.
        if state.enable_panel:
            tasks.insert(-3, Task("N8", "ConsultPanelAgent", "多学科会诊合议", ["clinical_guideline_search"], ["T3"]))
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
6. `attached_images` 非空时，患者已经上传了图片并在等你看。除非你有明确理由跳过，
   否则请安排 VisionAgent（工具 medical_image_read）——上传了却没人看，比没上传更糟。
   并且请把它排在 InterviewAgent **之前**：判读所见（可疑点、值得追问的问题）
   要用来驱动本轮追问，排在问诊之后就晚了一整轮。

只输出 JSON：{{"reasoning": "一句话说明取舍", "tasks": [
  {{"task_id": "P1", "agent": "AgentName", "objective": "本任务目标", "required_tools": [...], "depends_on": [...]}}
]}}"""


def build_planner_prompt(state: ClinicalRunState, skill_registry: Any | None = None) -> list[dict[str, str]]:
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
        # Attached images were missing from this context entirely, and the
        # consequence was not subtle: a model-authored plan never scheduled
        # ``VisionAgent``, so an uploaded X-ray was accepted, stored, and never
        # looked at — 「上传 X 片无法自动解析」, with nothing anywhere saying why.
        # The rule plan schedules the read whenever images exist; the model needs
        # the same fact to make the same call.
        "attached_images": [
            {"kind": str(i.get("kind") or "other"), "deidentified": bool(i.get("deidentified"))}
            for i in (state.images or [])
        ],
        "agent_catalog": catalog,
        # What each skill is for, so the planner reasons about capabilities
        # rather than guessing from agent names alone.
        "skill_catalog": skill_registry.catalog(state.role) if skill_registry is not None else [],
        "loop_index": state.loop_index,
    }
    import json

    return [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT.format(max_tasks=MAX_PLAN_TASKS)},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]


#: Keys a model may put the task list under. All mean the same thing; insisting
#: on ``tasks`` alone silently discarded most real plans.
_PLAN_LIST_KEYS = ("tasks", "plan", "task_plan", "steps", "graph", "task_graph", "nodes")

#: Per-task field aliases. Shape only — never a guess about meaning.
_AGENT_KEYS = ("agent", "agent_name", "name", "agent_id")
_TASK_ID_KEYS = ("task_id", "id", "step_id", "node_id")
_TOOL_KEYS = ("required_tools", "tools", "tool_names")
_DEPENDS_KEYS = ("depends_on", "dependencies", "depends", "after", "requires")


def _first(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = item.get(key)
        if value:
            return value
    return None


def _plan_entries(payload: Any) -> list[Any]:
    """Find the task list in whatever container the model chose.

    Accepts a bare list, a list under any recognised key, or one level of nesting
    (``{"plan": {"tasks": [...]}}``). Shape tolerance only: every entry still has
    to name an agent that exists, and :func:`validate_plan` still checks roles,
    risk modes, tool subsets and cycles afterwards.
    """
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in _PLAN_LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for inner in _PLAN_LIST_KEYS:
                nested = value.get(inner)
                if isinstance(nested, list):
                    return nested
    return []


def parse_plan(payload: Any, diagnostics: list[str] | None = None) -> list[Task]:
    """Turn a model JSON payload into Tasks.

    Deliberately liberal about *shape* and unchanged about *content*: an entry is
    kept only when it names an agent, and nothing here decides whether that agent
    is allowed to run. Being strict about shape bought no safety at all — it just
    meant a competent plan under a ``"plan"`` key became a silent fallback to the
    rule-based graph, with the run reporting ``planner_mode: rule``.

    ``diagnostics`` collects why entries were dropped, so the planner can say what
    happened instead of leaving the operator to guess.
    """
    notes = diagnostics if diagnostics is not None else []
    raw_tasks = _plan_entries(payload)
    if not raw_tasks:
        notes.append(
            "找不到任务列表；已接受的键: " + "/".join(_PLAN_LIST_KEYS)
            + f"，实际收到: {sorted(payload)[:6] if isinstance(payload, dict) else type(payload).__name__}"
        )
        return []

    tasks: list[Task] = []
    for index, item in enumerate(raw_tasks):
        # A bare string is accepted only as an *exact* catalogue name. Fuzzy
        # matching would be the parser guessing at intent, which is the one thing
        # it must not do.
        if isinstance(item, str):
            if item in AGENT_CATALOG:
                tasks.append(Task(f"L{index + 1}", item, AGENT_CATALOG[item].description, [], [], origin="llm"))
            else:
                notes.append(f"条目 #{index + 1} 是字符串但不是已登记的 Agent 名: {item[:40]!r}")
            continue
        if not isinstance(item, dict):
            notes.append(f"条目 #{index + 1} 既不是对象也不是字符串: {type(item).__name__}")
            continue
        agent = _first(item, _AGENT_KEYS)
        if not agent:
            notes.append(f"条目 #{index + 1} 未指明 Agent（可用键: {'/'.join(_AGENT_KEYS)}）")
            continue
        agent = str(agent)
        tasks.append(
            Task(
                task_id=str(_first(item, _TASK_ID_KEYS) or f"L{index + 1}"),
                agent=agent,
                objective=str(item.get("objective") or item.get("goal")
                              or AGENT_CATALOG.get(agent, AgentSpec("", "", "")).description),
                required_tools=[str(t) for t in (_first(item, _TOOL_KEYS) or []) if isinstance(t, str)],
                depends_on=[str(d) for d in (_first(item, _DEPENDS_KEYS) or []) if isinstance(d, str)],
                origin="llm",
            )
        )
    return tasks


class PlannerAgent:
    """Produces the task graph, preferring a validated LLM plan when available."""

    name = "PlannerAgent"
    skill_id = "yaobi.planning"

    def __init__(self, llm: Any | None = None, skill_registry: Any | None = None) -> None:
        self.llm = llm
        self.skill_registry = skill_registry

    def run(self, state: ClinicalRunState) -> ClinicalRunState:
        fallback = rule_plan(state)
        tasks, mode, note = fallback, "rule", "llm_not_configured"

        if self.llm is not None and getattr(self.llm, "available", False):
            proposal, note, diagnostics = self._propose(state)
            if proposal:
                ok, problems = validate_plan(proposal, state)
                if ok:
                    tasks, mode = proposal, "llm"
                    note = "llm_plan_accepted"
                else:
                    note = "llm_plan_rejected:" + "; ".join(problems[:3])
                    state.warn(f"LLM 规划被规则层驳回，已回退默认计划: {problems[:3]}")
            elif note == "llm_responded":
                # The model answered and nothing usable came out. This used to
                # leave the note at "llm_responded" with no warning at all, so the
                # run reported `planner_mode: rule` next to a note saying the model
                # replied, and the operator had no way to tell which it was.
                note = "llm_plan_unparseable:" + ("; ".join(diagnostics[:3]) or "empty task list")
                state.warn(f"LLM 规划无法解析为任务，已回退默认计划: {diagnostics[:3] or ['空任务列表']}")

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

    def _propose(self, state: ClinicalRunState) -> tuple[list[Task], str, list[str]]:
        from ..llm.base import LLMError

        if not state.budget.reserve_llm():
            return [], "llm_budget_exhausted", []
        try:
            response = self.llm.chat(build_planner_prompt(state, self.skill_registry),
                                     temperature=0.0, max_tokens=1200, response_format_json=True)
        except (LLMError, Exception) as exc:  # noqa: BLE001 - a failed planner must never fail the run
            state.warn(f"LLM 规划调用失败，已回退默认计划: {exc!r}")
            return [], f"llm_error:{type(exc).__name__}", []
        state.budget.charge_llm_tokens(response.total_tokens)
        diagnostics: list[str] = []
        from ..llm.base import extract_json_with_repairs

        payload, repairs = extract_json_with_repairs(response.text, {})
        if repairs:
            # Noted on the state, not just in the diagnostics, because a *successful*
            # plan that needed repairing is the interesting case: "unclosed" almost
            # always means the plan hit max_tokens, which is a configuration fix
            # rather than a model failure, and the operator cannot infer that from a
            # task list that came out looking fine.
            message = "规划输出经 JSON 修复后才可解析: " + "、".join(repairs)
            diagnostics.append(message)
            state.note(message)
        tasks = parse_plan(payload, diagnostics)
        if not tasks and not diagnostics:
            diagnostics.append(f"响应不含可解析内容（前 120 字）: {(response.text or '')[:120]!r}")
        return tasks, "llm_responded", diagnostics

    @staticmethod
    def _append_critic(tasks: list[Task]) -> list[Task]:
        """The safety critic is appended by the system, never by the planner."""
        tasks = [t for t in tasks if t.agent != "CriticAgent"]
        return tasks + [Task("SAFETY", "CriticAgent", "安全审查、引用校验与修复请求", [], [])]
