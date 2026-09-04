"""会诊子体：consult subagents with a conservative-first synthesis.

Grok Build's subagents give a child its own context window, a coarse capability
mode, and a persona overlay that shapes behaviour without changing what the child
is allowed to touch. That decomposition is exactly right for a clinical panel:
several specialists look at the same case from genuinely different angles, and the
value comes from their *disagreement*, which a single context tends to average
away into one confident-sounding answer.

Four adaptations make it safe here.

**No subagent is ever prescriptive.** ``consult_mode`` can only narrow a skill's
tool grant, and ``advisory`` — the widest mode — still excludes
``formula_composition_search``, ``herb_dose_distribution`` and
``physician_review_submit``. Panel members advise; the deterministic pipeline
prescribes. This is enforced in :meth:`SkillSpec.effective_tools`, so a persona
file cannot grant its way past it.

**Depth is one.** A consult cannot convene its own consult. Same reason Grok caps
nesting: it keeps the tree auditable and stops a runaway fan-out, and in a
clinical setting an unbounded referral chain is also a liability nobody can review.

**Budget is carved, not shared.** Each member gets a slice of the parent budget.
A member that burns its slice degrades to nothing; it cannot starve the dose
safety checks that run after the panel.

**Synthesis is not a vote.** This is the deliberate inversion of the pattern.
Majority voting is right for "is this bug real" and wrong for "is this cauda
equina" — one member seeing an emergency outweighs four who do not. So the panel
takes the *maximum* urgency any member raised and the *union* of their safety
concerns, and reports agreement separately as information for the operator rather
than as a filter on the output.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMError
from ..state import ClinicalRunState
from ..tools import CapabilityBroker, ToolRegistry

#: Depth cap. A consult may not convene a consult.
MAX_DEPTH = 1

#: Default thread count for a panel. Members are I/O bound, so this is about
#: overlapping network waits, not CPU. Capped low because each thread holds an
#: LLM connection and a hospital deployment may be behind a modest proxy.
DEFAULT_CONCURRENCY = 4

#: Urgency ladder, least to most severe. The panel result takes the maximum.
URGENCY_ORDER = ("routine", "expedited", "urgent", "emergency")

PERSONA_SYSTEM_PROMPT = """你是骨科多学科会诊中的**{persona_label}**。

{persona_instructions}

## 本次会诊的边界

1. 你只出**会诊意见**，不出处方。方剂与剂量由独立链路处理，你**不得输出任何药名剂量**。
2. 你只能调用系统列出的工具；越权调用会被拒绝并记入安全审查。
3. 你的意见必须落到你实际看到的材料或调用过的工具结果上；没有依据就写"证据不足"。
4. 如果你认为存在急症风险，**必须**在 `urgency` 里如实上调，不要因为别人可能不同意而保留。
   合议采取"最保守优先"，你上调不会被投票否决。
5. 用中文，控制在你的专业视角内——不要复述别人的专科内容。

## 输出格式

只输出 JSON：
{{
  "urgency": "routine|expedited|urgent|emergency",
  "key_findings": ["从你的专科视角看到的关键点"],
  "concerns": ["你认为必须处理或排除的风险"],
  "recommend_next": ["下一步该做什么（检查、查体、转诊、康复动作）"],
  "questions_for_patient": ["你认为还必须问清的问题"],
  "dissent": "如果你预期与其它专科看法不同，说明分歧点；否则留空",
  "evidence_note": "你的意见依赖什么证据，强度如何"
}}"""

#: Bundled personas. Each is a *behavioural overlay*: it changes what the member
#: attends to, never what it may call. A site can add its own via a SKILL.md
#: with ``persona:`` set.
PERSONAS: dict[str, dict[str, Any]] = {
    "ortho_attending": {
        "label": "骨科主任医师",
        "consult_mode": "screening",
        "instructions": (
            "你的首要职责是**不漏诊急症**。按「先排除危险、再考虑常见」的顺序思考：\n"
            "马尾神经综合征、脊柱感染、转移瘤、病理性骨折、进行性神经缺损、血管急症。\n"
            "对每一条，明确写出「已排除/未排除/无法判断」，而不是笼统说「考虑腰肌劳损」。\n"
            "同时判断是否需要影像，以及需要哪一种（依据 Ottawa/NEXUS 一类的成熟规则，"
            "而不是「拍个片看看」）。"
        ),
        "inputs": ("complaint", "facts", "screening"),
        "outputs": ("urgency", "concerns", "recommend_next"),
    },
    "pain_specialist": {
        "label": "疼痛科医师",
        "consult_mode": "screening",
        "instructions": (
            "你从**疼痛机制**入手：这是外周痛感受性、神经病理性、还是中枢敏化为主？\n"
            "依据是疼痛性质、分布是否符合解剖、有无过敏性疼痛与异常疼痛、"
            "夜间与静息表现、以及既往治疗的反应模式。\n"
            "机制不同则处理完全不同——请明确写出你判断的机制及其把握程度。"
        ),
        "inputs": ("complaint", "facts"),
        "outputs": ("key_findings", "recommend_next"),
    },
    "rehab_specialist": {
        "label": "康复科医师",
        "consult_mode": "evidence_only",
        "instructions": (
            "你关注**功能与暴露**：现在能做什么、不能做什么，以及是什么日常暴露在维持这个问题。\n"
            "给出可执行到「每天做几次、做多久、什么感觉时停」的具体动作与工位/姿势调整，"
            "而不是「注意休息」「加强锻炼」这类无法执行的话。\n"
            "同时识别恐动与灾难化倾向（黄旗），它们比影像更能预测一年后的功能。"
        ),
        "inputs": ("complaint", "facts", "function_scores"),
        "outputs": ("recommend_next",),
    },
    "tcm_orthopedist": {
        "label": "中医骨伤科医师",
        "consult_mode": "evidence_only",
        "instructions": (
            "你按**辨证**思路工作：从四诊材料判断证型（寒湿、湿热、气滞血瘀、肝肾亏虚、气血两虚等），"
            "并明确指出**还缺哪些四诊信息**才能把握住。\n"
            "缺舌脉时不要硬凑主证型，写「待辨证」并说明需要补什么。\n"
            "只给治法方向（如补益肝肾、活血通络），**不得给出方剂组成与克数**。"
        ),
        "inputs": ("complaint", "facts", "four_diagnoses"),
        "outputs": ("key_findings", "questions_for_patient"),
    },
    "clinical_pharmacist": {
        "label": "临床药师",
        "consult_mode": "screening",
        "instructions": (
            "你只看**用药安全**：现有用药之间、以及与拟用中药之间的相互作用、禁忌、"
            "特殊人群调整、围术期与椎管内麻醉的抗凝管理。\n"
            "骨科高频风险请逐条核对：NSAID+抗凝、三重打击（ACEI/ARB+利尿剂+NSAID）、"
            "NSAID+激素/SSRI、阿片+苯二氮、曲马多+SSRI、双膦酸盐与阳离子/肾功能/食管、"
            "地舒单抗与低钙、甲氨蝶呤+NSAID/TMP-SMX、秋水仙碱+CYP3A4/P-gp 抑制剂。\n"
            "发现风险时写清**机制**与**具体处理**，不要只说「注意监测」。"
        ),
        "inputs": ("facts", "medications"),
        "outputs": ("concerns", "recommend_next"),
    },
}

#: Which personas to convene by default, keyed by what the case looks like.
DEFAULT_PANEL = ("ortho_attending", "clinical_pharmacist")


@dataclass
class ConsultOpinion:
    persona: str
    label: str
    urgency: str = "routine"
    key_findings: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    recommend_next: list[str] = field(default_factory=list)
    questions_for_patient: list[str] = field(default_factory=list)
    dissent: str = ""
    evidence_note: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    tool_calls: int = 0
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona": self.persona, "label": self.label, "urgency": self.urgency,
            "key_findings": self.key_findings, "concerns": self.concerns,
            "recommend_next": self.recommend_next,
            "questions_for_patient": self.questions_for_patient,
            "dissent": self.dissent, "evidence_note": self.evidence_note,
            "evidence_ids": self.evidence_ids, "tool_calls": self.tool_calls,
            "ok": self.ok, "error": self.error,
        }


@dataclass
class PanelResult:
    """The convened panel's combined view, with agreement reported separately."""

    opinions: list[ConsultOpinion] = field(default_factory=list)
    urgency: str = "routine"
    concerns: list[str] = field(default_factory=list)
    recommend_next: list[str] = field(default_factory=list)
    questions_for_patient: list[str] = field(default_factory=list)
    dissents: list[str] = field(default_factory=list)
    #: How many members agreed on the final urgency. Information, not a filter.
    agreement: str = ""
    convened: list[str] = field(default_factory=list)
    mode: str = "not_run"
    #: Threads actually used. 1 means the sequential path ran.
    concurrency: int = 1
    #: Per-member merge record: how many evidence rows, warnings and traces each
    #: member contributed, and how its scope-local ids were remapped.
    merge: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "opinions": [o.to_dict() for o in self.opinions],
            "urgency": self.urgency, "concerns": self.concerns,
            "recommend_next": self.recommend_next,
            "questions_for_patient": self.questions_for_patient,
            "dissents": self.dissents, "agreement": self.agreement,
            "convened": self.convened, "mode": self.mode,
            "concurrency": self.concurrency, "merge": self.merge,
        }


def choose_panel(state: ClinicalRunState) -> list[str]:
    """Pick the personas this case warrants, deterministically.

    Rule-chosen rather than model-chosen on purpose: the pharmacist must be
    convened whenever there is a medication list, and that must not depend on a
    model noticing. The model's autonomy lives *inside* each member.
    """
    chosen = list(DEFAULT_PANEL)
    facts = state.facts or {}
    text = " ".join(str(v) for v in (state.complaint, facts.get("pain_location"), facts.get("neuro_symptoms")) if v)

    if facts.get("four_diagnoses") or "舌" in text or state.role == "physician":
        chosen.append("tcm_orthopedist")
    if any(t in text for t in ("麻", "无力", "放射", "窜", "灼", "电")):
        chosen.append("pain_specialist")
    if any(t in text for t in ("月", "年", "反复", "一直", "慢性")) or facts.get("odi") is not None:
        chosen.append("rehab_specialist")
    return list(dict.fromkeys(chosen))


class ConsultSubagent:
    """One panel member: its own context, its own narrowed tool set."""

    def __init__(
        self,
        persona: str,
        llm: Any,
        *,
        skill_id: str = "yaobi.consult_panel",
        depth: int = 1,
        max_steps: int = 3,
    ) -> None:
        if persona not in PERSONAS:
            raise ValueError(f"unknown consult persona {persona!r}")
        if depth > MAX_DEPTH:
            raise ValueError(f"consult depth {depth} exceeds MAX_DEPTH={MAX_DEPTH}")
        self.persona = persona
        self.profile = PERSONAS[persona]
        self.llm = llm
        self.skill_id = skill_id
        self.depth = depth
        self.max_steps = max_steps

    @property
    def label(self) -> str:
        return str(self.profile.get("label") or self.persona)

    def run(
        self,
        state: ClinicalRunState,
        tools: ToolRegistry,
        registry: Any,
        *,
        health: Any = None,
        sub_budget: Any = None,
        scope: Any = None,
    ) -> ConsultOpinion:
        """Run this member as a bounded tool loop and return its opinion.

        ``scope`` is a :class:`~yaobi_harness.agent.scope.MemberScope` when the
        panel runs concurrently: the loop then writes evidence into the scope
        rather than into the shared state, and the caller merges it afterwards.
        Passing ``None`` runs directly against ``state``, which is what the
        sequential path does.
        """
        from .toolloop import ToolLoop

        opinion = ConsultOpinion(self.persona, self.label)
        spec = registry.specs.get(self.skill_id) if registry else None
        if spec is None:
            opinion.ok, opinion.error = False, f"技能 {self.skill_id} 未登记"
            return opinion
        if self.llm is None or not getattr(self.llm, "available", False):
            opinion.ok, opinion.error = False, "未配置模型，会诊子体不可用"
            return opinion

        mode = str(self.profile.get("consult_mode") or spec.consult_mode or "screening")
        granted = spec.effective_tools(mode)
        target = scope if scope is not None else state
        budget = sub_budget if sub_budget is not None else getattr(target, "budget", None)
        broker = _member_broker(state, registry, self.skill_id, granted, health=health, budget=budget)

        loop = ToolLoop(
            self.llm, tools, broker, target,
            agent_name=self.label, skill_id=self.skill_id, skill_spec=spec,
            max_steps=self.max_steps,
        )
        # The member sees only its own narrowed set, not the skill's full grant.
        loop.allowed_tool_specs = lambda: _specs_for(granted)  # type: ignore[method-assign]
        # The persona is a *system-prompt* overlay, which is the whole mechanism.
        # Previously it only reached the model as a JSON field in the user message
        # while the system prompt stayed generic — so every member read the same
        # instructions and the "independent perspectives" the panel exists to
        # produce were much weaker than the design claimed.
        loop.persona_prompt = self.system_prompt(schema_name="ConsultOpinion")  # type: ignore[attr-defined]

        # A scope already owns its budget, so nothing needs swapping. The
        # sequential path still needs the swap, and it is safe there because only
        # one member runs at a time.
        swap = scope is None and sub_budget is not None
        original_budget = state.budget if swap else None
        if swap:
            state.budget = sub_budget
        try:
            result = loop.run(
                objective=f"以{self.label}的专科视角给出会诊意见",
                context=self._context(target),
                schema_name="ConsultOpinion",
            )
        except (LLMError, Exception) as exc:  # noqa: BLE001 - one member must not kill the panel
            opinion.ok, opinion.error = False, f"{type(exc).__name__}: {exc}"[:200]
            return opinion
        finally:
            if swap:
                state.budget = original_budget

        if not result.ok or not isinstance(result.output, dict):
            opinion.ok, opinion.error = False, result.error or result.mode
            return opinion

        payload = result.output
        opinion.urgency = _clamp_urgency(payload.get("urgency"))
        opinion.key_findings = _strings(payload.get("key_findings"))
        opinion.concerns = _strings(payload.get("concerns"))
        opinion.recommend_next = _strings(payload.get("recommend_next"))
        opinion.questions_for_patient = _strings(payload.get("questions_for_patient"))
        opinion.dissent = str(payload.get("dissent") or "")[:400]
        opinion.evidence_note = str(payload.get("evidence_note") or "")[:400]
        opinion.evidence_ids = list(result.evidence_ids)
        opinion.tool_calls = sum(1 for s in result.steps if s.kind == "tool_call")
        return opinion

    def system_prompt(self, schema_name: str = "ConsultOpinion") -> str:
        """This member's system prompt: persona overlay plus the output contract.

        The overlay changes what the member attends to, never what it may call —
        the tool set is narrowed separately, by ``consult_mode``.
        """
        from .toolloop import schema_hint

        return PERSONA_SYSTEM_PROMPT.format(
            persona_label=self.label,
            persona_instructions=str(self.profile.get("instructions") or ""),
        ) + f"\n\n严格按下面的字段输出：\n{schema_hint(schema_name)}"

    def _context(self, state: Any) -> dict[str, Any]:
        """What this member is shown. Doses and signatures are never included."""
        outputs = state.outputs or {}
        return {
            "persona_brief": str(self.profile.get("instructions") or ""),
            "complaint": (state.complaint or "")[:4000],
            "role": state.role,
            "risk_mode": state.risk_mode,
            "facts": {k: v for k, v in (state.facts or {}).items() if k != "physician_review"},
            "screening": (outputs.get("intake") or {}).get("screening", {}),
            "interview": (outputs.get("interview") or {}).get("coverage", {}),
            "expected_outputs": list(self.profile.get("outputs") or ()),
        }


class ConsultPanel:
    """Convenes members, runs them concurrently, and synthesises conservatively.

    Members are I/O bound — each spends its time waiting on an LLM endpoint — so
    a thread pool turns five sequential round trips into roughly one. What makes
    that safe is :mod:`yaobi_harness.agent.scope`: each member writes into its own
    scope and the results are merged in *convened order*, so the evidence ledger
    is identical whether the panel ran concurrently or one member at a time.
    """

    def __init__(
        self,
        llm: Any,
        *,
        skill_id: str = "yaobi.consult_panel",
        max_members: int = 5,
        concurrency: int | None = None,
    ) -> None:
        self.llm = llm
        self.skill_id = skill_id
        self.max_members = max_members
        #: Threads to use. 1 forces the sequential path, which is what a
        #: debugging session or a deterministic-stub test wants.
        self.concurrency = concurrency if concurrency is not None else _default_concurrency()

    def run(
        self,
        state: ClinicalRunState,
        tools: ToolRegistry,
        registry: Any,
        *,
        personas: list[str] | None = None,
        health: Any = None,
    ) -> PanelResult:
        chosen = [p for p in (personas or choose_panel(state)) if p in PERSONAS][: self.max_members]
        result = PanelResult(convened=chosen)
        if not chosen:
            result.mode = "no_members"
            return result
        if self.llm is None or not getattr(self.llm, "available", False):
            result.mode = "llm_unavailable"
            return result

        workers = max(1, min(self.concurrency, len(chosen)))
        result.concurrency = workers
        if workers == 1:
            self._run_sequential(state, tools, registry, chosen, health, result)
        else:
            self._run_concurrent(state, tools, registry, chosen, health, result)
        return self.synthesise(result)

    # ------------------------------------------------------------- execution
    def _run_sequential(
        self,
        state: ClinicalRunState,
        tools: ToolRegistry,
        registry: Any,
        chosen: list[str],
        health: Any,
        result: PanelResult,
    ) -> None:
        for persona in chosen:
            member = ConsultSubagent(persona, self.llm, skill_id=self.skill_id)
            result.opinions.append(member.run(
                state, tools, registry, health=health,
                sub_budget=_carve_budget(state.budget, len(chosen)),
            ))

    def _run_concurrent(
        self,
        state: ClinicalRunState,
        tools: ToolRegistry,
        registry: Any,
        chosen: list[str],
        health: Any,
        result: PanelResult,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from .scope import MemberScope, merge_scopes

        members = [ConsultSubagent(p, self.llm, skill_id=self.skill_id) for p in chosen]
        scopes = [
            MemberScope(state, _carve_budget(state.budget, len(chosen)), label=member.label)
            for member in members
        ]

        def work(index: int) -> ConsultOpinion:
            member, scope = members[index], scopes[index]
            try:
                return member.run(state, tools, registry, health=health,
                                  sub_budget=scope.budget, scope=scope)
            except Exception as exc:  # noqa: BLE001 - a worker must never escape
                return ConsultOpinion(member.persona, member.label, ok=False,
                                      error=f"{type(exc).__name__}: {exc}"[:200])

        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(members)),
                                thread_name_prefix="yaobi-consult") as pool:
            # `map` preserves input order, so opinions come back in convened
            # order regardless of which member finished first.
            opinions = list(pool.map(work, range(len(members))))

        # Merge after every member has finished, in roster order: this is where
        # evidence ids are allocated, and it is why the ledger is reproducible.
        reports = merge_scopes(state, scopes)
        remaps = {report.member: report.evidence_remap for report in reports}
        for opinion in opinions:
            remap = remaps.get(opinion.label, {})
            opinion.evidence_ids = [remap.get(eid, eid) for eid in opinion.evidence_ids]
        result.opinions = opinions
        result.merge = [report.to_dict() for report in reports]

    @staticmethod
    def synthesise(result: PanelResult) -> PanelResult:
        """Conservative-first: maximum urgency, union of concerns.

        Agreement is reported, not applied. Four members calling something
        routine does not downgrade the one member who called it an emergency —
        in this direction the asymmetry of cost is the whole argument.
        """
        usable = [o for o in result.opinions if o.ok]
        if not usable:
            result.mode = "all_members_failed"
            return result

        result.urgency = max(usable, key=lambda o: URGENCY_ORDER.index(o.urgency)).urgency
        result.concerns = _dedupe(c for o in usable for c in o.concerns)
        result.recommend_next = _dedupe(r for o in usable for r in o.recommend_next)
        result.questions_for_patient = _dedupe(q for o in usable for q in o.questions_for_patient)
        result.dissents = _dedupe(o.dissent for o in usable if o.dissent)

        agreeing = sum(1 for o in usable if o.urgency == result.urgency)
        result.agreement = f"{agreeing}/{len(usable)} 位会诊者给出该紧急度"
        if agreeing < len(usable):
            result.dissents.insert(
                0,
                f"紧急度存在分歧，已按最保守意见采纳（{result.urgency}）；"
                f"{len(usable) - agreeing} 位会诊者判断更低。",
            )
        result.mode = "panel"
        return result


# ------------------------------------------------------------------- helpers
def _default_concurrency() -> int:
    """Threads to use, overridable for debugging or a rate-limited endpoint."""
    raw = os.environ.get("YAOBI_PANEL_CONCURRENCY")
    if not raw:
        return DEFAULT_CONCURRENCY
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return DEFAULT_CONCURRENCY


def _member_broker(
    state: ClinicalRunState,
    registry: Any,
    skill_id: str,
    granted: tuple[str, ...],
    *,
    health: Any = None,
    budget: Any = None,
) -> CapabilityBroker:
    """A broker that additionally enforces the member's narrowed tool set.

    The parent broker's checks all still run — this only ever subtracts. Wrapping
    rather than replacing matters: a member cannot reach a tool the run's role and
    risk mode already forbid, no matter what its consult mode says.
    """
    broker = CapabilityBroker(
        state.role, state.risk_mode,
        budget=budget if budget is not None else state.budget,
        skill_registry=registry, active_skill=skill_id, health=health,
    )
    allowed = set(granted)
    inner_allow = broker.allow

    def allow(tool: str) -> tuple[bool, str]:
        if tool not in allowed:
            return False, "consult_mode_denied:tool_outside_member_capability"
        return inner_allow(tool)

    broker.allow = allow  # type: ignore[method-assign]
    return broker


def _specs_for(names: tuple[str, ...]) -> list[Any]:
    from ..tools import tool_specs

    keep = set(names)
    return [spec for spec in tool_specs() if spec.name in keep]


def _carve_budget(parent: Any, members: int) -> Any:
    """Give a member a slice of the parent budget, charged back on completion.

    A member cannot overspend into the dose-safety checks that follow the panel,
    which is the practical reason a shared budget object would be wrong here.
    """
    from ..state import Budget

    if parent is None:
        return None
    members = max(1, members)
    remaining_calls = max(0, parent.max_llm_calls - parent.used_llm_calls)
    remaining_tokens = max(0, parent.max_llm_tokens - parent.used_llm_tokens)
    remaining_tools = max(0, parent.max_tool_calls - parent.used_tool_calls)

    slice_budget = Budget(
        max_loops=1,
        max_tool_calls=max(1, remaining_tools // (members * 2) or 1),
        max_questions=0,
        max_llm_calls=max(1, remaining_calls // (members * 2) or 1),
        max_llm_tokens=max(1000, remaining_tokens // (members * 2)),
    )
    slice_budget.parent = parent  # type: ignore[attr-defined]
    return _ChargebackBudget(slice_budget, parent)


class _ChargebackBudget:
    """A budget slice that also charges the parent, so totals stay honest.

    Without the chargeback the console would report a run that used 4 LLM calls
    when it really used 20, and the parent's own ceilings would stop meaning
    anything once a panel had run.
    """

    def __init__(self, slice_budget: Any, parent: Any) -> None:
        self._slice = slice_budget
        self._parent = parent

    def __getattr__(self, name: str) -> Any:
        return getattr(self._slice, name)

    def reserve_llm(self) -> bool:
        if not self._slice.reserve_llm():
            return False
        if not self._parent.reserve_llm():
            # The slice was charged for a call the parent then refused. Handing
            # it back matters under concurrency: several members hitting the
            # parent ceiling at once would otherwise each burn a slice call they
            # never spent, and their remaining slices would silently shrink.
            self._slice.refund_llm()
            return False
        return True

    def charge_llm_tokens(self, tokens: int) -> None:
        self._slice.charge_llm_tokens(tokens)
        self._parent.charge_llm_tokens(tokens)

    def can_afford_tool(self) -> bool:
        return self._slice.can_afford_tool() and self._parent.can_afford_tool()

    def charge_tool(self) -> None:
        self._slice.charge_tool()
        self._parent.charge_tool()


def _clamp_urgency(value: Any) -> str:
    text = str(value or "routine").strip().lower()
    return text if text in URGENCY_ORDER else "routine"


def _strings(value: Any, limit: int = 10) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:limit]


def _dedupe(values: Any, limit: int = 14) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))[:limit]


def panel_to_json(result: PanelResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False)
