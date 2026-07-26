"""Clinical agents.

Each agent implements :class:`Agent` and receives the run state, the tool
registry and a broker scoped to its own skill. Agents never authorise
themselves: every tool call goes through the broker, and every released
assertion is registered as a :class:`~yaobi_harness.state.Claim` bound to the
evidence that supports it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from ..safety import incompatibility as incompat
from ..safety import red_flags
from ..state import NON_RELEASABLE_LEVELS, ClinicalRunState
from ..tools import CapabilityBroker, ToolRegistry, ToolResult, herbs_in
from . import cognition

MINIMUM_INFO = [
    "起病时间", "疼痛部位/放射", "神经症状", "大小便/会阴感觉", "发热外伤肿瘤史",
    "妊娠/年龄/肝肾功能", "当前用药/过敏", "舌脉", "疼痛评分(VAS)", "功能受限(ODI)",
]

DEFAULT_QUESTIONS = [
    "疼痛什么时候开始的？是突然发生还是逐渐加重？",
    "有没有大小便困难、失禁或会阴部（骑车接触鞍座的区域）麻木？",
    "腿有没有越来越无力、走路不稳或抬不起脚？",
    "近期有没有外伤、跌倒、发热、体重下降或肿瘤病史？",
    "目前在吃什么药（尤其抗凝药）？有没有药物过敏？",
]


class Agent(Protocol):
    """Uniform agent contract; the runner never special-cases a signature."""

    name: str
    skill_id: str

    def run(self, state: ClinicalRunState, tools: ToolRegistry, broker: CapabilityBroker) -> ClinicalRunState:
        ...


class BaseAgent:
    name = "BaseAgent"
    skill_id = ""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def run(self, state: ClinicalRunState, tools: ToolRegistry, broker: CapabilityBroker) -> ClinicalRunState:
        raise NotImplementedError


def record_tool(state: ClinicalRunState, result: ToolResult) -> str:
    """Register a tool result in the evidence ledger at the level the *tool* declares."""
    return state.add_evidence(
        result.resolved_level(),
        result.tool,
        result.summary,
        result.data,
        ok=result.ok,
        error=result.error,
        source_version=result.source_version,
    )


def _finding_label(finding: dict[str, Any]) -> str:
    """Human-readable name for a rule-pack hit or a database interaction row."""
    title = finding.get("title")
    if title:
        return str(title)
    return f"{finding.get('subject', '?')}+{finding.get('object', '?')}"


def require_ok(state: ClinicalRunState, result: ToolResult, context: str) -> bool:
    if result.ok:
        return True
    state.fail_closed(f"关键工具失败: {context}:{result.tool}:{result.error or result.summary}")
    return False


# --------------------------------------------------------------------- intake

class TimelineAgent(BaseAgent):
    name = "TimelineAgent"
    skill_id = "yaobi.timeline"

    def run(self, state, tools, broker):
        events: list[dict[str, Any]] = [{"source": "chief_complaint", "text": state.complaint}]
        evidence_ids: list[str] = []
        research_id = state.facts.get("research_patient_id")
        if research_id:
            result = tools.call(broker, "patient_timeline_search", research_patient_id=research_id)
            evidence_ids.append(record_tool(state, result))
            if result.ok:
                events += [
                    {"source": "prior_visit", "month": v.get("就诊月份"), "diagnosis": v.get("中医诊断")}
                    for v in result.data.get("visits", [])
                ]
            else:
                state.warn(f"历史就诊检索不可用({result.summary})，时间线仅基于本次主诉")
        state.outputs["timeline"] = {
            "events": events,
            "limitations": "单轮输入；仅在提供 research_patient_id 时才有既往轨迹",
        }
        state.trace(self.name, "timeline", output_summary=f"{len(events)}个事件", evidence_ids=evidence_ids)
        return state


class IntakeAgent(BaseAgent):
    name = "IntakeAgent"
    skill_id = "yaobi.intake"

    def run(self, state, tools, broker):
        result = tools.call(broker, "red_flag_evidence_search", text=state.complaint)
        evidence_id = record_tool(state, result)
        if not require_ok(state, result, "intake_red_flag"):
            return state

        screening = dict(result.data)
        rule_hits = list(screening.get("hits", []))

        # The model may only *add* signals; it can never clear a rule-based hit.
        llm_signals = cognition.semantic_red_flag_signals(state, self.llm, rule_hits)
        if llm_signals:
            merged = red_flags.merge_llm_hits(
                red_flags.ScreenResult(
                    hits=[red_flags.RedFlagHit(**{**h, "source": h.get("source", "rule")}) for h in rule_hits],
                    soft_hits=[],
                ),
                llm_signals,
            )
            screening["hits"] = [h.to_dict() for h in merged.hits]
            screening["llm_added"] = len(merged.hits) - len(rule_hits)

        if screening.get("hits"):
            state.risk_mode = "urgent"
        elif screening.get("soft_hits"):
            state.warn("存在待证实的弱风险信号，建议线下评估以排除结构性病因")

        state.missing_information = [x for x in MINIMUM_INFO if x not in state.facts]
        allowed = state.budget.reserve_questions(3 if state.risk_mode == "urgent" else 5)
        state.open_questions = cognition.followup_questions(state, self.llm, DEFAULT_QUESTIONS, allowed)

        state.outputs["intake"] = {
            "risk_mode": state.risk_mode,
            "screening": screening,
            "missing_information": state.missing_information,
            "open_questions": state.open_questions,
        }
        for hit in screening.get("hits", []):
            state.add_claim("red_flag", f"{hit.get('signal')}:{hit.get('term')}", [evidence_id], confidence=0.9,
                            origin=hit.get("source", "rule"))
        state.trace(
            self.name, "screen_and_gap",
            output_summary=f"risk={state.risk_mode}; hits={len(screening.get('hits', []))}; missing={len(state.missing_information)}",
            evidence_ids=[evidence_id],
        )
        return state


# --------------------------------------------------------------------- urgent

class UrgentPlannerAgent(BaseAgent):
    name = "UrgentPlannerAgent"
    skill_id = "yaobi.urgent_triage"

    def run(self, state, tools, broker):
        hits = state.outputs.get("intake", {}).get("screening", {}).get("hits", [])
        hypotheses = sorted({h.get("signal", "unclassified_urgent_risk") for h in hits}) or ["unclassified_urgent_risk"]
        state.outputs["urgent_planner"] = {
            "dangerous_hypotheses": hypotheses,
            "ask_now": state.open_questions[:3] or DEFAULT_QUESTIONS[:3],
            "do_not_wait_for_answers": bool(hits),
            "max_questions": 3,
        }
        state.trace(self.name, "urgent_plan_tasks", output_summary=",".join(hypotheses))
        return state


class UrgentCareAgent(BaseAgent):
    name = "UrgentCareAgent"
    skill_id = "yaobi.urgent_triage"

    def run(self, state, tools, broker):
        guideline = tools.call(broker, "clinical_guideline_search", topic="acute low back pain and non-spine emergency red flags")
        guideline_id = record_tool(state, guideline)
        resource = tools.call(broker, "emergency_resource_lookup", location=state.facts.get("location", "中国大陆"))
        resource_id = record_tool(state, resource)
        if not (require_ok(state, guideline, "urgent_guideline") and require_ok(state, resource, "urgent_resource")):
            return state

        hypotheses = state.outputs.get("urgent_planner", {}).get("dangerous_hypotheses", [])
        phone = resource.data.get("emergency_phone")
        call_advice = f"请立即拨打{phone}" if phone else "请立即拨打当地官方急救电话"

        state.outputs["urgent_action_plan"] = {
            "risk_judgement": f"存在{hypotheses or ['急症']}风险信号，需要优先排除可致残或致命情况。",
            "why_urgent": "红旗信号不能通过线上问诊安全排除，延误可能导致神经功能损害或危及生命。",
            "immediate_action": f"现在不要等待完整线上问诊；{call_advice}，或由家属陪同立即去急诊。",
            "transport_advice": "尿潴留、会阴麻木、进行性无力、胸痛呼吸困难或晕厥时优先急救转运；不要自行驾车。",
            "during_transport": ["保持相对静止", "准备既往病历、影像、用药和过敏信息", "记录症状开始及进展时间"],
            "do_not": ["不要推拿、正骨、牵引", "不要自行加量止痛药或镇静药", "不要因线上建议延迟急诊"],
            "tell_clinicians": ["起病时间", "神经症状进展", "大小便和会阴感觉变化", "胸痛/呼吸困难/发热/外伤/肿瘤/抗凝用药史"],
            "possible_exams": ["生命体征和神经系统查体", "腰椎MRI/CT或心肺急诊检查按医师判断", "血常规、炎症指标等按疑点选择"],
            "key_questions": state.outputs.get("urgent_planner", {}).get("ask_now", [])[:3],
            "escalate_if": ["无力或麻木进展", "新发大小便异常", "发热寒战", "胸痛呼吸困难/意识改变"],
            "uncertainty": "这是风险分层与行动建议，不是确诊；线上不能排除严重疾病。",
        }
        state.add_claim(
            "urgent_action",
            "存在红旗信号，需线下急诊评估",
            [eid for eid in (guideline_id, resource_id) if state.evidence[eid].level not in NON_RELEASABLE_LEVELS],
            confidence=0.85,
        )
        state.release_status = "urgent_action_plan"
        state.trace(self.name, "urgent_action", evidence_ids=[guideline_id, resource_id], output_summary="动态急症行动计划")
        return state


# -------------------------------------------------------------------- routine

class BiomedicalAgent(BaseAgent):
    name = "BiomedicalAgent"
    skill_id = "yaobi.biomedical_differential"

    DIFFERENTIALS = [
        "非特异性腰痛/腰肌劳损",
        "腰椎间盘突出伴神经根病",
        "腰椎管狭窄",
        "腰椎滑脱/峡部裂",
        "骶髂关节源性疼痛",
        "髋关节病变牵涉痛",
        "脊柱感染/肿瘤/骨折需按红旗排除",
    ]

    def run(self, state, tools, broker):
        result = tools.call(broker, "clinical_guideline_search", topic="low back pain differential")
        evidence_id = record_tool(state, result)
        if not require_ok(state, result, "biomedical_guideline"):
            return state
        state.outputs["biomedical"] = {
            "differentials": self.DIFFERENTIALS,
            "exam_advice": [
                "神经定位体检（肌力MRC分级、感觉平面、腱反射、直腿抬高/股神经牵拉）",
                "红旗或持续神经根症状时线下影像（X线/MRI按适应证）",
                "记录VAS疼痛评分与ODI功能评分作为随访基线",
            ],
            "evidence_note": "当前指南源为占位数据，鉴别列表属模型/规则推理，未获授权指南背书",
        }
        for item in self.DIFFERENTIALS:
            state.add_claim("differential", item, [evidence_id], confidence=0.4)
        state.trace(self.name, "differential", evidence_ids=[evidence_id])
        return state


class TCMPatternAgent(BaseAgent):
    name = "TCMPatternAgent"
    skill_id = "yaobi.tcm_pattern"

    def run(self, state, tools, broker):
        result = tools.call(broker, "tcm_pattern_knowledge_search", text=state.complaint)
        evidence_id = record_tool(state, result)
        if not require_ok(state, result, "tcm_pattern"):
            return state
        patterns = result.data["patterns"]
        state.outputs["tcm_pattern"] = {
            "primary_pattern": patterns[0],
            "candidate_patterns": patterns,
            "counter_evidence_needed": ["寒热表现", "舌脉", "疼痛固定或游走", "乏力与夜痛"],
            "confidence": 0.35 if patterns == ["待辨证"] else 0.5,
        }
        state.add_claim("pattern", f"主要证型倾向: {patterns[0]}", [evidence_id], confidence=0.4)
        state.trace(self.name, "pattern", evidence_ids=[evidence_id])
        return state


class ExpertCaseAgent(BaseAgent):
    name = "ExpertCaseAgent"
    skill_id = "yaobi.expert_case_reasoning"

    def run(self, state, tools, broker):
        similar = tools.call(broker, "similar_case_search", query=state.complaint)
        counter = tools.call(broker, "counterexample_case_search", query=state.complaint)
        evidence_ids = [record_tool(state, similar), record_tool(state, counter)]
        if not (similar.ok and counter.ok):
            state.fail_closed("专家病例检索失败")
            return state
        state.outputs["expert_cases"] = {
            "similar": similar.data.get("cases", []),
            "counterexamples": counter.data.get("cases", []),
            "limitation": "已脱敏的单一专家经验库，不代表因果疗效证据",
        }
        state.trace(self.name, "retrieve_cases", evidence_ids=evidence_ids)
        return state


class MedicationSafetyAgent(BaseAgent):
    """Screens the patient's existing western medications.

    This runs for every role on the routine path, because the highest-value
    finding a musculoskeletal service can make is often not the diagnosis but
    "the NSAID you are taking alongside your warfarin is a bleeding risk". A
    blocking finding escalates the run to ``needs_examination`` rather than
    letting ordinary advice go out unqualified.
    """

    name = "MedicationSafetyAgent"
    skill_id = "yaobi.medication_safety"

    def run(self, state, tools, broker):
        medications = [str(m) for m in state.facts.get("medications", []) if str(m).strip()]
        conditions = [str(c) for c in state.facts.get("conditions", []) if str(c).strip()]
        if not medications:
            state.outputs["medication_safety"] = {
                "findings": [],
                "medications_reviewed": [],
                "note": "未提供当前用药清单；无法进行相互作用审查",
                "reviewed": False,
            }
            state.missing_information = list(dict.fromkeys(state.missing_information + ["当前用药清单"]))
            state.trace(self.name, "no_medication_list")
            return state

        result = tools.call(
            broker, "drug_interaction_check",
            medications=medications, conditions=conditions,
            include_herbs=herbs_in(state.outputs.get("formula", {}).get("herbs", [])),
        )
        evidence_id = record_tool(state, result)
        if not require_ok(state, result, "drug_interaction"):
            return state

        findings = result.data.get("rule_findings", []) + result.data.get("database_findings", [])
        blocking = result.data.get("blocking", [])
        state.outputs["medication_safety"] = {
            "findings": findings,
            "blocking": blocking,
            "medications_reviewed": medications,
            "conditions_considered": result.data.get("conditions", []),
            "coverage_note": result.data.get("coverage_note", ""),
            "reviewed": True,
        }
        for finding in findings:
            state.add_claim(
                "drug_interaction",
                f"{finding.get('title') or finding.get('subject')} [{finding.get('severity')}]",
                [evidence_id], confidence=0.7,
            )
        if blocking:
            state.safety_issues += [f"用药安全: {_finding_label(f)} ({f.get('severity')})" for f in blocking]
            if state.release_status in ("needs_more_information", "treatment_advice_only"):
                state.release_status = "needs_examination"
        state.trace(self.name, "interaction_screen", evidence_ids=[evidence_id],
                    output_summary=f"{len(findings)}条发现, {len(blocking)}条需阻断")
        return state


class FormulaAgent(BaseAgent):
    name = "FormulaAgent"
    skill_id = "yaobi.formula_design"

    def run(self, state, tools, broker):
        pattern = state.outputs.get("tcm_pattern", {}).get("primary_pattern", "气血痹阻证")
        result = tools.call(broker, "formula_composition_search", pattern=pattern)
        evidence_id = record_tool(state, result)
        if not require_ok(state, result, "formula_search"):
            return state
        herbs = result.data["herbs"]

        # 十八反/十九畏 are absolute: a violating candidate never reaches dosing.
        violations = incompat.check_combination(herbs)
        if violations:
            state.safety_issues += [f"候选方配伍禁忌: {v['detail']}" for v in violations]
            state.release_status = "treatment_advice_only"
            state.trace(self.name, "formula_blocked", evidence_ids=[evidence_id], output_summary="配伍禁忌")
            return state

        state.outputs["formula"] = {
            "formula_name": result.data["formula_name"],
            "treatment_principle": ["补益肝肾", "活血通络", "祛风除湿"],
            "herbs": herbs,
            "combination_check": "十八反/十九畏通过",
        }
        state.trace(self.name, "formula_candidates", evidence_ids=[evidence_id])
        return state


class DoseAgent(BaseAgent):
    """Generates per-herb doses only when every safety gate passes.

    Gates: stratified expert-case support, minimum sample size, dispersion,
    authorised pharmacopoeia range **containment of the proposed dose**,
    special-population clearance, confirmed medication/allergy history,
    herb-herb 十八反/十九畏, pregnancy contraindications and risk-herb review.
    """

    name = "DoseAgent"
    skill_id = "yaobi.dose_generation"

    def run(self, state, tools, broker):
        herbs = herbs_in(state.outputs.get("formula", {}).get("herbs", []))
        if not herbs:
            state.safety_issues.append("无候选方药味，无法生成剂量")
            state.release_status = "insufficient_evidence"
            return state

        population = dict(state.facts.get("special_population", {}))
        pattern = state.outputs.get("tcm_pattern", {}).get("primary_pattern")

        distribution = tools.call(broker, "herb_dose_distribution", herbs=herbs, pattern=pattern, age=population.get("age"))
        special = tools.call(broker, "special_population_check", **population)
        interaction = tools.call(
            broker, "interaction_check",
            herbs=herbs,
            medications=state.facts.get("medications", []),
            allergies=state.facts.get("allergies", []),
            medications_confirmed=state.facts.get("medications_confirmed", False),
            allergies_confirmed=state.facts.get("allergies_confirmed", False),
            pregnancy=population.get("pregnancy"),
        )
        evidence_ids = [record_tool(state, distribution), record_tool(state, special), record_tool(state, interaction)]
        if not all(r.ok for r in (distribution, special, interaction)):
            state.fail_closed("处方安全关键工具失败")
            return state

        stats = distribution.data.get("distributions", {})
        candidate_doses = {h: stats.get(h, {}).get("median_g") for h in herbs}
        proposed = {h: d for h, d in candidate_doses.items() if d is not None}

        # The pharmacopoeia check must see the doses we intend to draft.
        pharmacopeia = tools.call(broker, "pharmacopeia_check", herbs=herbs, doses=proposed)
        evidence_ids.append(record_tool(state, pharmacopeia))
        if not require_ok(state, pharmacopeia, "pharmacopeia"):
            return state

        checked = {c["herb"]: c for c in pharmacopeia.data.get("checked", [])}
        blockers = self._collect_blockers(herbs, stats, checked, special, interaction)
        dose_safety = {
            "candidate_doses_g": proposed,
            "authorized_ranges": {h: checked.get(h, {}).get("range_g") for h in herbs},
            "out_of_range": [h for h, c in checked.items() if c.get("dose_within_range") is False],
            "blockers": blockers,
            "combination_violations": interaction.data.get("combination_violations", []),
            "pregnancy_violations": interaction.data.get("pregnancy_violations", []),
        }
        state.outputs["dose_safety"] = dose_safety

        if blockers:
            state.safety_issues += blockers
            state.release_status = "treatment_advice_only"
            state.trace(self.name, "dose_blocked", evidence_ids=evidence_ids, output_summary=f"{len(blockers)}项未通过")
            return state

        draft_herbs = [
            {
                "herb_name": herb,
                "dose_value": proposed[herb],
                "dose_unit": "g",
                "processing": "需药师/医师确认炮制",
                "administration": "常规煎服",
                "clinical_role": "按治法配伍",
                "dose_evidence_ids": [evidence_ids[0], evidence_ids[-1]],
                "authorized_range_g": checked.get(herb, {}).get("range_g"),
                "sample_n": stats.get(herb, {}).get("n"),
                "risk_flags": checked.get(herb, {}).get("risk_flags", []),
                "physician_status": "pending",
            }
            for herb in herbs
        ]
        draft = {
            "formula_name": state.outputs["formula"]["formula_name"],
            "herbs": draft_herbs,
            "decoction": {"frequency": "每日1剂", "times_per_day": 2, "duration_days": 7},
            "overall_uncertainty": self._uncertainty(stats, herbs),
            "requires_physician_approval": True,
            "dose_safety": dose_safety,
        }
        draft["prescription_hash"] = hashlib.sha256(
            json.dumps(draft_herbs, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:32]
        state.outputs["prescription_draft"] = draft
        for herb in draft_herbs:
            state.add_claim(
                "dose", f"{herb['herb_name']} {herb['dose_value']}g",
                herb["dose_evidence_ids"], confidence=0.5,
            )
        state.release_status = "draft_for_physician"
        state.trace(self.name, "dose_generated", evidence_ids=evidence_ids)
        return state

    @staticmethod
    def _collect_blockers(herbs, stats, checked, special, interaction) -> list[str]:
        blockers: list[str] = []
        no_evidence = [h for h in herbs if not stats.get(h, {}).get("median_g")]
        low_n = [h for h in herbs if stats.get(h, {}).get("median_g") and not stats.get(h, {}).get("meets_min_n")]
        outliers = [h for h in herbs if stats.get(h, {}).get("outlier_flag")]
        wide = [h for h in herbs if stats.get(h, {}).get("wide_dispersion")]
        no_range = [h for h in herbs if not checked.get(h, {}).get("authorized_range_available")]
        out_of_range = [h for h, c in checked.items() if c.get("dose_within_range") is False]
        unchecked = [h for h, c in checked.items() if c.get("dose_checked") is False]
        risk = [h for h, c in checked.items() if any(f.startswith("risk_herb") for f in c.get("risk_flags", []))]

        if no_evidence:
            blockers.append(f"缺少分层剂量证据: {no_evidence}")
        if low_n:
            blockers.append(f"专家样本量不足(<3例): {low_n}")
        if outliers:
            blockers.append(f"剂量异常值/超出授权范围倍数: {outliers}")
        if wide:
            blockers.append(f"剂量离散度过大，中位数不可靠: {wide}")
        if no_range:
            blockers.append(f"缺少授权药典范围: {no_range}")
        if out_of_range:
            blockers.append(f"拟用剂量超出授权药典范围: {out_of_range}")
        if unchecked:
            blockers.append(f"未提交剂量做药典校验: {unchecked}")
        if risk:
            blockers.append(f"风险药需专项审查: {risk}")
        if not special.data.get("pass"):
            blockers.append(f"特殊人群必要信息缺失或风险: {special.data}")
        if not interaction.data.get("pass"):
            blockers.append(f"配伍/相互作用/过敏/妊娠风险: {interaction.data.get('risk_flags')}")
        return blockers

    @staticmethod
    def _uncertainty(stats: dict[str, Any], herbs: list[str]) -> str:
        sample_sizes = [stats.get(h, {}).get("n", 0) for h in herbs]
        smallest = min(sample_sizes) if sample_sizes else 0
        if smallest >= 20:
            return "low"
        return "medium" if smallest >= 8 else "high"


class PhysicianReviewAgent(BaseAgent):
    """Closes the loop: a draft only becomes releasable after signed review."""

    name = "PhysicianReviewAgent"
    skill_id = "yaobi.physician_review"

    def run(self, state, tools, broker):
        draft = state.outputs.get("prescription_draft")
        if not draft:
            state.outputs["physician_review"] = {
                "approved": False,
                "problems_checked": ["no_draft_to_review"],
                "note": "无处方草案，无需审核",
            }
            return state

        review_input = state.facts.get("physician_review", {})
        approvals = review_input.get("approvals") or {}
        physician_id = review_input.get("physician_id")
        signature = review_input.get("signature")
        if not (approvals and physician_id and signature):
            state.outputs["physician_review"] = {
                "approved": False,
                "problems_checked": ["awaiting_physician_signature_and_per_herb_approval"],
                "note": "等待医师逐味审核与签名；草案不得作为最终处方使用",
            }
            state.trace(self.name, "await_review", output_summary="等待医师签名")
            return state

        result = tools.call(
            broker, "physician_review_submit",
            prescription=draft, approvals=approvals, physician_id=physician_id, signature=signature,
        )
        evidence_id = record_tool(state, result)
        state.outputs["physician_review"] = {
            "approved": bool(result.data.get("approved")),
            "problems_checked": result.data.get("problems", []),
            "physician_id": physician_id,
        }
        if result.data.get("approved"):
            state.release_status = "approved_by_physician"
            state.requires_physician_approval = False
        else:
            state.safety_issues += [f"医师审核未通过: {p}" for p in result.data.get("problems", [])]
        state.trace(self.name, "physician_review", evidence_ids=[evidence_id],
                    output_summary="approved" if result.data.get("approved") else "rejected")
        return state


# --------------------------------------------------------------------- critic

class CriticAgent(BaseAgent):
    """Terminal safety gate. Runs on every path, including failed ones."""

    name = "CriticAgent"
    skill_id = "yaobi.safety_critic"

    def run(self, state, tools=None, broker=None):
        issues: list[str] = []
        checks_run: list[str] = []
        repair_requests: list[dict[str, str]] = []
        draft = state.outputs.get("prescription_draft", {})

        checks_run.append("urgent_mode_no_prescription")
        if state.risk_mode == "urgent" and draft:
            issues.append("急症模式不得发布处方")

        checks_run.append("role_permission")
        if state.release_status in ("draft_for_physician", "approved_by_physician") and state.role != "physician":
            issues.append("非医师角色不得生成或放行含剂量处方")

        checks_run.append("physician_approval_flag")
        if state.release_status == "draft_for_physician" and not draft.get("requires_physician_approval"):
            issues.append("处方草案缺少医师审核标记")

        checks_run.append("failed_tool_evidence")
        if any(not e.payload.get("tool_ok", True) for e in state.evidence.values()) and state.release_status not in {
            "failed_closed", "blocked"
        }:
            issues.append("存在失败工具证据，不能发布")

        checks_run.append("citation_guard")
        uncited = self._citation_guard(state)
        if uncited:
            state.warn(f"以下结论缺少可放行证据支撑，仅作提示性内容: {[u['text'] for u in uncited][:5]}")
        blocking = [u["text"] for u in uncited if u["kind"] in self.HIGH_STAKES_CLAIM_KINDS]
        if blocking:
            issues.append(f"高风险结论缺少可放行证据: {blocking[:5]}")

        checks_run.append("independent_dose_range_recheck")
        out_of_range = self._recheck_doses(draft)
        if out_of_range:
            issues.append(f"独立复核发现剂量超出授权范围: {out_of_range}")
            repair_requests.append({"agent": "DoseAgent", "reason": "dose_out_of_authorized_range"})

        checks_run.append("independent_combination_recheck")
        violations = incompat.check_combination([h.get("herb_name", "") for h in draft.get("herbs", [])])
        if violations:
            issues.append(f"独立复核发现配伍禁忌: {[v['detail'] for v in violations]}")
            repair_requests.append({"agent": "FormulaAgent", "reason": "combination_contraindication"})

        checks_run.append("red_flag_recheck")
        if state.risk_mode == "routine":
            rescreen = red_flags.screen(" ".join([state.complaint, json.dumps(state.facts, ensure_ascii=False)]))
            if rescreen.hits:
                issues.append(f"复核阶段发现未处理的红旗信号: {[h.signal for h in rescreen.hits][:3]}")
                repair_requests.append({"agent": "IntakeAgent", "reason": "late_red_flag_detected"})

        checks_run.append("llm_adversarial_review")
        for item in cognition.adversarial_issues(state, self.llm):
            if item["severity"] == "block":
                issues.append(f"[LLM审查] {item['issue']}")
            else:
                state.warn(f"[LLM审查] {item['issue']}")

        state.safety_issues += issues
        state.outputs["safety_audit"] = {
            "checks_run": checks_run,
            "issues": issues,
            "uncited_claims": uncited,
            "repair_requests": repair_requests,
        }
        if issues and state.release_status != "failed_closed":
            state.release_status = "blocked"
        state.trace(self.name, "audit", output_summary=";".join(issues) or "未发现阻断问题")
        return state

    #: Claim kinds that block release when they lack releasable evidence.
    #: Lower-stakes claims (differentials, pattern hypotheses) are downgraded to
    #: advisory content instead, so a placeholder knowledge base does not stall
    #: the whole run while still never masquerading as evidence.
    HIGH_STAKES_CLAIM_KINDS = {"dose", "urgent_action"}

    @staticmethod
    def _citation_guard(state: ClinicalRunState) -> list[dict[str, str]]:
        """Claims must rest on at least one releasable piece of evidence."""
        releasable = state.releasable_evidence_ids()
        return [
            {"kind": c.kind, "text": f"{c.kind}:{c.text}"}
            for c in state.claims
            if not (set(c.evidence_ids) & releasable)
        ]

    @staticmethod
    def _recheck_doses(draft: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for herb in draft.get("herbs", []):
            rng = herb.get("authorized_range_g")
            dose = herb.get("dose_value")
            if rng and dose is not None and not (rng[0] <= float(dose) <= rng[1]):
                out.append(f"{herb.get('herb_name')}:{dose}g∉{rng}")
        return out
