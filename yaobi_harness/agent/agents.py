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
from .toolloop import ToolLoop

#: Patient-facing names for the internal red-flag signal keys. Without these a
#: raw Python list repr leaks into the text a patient reads.
SIGNAL_NAMES = {
    "cauda_equina": "马尾神经受压",
    "cardiopulmonary": "心肺急症",
    "vascular_dvt_pe": "深静脉血栓/肺栓塞",
    "infection_or_tumor": "感染或肿瘤",
    "septic_joint_or_osteomyelitis": "化脓性关节炎/骨髓炎",
    "fracture": "骨折或外伤",
    "progressive_neuro": "进行性神经功能缺损",
    "cervical_myelopathy": "脊髓型颈椎病",
    "compartment_syndrome": "骨筋膜室综合征",
    "night_pain": "夜间痛",
    "chronic_bone_fragility": "骨质疏松/骨脆性",
    "age_extremes": "高龄因素",
    "unclassified_urgent_risk": "未分类急症风险",
}


def signal_text(signals: list[str]) -> str:
    """Render internal signal keys as a readable Chinese phrase."""
    names = [SIGNAL_NAMES.get(s, s) for s in signals] or ["急症"]
    return "、".join(dict.fromkeys(names))


#: Information gap -> the fact keys that satisfy it.
#:
#: Gap labels are human-readable and were previously compared directly against
#: ``state.facts`` keys, which never matched — so every gap stayed "missing"
#: no matter what the patient had already told us, and the dialogue re-asked
#: questions it had just had answered.
INFORMATION_GAPS: dict[str, tuple[str, ...]] = {
    "起病时间": ("onset",),
    "疼痛部位/放射": ("pain_location",),
    "神经症状": ("neuro_symptoms",),
    "大小便/会阴感觉": ("bowel_bladder",),
    "发热外伤肿瘤史": ("fever_trauma_tumor",),
    "妊娠/年龄/肝肾功能": ("pregnancy", "age", "renal", "liver"),
    "当前用药/过敏": ("medications_confirmed", "allergies_confirmed"),
    "舌脉": ("four_diagnoses",),
    "疼痛评分(VAS)": ("vas",),
    "功能受限(ODI)": ("odi",),
}

MINIMUM_INFO = list(INFORMATION_GAPS)


def missing_information(facts: dict[str, Any]) -> list[str]:
    """Gaps still open, given what we already know.

    Special-population answers live in a nested dict for the dose pipeline, so
    they are flattened here before checking.
    """
    known = dict(facts)
    known.update(facts.get("special_population") or {})
    return [
        gap for gap, keys in INFORMATION_GAPS.items()
        if not all(known.get(key) is not None for key in keys)
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
    #: Schema the model must satisfy when this agent runs autonomously.
    output_schema = ""
    #: Where this agent writes its result in ``state.outputs``.
    output_key = ""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def run(self, state: ClinicalRunState, tools: ToolRegistry, broker: CapabilityBroker) -> ClinicalRunState:
        raise NotImplementedError

    # ------------------------------------------------------------- autonomy
    def skill_spec(self, broker: CapabilityBroker) -> Any | None:
        registry = getattr(broker, "skill_registry", None)
        return registry.specs.get(self.skill_id) if registry else None

    def autonomous(
        self,
        state: ClinicalRunState,
        tools: ToolRegistry,
        broker: CapabilityBroker,
        objective: str,
        context: dict[str, Any],
    ) -> Any | None:
        """Try the model-driven tool loop; return None to use the fallback.

        The skill decides whether this is even attempted (``autonomous: true``),
        which keeps the choice in the reviewed policy file rather than in code.
        """
        spec = self.skill_spec(broker)
        if spec is None or not getattr(spec, "autonomous", False):
            return None
        loop = ToolLoop(
            self.llm, tools, broker, state,
            agent_name=self.name, skill_id=self.skill_id, skill_spec=spec,
        )
        if not loop.available:
            return None
        result = loop.run(objective, context, spec.output_schema or self.output_schema)
        state.outputs.setdefault("autonomy", {})[self.name] = result.to_dict()
        return result if result.ok else None

    def bind_autonomous_output(self, state: ClinicalRunState, result: Any, kind: str) -> list[str]:
        """Record the model's answer, citing the evidence it actually gathered."""
        evidence_ids = result.citations or result.evidence_ids
        output = dict(result.output or {})
        output["_produced_by"] = "llm_tool_loop"
        output["_evidence_ids"] = evidence_ids
        state.outputs[self.output_key] = output
        state.trace(
            self.name, "autonomous_run",
            output_summary=f"{len(result.steps)}步/{len(result.evidence_ids)}次取证",
            evidence_ids=evidence_ids,
        )
        return evidence_ids


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

        state.missing_information = missing_information(state.facts)
        # Questioning belongs to InterviewAgent, which composes it from the axis
        # table and has the model word it. Generating a second, unrelated set here
        # spent an LLM call and a slice of the question budget on questions the
        # interview then overwrote. Intake now records the gaps and stops there;
        # the runner backfills questions only if no interview ran.

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


class InterviewAgent(BaseAgent):
    """Model-driven history taking, bounded by rule-derived axes.

    Sits between intake and everything else: intake decides *whether this is an
    emergency*, the interview decides *whether we know enough to proceed*. Making
    it a graph node rather than a chat-only concern means a single-shot ``run``
    also reports its history deficit, instead of quietly assessing a case on four
    facts.
    """

    name = "InterviewAgent"
    skill_id = "yaobi.interview"
    output_schema = "InterviewProgress"
    output_key = "interview"

    def __init__(self, llm: Any | None = None, loop: Any | None = None) -> None:
        super().__init__(llm)
        #: Shared across turns by :class:`~yaobi_harness.conversation.ConversationSession`
        #: so stall detection spans the conversation rather than one call.
        self.loop = loop

    def interview_loop(self, broker: CapabilityBroker) -> Any:
        from ..interview.loop import InterviewLoop

        if self.loop is None:
            self.loop = InterviewLoop(self.llm, skill_spec=self.skill_spec(broker))
        return self.loop

    def run(self, state, tools, broker):
        from ..interview.axes import coverage

        loop = self.interview_loop(broker)
        # 四诊 completeness gates a dose-bearing draft, so it is required only when
        # this run may actually produce one.
        prescriptive = state.role == "physician" and state.allow_prescription
        round_result = loop.next_round(
            state.facts, state.complaint,
            role=state.role, risk_mode=state.risk_mode,
            prescriptive=prescriptive, budget=state.budget,
        )
        verdict = round_result.verdict
        report = coverage(state.facts, state.complaint, role=state.role)

        state.outputs[self.output_key] = {
            "coverage": report,
            "questions": [q.to_dict() for q in round_result.questions],
            "verdict": verdict.to_dict() if verdict else {},
            "rounds_used": loop.rounds_used,
            "composer": round_result.composer,
            "rejected": round_result.rejected,
            "model_claimed_complete": round_result.model_claimed_complete,
            "_produced_by": "llm_interview_loop" if round_result.composer == "llm" else "probe_bank",
        }
        # The interview's questions become the run's open questions, so every
        # surface (CLI, console, chat) shows the same enquiry.
        asked = [q.question for q in round_result.questions]
        if asked:
            allowed = state.budget.reserve_questions(len(asked))
            state.open_questions = asked[: allowed or len(asked)]

        if verdict is not None:
            if verdict.verdict == "blocked":
                labels = "、".join(verdict.to_dict()["blocking_labels"]) or "必答项缺失"
                state.warn(f"问诊必答项未闭合，不能进入含剂量或处方环节: {labels}")
                state.safety_issues.append(
                    "问诊充分性判定为 blocked：" + (verdict.reason or "必答问诊轴未获答复")
                )
            elif verdict.deficit:
                state.warn(f"问诊带缺口继续({verdict.verdict}): {verdict.reason}")
            for contradiction in verdict.contradictions:
                state.warn(f"病史存在需澄清之处: {contradiction}")

        state.trace(
            self.name, "interview_round",
            output_summary=(
                f"round={loop.rounds_used}; composer={round_result.composer}; "
                f"verdict={verdict.verdict if verdict else 'n/a'}; "
                f"coverage={report['ratio']}"
            ),
        )
        return state


class ConsultPanelAgent(BaseAgent):
    """Convenes the multi-speciality panel and records its combined view.

    Deliberately *not* wired into the release-status machine: the panel raises
    concerns and can raise urgency, but a status change still comes from the
    ordinary screening and safety machinery. A panel of models must not be able
    to talk the run into a more permissive outcome — only a more cautious one.
    """

    name = "ConsultPanelAgent"
    skill_id = "yaobi.consult_panel"
    output_schema = "ConsultOpinion"
    output_key = "consult_panel"

    def run(self, state, tools, broker):
        from .panel import ConsultPanel

        registry = getattr(broker, "skill_registry", None)
        if registry is None or self.skill_id not in getattr(registry, "specs", {}):
            state.warn("未登记会诊技能，跳过多学科会诊")
            return state

        panel = ConsultPanel(self.llm, skill_id=self.skill_id)
        result = panel.run(state, tools, registry, health=broker.health)
        state.outputs[self.output_key] = result.to_dict()

        if result.mode != "panel":
            state.trace(self.name, "panel_skipped", output_summary=result.mode)
            return state

        # Urgency may only be raised. A panel that thinks a rule-flagged
        # emergency is routine changes nothing.
        if result.urgency in ("urgent", "emergency") and state.risk_mode != "urgent":
            state.risk_mode = "urgent"
            state.warn(f"多学科会诊上调紧急度为 {result.urgency}（最保守优先），已切换急症模式")
        for concern in result.concerns[:6]:
            state.warn(f"会诊关切: {concern}")
        for dissent in result.dissents[:3]:
            state.warn(f"会诊分歧: {dissent}")

        state.trace(
            self.name, "panel",
            output_summary=f"{len(result.opinions)}位会诊者; urgency={result.urgency}; {result.agreement}",
            evidence_ids=[eid for o in result.opinions for eid in o.evidence_ids],
        )
        return state


class VisionAgent(BaseAgent):
    """Reads the images attached to a run. Descriptive, never diagnostic.

    Runs deterministically rather than as a tool loop: there is exactly one call
    to make per image, so a ReAct loop would add latency and a chance of drift
    without adding a decision. What the *model* contributes here is the reading
    itself, inside :mod:`yaobi_harness.vision`.
    """

    name = "VisionAgent"
    skill_id = "yaobi.vision_read"
    output_schema = "ImageFindings"
    output_key = "image_findings"

    #: Vision may escalate, so a surface finding maps onto a screening signal.
    URGENT_KEYWORDS = {
        "cauda_equina": ("鞍区", "会阴", "失禁"),
        "vascular_dvt_pe": ("发紫", "苍白", "花斑", "肿胀", "张力"),
        "compartment_syndrome": ("张力", "水疱", "苍白", "肌腹"),
        "infection_or_tumor": ("红肿", "窦道", "脓", "坏死", "分界"),
        "fracture": ("畸形", "成角", "短缩", "骨皮质中断"),
    }

    def run(self, state, tools, broker):
        images = list(state.images or [])
        if not images:
            return state

        reads: list[dict[str, Any]] = []
        evidence_ids: list[str] = []
        for image in images[:6]:
            result = tools.call(
                broker, "medical_image_read",
                image=str(image.get("ref") or ""),
                kind=str(image.get("kind") or "other"),
                context=(state.complaint or "")[:600],
                # Attestation travels with the image, so the run records who
                # asserted de-identification rather than the tool assuming it.
                deidentified=bool(image.get("deidentified")),
            )
            evidence_ids.append(record_tool(state, result))
            if not result.ok:
                state.warn(f"图片判读未完成: {result.summary}")
                continue
            payload = dict(result.data)
            reads.append(payload)
            if payload.get("phi_detected"):
                state.warn(
                    "上传图片含可识别身份信息，已拒绝判读并丢弃结果。请遮盖姓名/ID/日期/条码/人脸后重传。"
                )
                state.safety_issues.append("影像通道检出未去标识化图片，已拒绝判读")
                continue
            for signal in payload.get("urgent_signals", []):
                state.warn(f"图片可见急症外观信号: {signal}")

        signals = self._signals(reads)
        if signals:
            # Vision widens caution only. It can add a signal and switch the run
            # to urgent mode; it can never clear one the rules already raised.
            state.risk_mode = "urgent"
            state.warn("图片外观提示急症风险，已升级为急症模式: " + signal_text(sorted(signals)))
            screening = state.outputs.setdefault("intake", {}).setdefault("screening", {})
            screening.setdefault("hits", []).extend(
                {"signal": s, "term": "image_finding", "source": "vision", "tier": "hard"} for s in sorted(signals)
            )

        state.outputs[self.output_key] = {
            "image_kind": reads[0].get("image_kind", "other") if reads else "none",
            "readable": bool(reads and reads[0].get("readable")),
            "observations": [o for r in reads for o in r.get("observations", [])][:20],
            "not_assessable": [n for r in reads for n in r.get("not_assessable", [])][:12],
            "urgent_signals": [u for r in reads for u in r.get("urgent_signals", [])][:8],
            "suggest_ask": [s for r in reads for s in r.get("suggest_ask", [])][:8],
            "suggest_exam": [s for r in reads for s in r.get("suggest_exam", [])][:8],
            "caveat": "模型视觉判读，证据等级为 model_reasoning，不能替代正式阅片或体格检查",
            "requires_formal_read": True,
            "reads": reads,
            "_evidence_ids": evidence_ids,
        }
        state.trace(
            self.name, "read_images",
            output_summary=f"{len(reads)}/{len(images)} 张完成判读; 升级信号={sorted(signals) or '无'}",
            evidence_ids=evidence_ids,
        )
        return state

    def _signals(self, reads: list[dict[str, Any]]) -> set[str]:
        found: set[str] = set()
        for read in reads:
            for text in read.get("urgent_signals", []):
                for signal, keywords in self.URGENT_KEYWORDS.items():
                    if any(k in str(text) for k in keywords):
                        found.add(signal)
        return found


class OsteoporosisAgent(BaseAgent):
    """Fragility-fracture risk, FRAX element capture and drug prerequisites."""

    name = "OsteoporosisAgent"
    skill_id = "yaobi.osteoporosis_risk"
    output_schema = "BiomedicalAssessment"
    output_key = "osteoporosis_risk"

    FALLBACK_DIFFERENTIALS = [
        "病理性骨折需先排除（转移瘤、多发性骨髓瘤）",
        "骨软化症",
        "原发性甲状旁腺功能亢进",
        "Paget 骨病",
        "原发性骨质疏松（绝经后/老年性）",
        "糖皮质激素相关骨质疏松",
    ]
    FALLBACK_EXAMS = [
        "血校正钙、25-OH 维生素 D（启动抗骨吸收治疗前必须）",
        "肾功能 eGFR（决定双膦酸盐可否使用）",
        "碱性磷酸酶、血磷（排除骨软化与 Paget 病）",
        "DXA 腰椎+股骨颈，记录 T 值与测量日期",
        "胸腰段侧位片或 VFA，查找无症状椎体骨折",
        "身高测量并与年轻时最高身高比较（下降 >4cm 提示椎体骨折）",
        "近一年跌倒次数与镇静/抗胆碱类用药审查",
    ]

    def run(self, state, tools, broker):
        result = self.autonomous(
            state, tools, broker,
            objective="评估本例的骨质疏松与脆性骨折风险，采集 FRAX 要素，给出检查与用药前置条件。",
            context={
                "chief_complaint": state.complaint,
                "known_facts": {k: v for k, v in state.facts.items() if k != "physician_review"},
                "interview_coverage": (state.outputs.get("interview") or {}).get("coverage", {}),
                "image_findings": (state.outputs.get("image_findings") or {}).get("observations", []),
            },
        )
        if result is not None:
            evidence_ids = self.bind_autonomous_output(state, result, "osteoporosis")
            for item in state.outputs[self.output_key].get("differentials", [])[:8]:
                state.add_claim("differential", str(item), evidence_ids, confidence=0.5, origin="llm")
            return state

        guideline = tools.call(broker, "clinical_guideline_search", topic="osteoporosis fracture risk")
        evidence_id = record_tool(state, guideline)
        state.outputs[self.output_key] = {
            "differentials": self.FALLBACK_DIFFERENTIALS,
            "exam_advice": self.FALLBACK_EXAMS,
            "evidence_note": (
                "未获授权指南背书" if guideline.is_stub or not guideline.ok
                else f"依据 {guideline.summary}"
            ),
            "_produced_by": "rule",
            "_evidence_ids": [evidence_id],
        }
        state.trace(self.name, "fallback_assessment", output_summary="规则路径给出骨质疏松风险评估",
                    evidence_ids=[evidence_id])
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
            "risk_judgement": f"存在「{signal_text(hypotheses)}」相关的风险信号，需要优先排除可致残或致命情况。",
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
    """Western differential diagnosis.

    Runs as a model-driven tool loop when its skill is marked autonomous; the
    hardcoded list below is the fallback, not the product.
    """

    name = "BiomedicalAgent"
    skill_id = "yaobi.biomedical_differential"
    output_schema = "BiomedicalAssessment"
    output_key = "biomedical"

    FALLBACK_DIFFERENTIALS = [
        "非特异性腰痛/腰肌劳损",
        "腰椎间盘突出伴神经根病",
        "腰椎管狭窄",
        "腰椎滑脱/峡部裂",
        "骶髂关节源性疼痛",
        "髋关节病变牵涉痛",
        "脊柱感染/肿瘤/骨折需按红旗排除",
    ]

    def run(self, state, tools, broker):
        result = self.autonomous(
            state, tools, broker,
            objective="对本例做西医鉴别诊断，并给出下一步查体与检查建议。先检索指南取证，再作答。",
            context={
                "chief_complaint": state.complaint,
                "red_flag_screening": state.outputs.get("intake", {}).get("screening", {}),
                "known_facts": {k: v for k, v in state.facts.items() if k != "physician_review"},
                "missing_information": state.missing_information,
            },
        )
        if result is not None:
            evidence_ids = self.bind_autonomous_output(state, result, "differential")
            for item in state.outputs[self.output_key].get("differentials", [])[:10]:
                state.add_claim("differential", str(item), evidence_ids, confidence=0.5, origin="llm")
            return state

        guideline = tools.call(broker, "clinical_guideline_search", topic="low back pain differential")
        evidence_id = record_tool(state, guideline)
        if not require_ok(state, guideline, "biomedical_guideline"):
            return state
        state.outputs[self.output_key] = {
            "differentials": self.FALLBACK_DIFFERENTIALS,
            "exam_advice": [
                "神经定位体检（肌力MRC分级、感觉平面、腱反射、直腿抬高/股神经牵拉）",
                "红旗或持续神经根症状时线下影像（X线/MRI按适应证）",
                "记录VAS疼痛评分与ODI功能评分作为随访基线",
            ],
            "evidence_note": "规则回退路径：鉴别列表为固定清单，未经模型针对本例推理",
            "_produced_by": "rule_fallback",
        }
        for item in self.FALLBACK_DIFFERENTIALS:
            state.add_claim("differential", item, [evidence_id], confidence=0.4)
        state.trace(self.name, "differential", evidence_ids=[evidence_id])
        return state


class TCMPatternAgent(BaseAgent):
    name = "TCMPatternAgent"
    skill_id = "yaobi.tcm_pattern"
    output_schema = "PatternAssessment"
    output_key = "tcm_pattern"

    def run(self, state, tools, broker):
        result = self.autonomous(
            state, tools, broker,
            objective="基于主诉与四诊信息辨证，给出主要证型、候选证型、支持证据与仍需补充的反证。",
            context={
                "chief_complaint": state.complaint,
                "four_diagnoses": state.facts.get("中医四诊") or state.facts.get("four_diagnoses"),
                "known_facts": {k: v for k, v in state.facts.items() if k != "physician_review"},
            },
        )
        if result is not None:
            evidence_ids = self.bind_autonomous_output(state, result, "pattern")
            primary = state.outputs[self.output_key].get("primary_pattern", "")
            state.add_claim("pattern", f"主要证型倾向: {primary}", evidence_ids, confidence=0.5, origin="llm")
            return state

        knowledge = tools.call(broker, "tcm_pattern_knowledge_search", text=state.complaint)
        evidence_id = record_tool(state, knowledge)
        if not require_ok(state, knowledge, "tcm_pattern"):
            return state
        patterns = knowledge.data["patterns"]
        state.outputs[self.output_key] = {
            "primary_pattern": patterns[0],
            "candidate_patterns": patterns,
            "counter_evidence_needed": ["寒热表现", "舌脉", "疼痛固定或游走", "乏力与夜痛"],
            "reasoning": "规则回退路径：按关键词匹配，未经四诊合参推理",
            "_produced_by": "rule_fallback",
        }
        state.add_claim("pattern", f"主要证型倾向: {patterns[0]}", [evidence_id], confidence=0.4)
        state.trace(self.name, "pattern", evidence_ids=[evidence_id])
        return state


class ExpertCaseAgent(BaseAgent):
    """Reasons over the expert corpus instead of just retrieving from it.

    When the corpus-mined skill is active this agent queries the practice
    profile, similar cases and counterexamples, then synthesises what the expert
    habitually does for this presentation — which is the part that actually
    transfers expertise. Without a model it falls back to plain retrieval.
    """

    name = "ExpertCaseAgent"
    skill_id = "yaobi.expert_case_reasoning"
    output_schema = "ExpertCaseEvidence"
    output_key = "expert_cases"

    def run(self, state, tools, broker):
        pattern = state.outputs.get("tcm_pattern", {}).get("primary_pattern")
        result = self.autonomous(
            state, tools, broker,
            objective=(
                "结合专家经验画像、相似病例与反例，说明这位专家对本类病例的惯常处理倾向，"
                "并明确指出经验的例数与局限。不要给出任何克数。"
            ),
            context={
                "chief_complaint": state.complaint,
                "tcm_pattern": pattern,
                "biomedical_differentials": state.outputs.get("biomedical", {}).get("differentials", []),
                "known_facts": {k: v for k, v in state.facts.items() if k != "physician_review"},
            },
        )
        if result is not None:
            evidence_ids = self.bind_autonomous_output(state, result, "expert_case")
            output = state.outputs[self.output_key]
            output.setdefault("limitation", "已脱敏的单一专家经验库，不代表因果疗效证据")
            practice = output.get("expert_practice")
            if practice:
                state.add_claim("expert_practice", str(practice)[:200], evidence_ids, confidence=0.5, origin="llm")
            return state

        similar = tools.call(broker, "similar_case_search", query=state.complaint)
        counter = tools.call(broker, "counterexample_case_search", query=state.complaint)
        profile = tools.call(broker, "expert_practice_profile", pattern=pattern)
        evidence_ids = [record_tool(state, similar), record_tool(state, counter), record_tool(state, profile)]
        if not (similar.ok and counter.ok):
            state.fail_closed("专家病例检索失败")
            return state
        state.outputs[self.output_key] = {
            "similar": similar.data.get("cases", []),
            "counterexamples": counter.data.get("cases", []),
            "expert_practice": profile.data if profile.ok else {},
            "limitation": "已脱敏的单一专家经验库，不代表因果疗效证据",
            "_produced_by": "rule_fallback",
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
    """Proposes a candidate formula. Never proposes a dose.

    An autonomously chosen herb list is still gated by 十八反/十九畏 here and by
    the dose pipeline downstream: an invented herb has no expert-case dose
    support and no authorised range, so it can never reach a dosed draft.
    """

    name = "FormulaAgent"
    skill_id = "yaobi.formula_design"
    output_schema = "FormulaCandidate"
    output_key = "formula"

    def run(self, state, tools, broker):
        pattern = state.outputs.get("tcm_pattern", {}).get("primary_pattern", "气血痹阻证")
        result = self.autonomous(
            state, tools, broker,
            objective=(
                f"针对证型「{pattern}」给出候选治法与方剂组成（只列药名，绝对不要给克数），"
                "并说明配伍思路。先检索候选方组成再作答。"
            ),
            context={
                "tcm_pattern": state.outputs.get("tcm_pattern", {}),
                "expert_cases": _trim_expert_cases(state.outputs.get("expert_cases", {})),
                "chief_complaint": state.complaint,
            },
        )
        herbs: list[str] = []
        evidence_ids: list[str] = []
        if result is not None:
            herbs = herbs_in(result.output.get("herbs", []))
            evidence_ids = result.citations or result.evidence_ids
        else:
            search = tools.call(broker, "formula_composition_search", pattern=pattern)
            evidence_ids = [record_tool(state, search)]
            if not require_ok(state, search, "formula_search"):
                return state
            herbs = herbs_in(search.data["herbs"])

        violations = incompat.check_combination(herbs)
        if violations:
            state.safety_issues += [f"候选方配伍禁忌: {v['detail']}" for v in violations]
            state.release_status = "treatment_advice_only"
            state.trace(self.name, "formula_blocked", evidence_ids=evidence_ids, output_summary="配伍禁忌")
            return state
        if not herbs:
            state.safety_issues.append("候选方为空，无法进入剂量环节")
            state.release_status = "insufficient_evidence"
            return state

        if result is not None:
            self.bind_autonomous_output(state, result, "formula")
            output = state.outputs[self.output_key]
            output["herbs"] = herbs
            output["combination_check"] = "十八反/十九畏通过"
            output["unseen_in_expert_corpus"] = _unseen_herbs(tools, herbs)
        else:
            state.outputs[self.output_key] = {
                "formula_name": "独活寄生汤加减候选",
                "treatment_principle": ["补益肝肾", "活血通络", "祛风除湿"],
                "herbs": herbs,
                "combination_check": "十八反/十九畏通过",
                "_produced_by": "rule_fallback",
            }
            state.trace(self.name, "formula_candidates", evidence_ids=evidence_ids)
        return state


def _trim_expert_cases(expert_cases: dict[str, Any]) -> dict[str, Any]:
    """Keep the synthesis, drop the bulky raw case list, before prompting."""
    return {
        "expert_practice": expert_cases.get("expert_practice"),
        "similar_count": len(expert_cases.get("similar", [])),
        "counterexample_count": len(expert_cases.get("counterexamples", [])),
        "limitation": expert_cases.get("limitation", ""),
    }


def _unseen_herbs(tools: ToolRegistry, herbs: list[str]) -> list[str]:
    """Herbs the expert corpus has never used — a flag, not a block."""
    try:
        known = {u["herb"] for p in tools.expert_profile().patterns.values()
                 for u in [h.to_dict() for h in p.core_herbs + p.adjunct_herbs]}
    except Exception:  # noqa: BLE001 - a profiling failure must not block the run
        return []
    return sorted(h for h in herbs if known and h not in known)


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
