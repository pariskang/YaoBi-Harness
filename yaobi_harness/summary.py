"""The structured clinical note produced once a consultation reaches a conclusion.

Everything else in this harness is built for the machine — evidence ids, release
statuses, capability checks. This module produces the artefact a *clinician*
actually needs at the end: a note in the sections a Chinese orthopaedic record
uses (主诉 / 现病史 / 既往史 / 四诊 / 辅助检查 / 诊断 / 治疗 / 医嘱 / 随访), assembled
from what the run established rather than from anything new.

Three properties make it safe to hand over:

* **Nothing is invented here.** Every section is filled from ``state.outputs``,
  ``state.facts`` and the evidence ledger. A section with no source comes back
  empty, and an empty section is printed as 「未采集」 rather than quietly omitted —
  a record that silently drops 既往史 reads as "nothing relevant", which is a
  clinically different claim from "not asked".
* **Doses appear only if the run produced them.** The note reproduces the
  prescription draft verbatim when one exists, with its per-herb signature status,
  and otherwise carries no gram values at all. It never re-derives a dose.
* **It says what it is.** A note built from a run that never reached a physician's
  signature is stamped as a draft, and the stamp is part of the structure rather
  than a footnote — so it cannot be lost by a surface that renders only the fields
  it recognises.

The model may *write* the narrative sections (see ``yaobi.clinical_summary``);
this module is what supplies it with the material and what produces the note when
no model is configured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .render import citation_bundle
from .state import NON_RELEASABLE_LEVELS, ClinicalRunState

#: Section order. This is the order a Chinese outpatient orthopaedic note is
#: written in, and the order a reviewing physician reads in, so it is fixed.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("chief_complaint", "主诉"),
    ("present_illness", "现病史"),
    ("past_history", "既往史与用药"),
    ("four_diagnoses", "中医四诊"),
    ("examination", "查体与量表"),
    ("investigations", "辅助检查与影像"),
    ("western_diagnosis", "西医诊断（鉴别）"),
    ("tcm_diagnosis", "中医诊断与证型"),
    ("risk_assessment", "风险评估与红旗"),
    ("medication_safety", "用药安全"),
    ("treatment_plan", "治疗计划"),
    ("prescription", "处方草案"),
    ("advice", "医嘱与自我照护"),
    ("followup", "随访与复诊指征"),
    ("uncertainty", "不确定性与局限"),
)

#: Statuses at which a note is worth producing. Before this the consultation has
#: not concluded, and a note over an unfinished history is a misleading document.
CONCLUDED_STATUSES = frozenset({
    "treatment_advice_only", "draft_for_physician", "approved_by_physician",
    "urgent_action_plan", "needs_examination",
})

#: Printed for a section nothing in the run could fill. Not omitted: a missing
#: 既往史 and an empty 既往史 are different clinical claims.
NOT_COLLECTED = "未采集"

_DOSE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:克|g\b|G\b|毫克|mg\b)")


@dataclass
class ClinicalNote:
    """A structured note. ``sections`` is ordered per :data:`SECTIONS`."""

    run_id: str
    role: str
    release_status: str
    risk_mode: str
    #: ``draft`` unless a physician signed every herb. Part of the structure so a
    #: surface cannot render the note and lose the caveat.
    status_label: str
    sections: dict[str, str] = field(default_factory=dict)
    #: The prescription exactly as the run produced it, or ``None``.
    prescription: dict[str, Any] | None = None
    citations: list[dict[str, str]] = field(default_factory=list)
    #: Evidence ids backing the note, releasable ones only.
    evidence_ids: list[str] = field(default_factory=list)
    #: ``rule`` when assembled deterministically, ``llm`` when the model wrote the
    #: narrative sections over this material.
    composed_by: str = "rule"
    disclaimer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "role": self.role,
            "release_status": self.release_status, "risk_mode": self.risk_mode,
            "status_label": self.status_label,
            "sections": [
                {"key": key, "title": title, "text": self.sections.get(key) or NOT_COLLECTED}
                for key, title in SECTIONS
            ],
            "prescription": self.prescription,
            "citations": self.citations,
            "evidence_ids": self.evidence_ids,
            "composed_by": self.composed_by,
            "disclaimer": self.disclaimer,
        }

    def to_text(self) -> str:
        """The note as plain text, ready to paste into a record system."""
        lines = [
            f"【骨科门诊病历摘要 · {self.status_label}】",
            f"运行编号：{self.run_id}    交付对象：{self.role}    "
            f"放行状态：{self.release_status}    风险模式：{self.risk_mode}",
            "",
        ]
        for key, title in SECTIONS:
            lines.append(f"■ {title}")
            lines.append((self.sections.get(key) or NOT_COLLECTED).strip())
            lines.append("")
        if self.citations:
            lines.append("■ 引用来源")
            for citation in self.citations:
                parts = [citation.get("source", ""), citation.get("version", ""),
                         citation.get("license", ""), citation.get("retrieved_at", "")]
                lines.append("· " + " / ".join(p for p in parts if p))
            lines.append("")
        if self.evidence_ids:
            lines.append("■ 证据台账")
            lines.append("、".join(self.evidence_ids))
            lines.append("")
        if self.disclaimer:
            lines.append(self.disclaimer)
        return "\n".join(lines).rstrip() + "\n"


def is_concluded(state: ClinicalRunState) -> bool:
    """Whether this run has got far enough for a note to mean anything."""
    return state.release_status in CONCLUDED_STATUSES


def build_note(state: ClinicalRunState, *, narrative: list[str] | None = None) -> ClinicalNote:
    """Assemble the note deterministically from what the run established.

    ``narrative`` is the dialogue's accumulated messages when there is one; a
    single-shot run has only its complaint. Nothing here consults a model, which
    is what makes this the fallback *and* the material.
    """
    outputs = state.outputs
    facts = dict(state.facts)
    note = ClinicalNote(
        run_id=state.run_id,
        role=state.role,
        release_status=state.release_status,
        risk_mode=state.risk_mode,
        status_label=_status_label(state),
        disclaimer=_disclaimer(state),
    )
    note.sections = {
        "chief_complaint": _chief_complaint(state, narrative),
        "present_illness": _present_illness(state, narrative, facts),
        "past_history": _past_history(facts),
        "four_diagnoses": _text_or_blank(facts.get("four_diagnoses")),
        "examination": _examination(facts),
        "investigations": _investigations(outputs),
        "western_diagnosis": _western_diagnosis(outputs),
        "tcm_diagnosis": _tcm_diagnosis(outputs),
        "risk_assessment": _risk_assessment(state, outputs),
        "medication_safety": _medication_safety(outputs),
        "treatment_plan": _treatment_plan(outputs),
        "prescription": _prescription_text(outputs),
        "advice": _advice(outputs),
        "followup": _followup(outputs),
        "uncertainty": _uncertainty(state, outputs),
    }
    note.prescription = outputs.get("prescription_draft")
    note.citations = citation_bundle(state)
    note.evidence_ids = [
        eid for eid, evidence in state.evidence.items()
        if evidence.level not in NON_RELEASABLE_LEVELS
    ]
    return note


# ------------------------------------------------------------------- sections
def _chief_complaint(state: ClinicalRunState, narrative: list[str] | None) -> str:
    """The first thing the patient said, not the whole accumulated narrative.

    A 主诉 is one line. Pasting six turns of dialogue into it is what makes a
    generated note unusable to the person who has to read it.
    """
    if narrative:
        return narrative[0].strip()
    return (state.complaint or "").split("。")[0].strip()


def _present_illness(state: ClinicalRunState, narrative: list[str] | None, facts: dict) -> str:
    parts = []
    if facts.get("onset"):
        parts.append(f"起病：{facts['onset']}")
    if facts.get("pain_location"):
        parts.append(f"部位：{facts['pain_location']}")
    if facts.get("radiation"):
        parts.append(f"放射：{facts['radiation']}")
    for key, label in (("morning_stiffness", "晨僵"), ("night_pain", "夜间痛"),
                       ("walking_tolerance", "行走耐受"), ("neuro_symptoms", "神经症状"),
                       ("bowel_bladder", "大小便与鞍区"), ("fever_trauma_tumor", "发热/外伤/肿瘤线索")):
        if facts.get(key):
            parts.append(f"{label}：{facts[key]}")
    body = "；".join(parts)
    # The narrative goes in as the patient's own words, after the structured
    # summary. A record that keeps only the extracted fields loses the phrasing a
    # reviewing physician often needs.
    if narrative and len(narrative) > 1:
        told = "；".join(m.strip() for m in narrative[1:] if m.strip())
        if told:
            body = f"{body}\n患者自述补充：{told}" if body else f"患者自述：{told}"
    return body


def _past_history(facts: dict) -> str:
    parts = []
    for key, label in (("past_history", "既往"), ("surgical_history", "手术史"),
                       ("fragility_risk", "骨脆性风险"), ("occupation", "职业"),
                       ("sleep", "睡眠"), ("yellow_flags", "心理社会因素")):
        if facts.get(key):
            parts.append(f"{label}：{facts[key]}")
    medications = facts.get("medications") or []
    if medications:
        parts.append("当前用药：" + "、".join(str(m) for m in medications))
    elif facts.get("medications_confirmed"):
        parts.append("当前用药：已确认无")
    allergies = facts.get("allergies") or []
    if allergies:
        parts.append("过敏：" + "、".join(str(a) for a in allergies))
    elif facts.get("allergies_confirmed"):
        parts.append("过敏：已确认无")
    population = facts.get("special_population") or {}
    demographic = [f"{k}={v}" for k, v in population.items() if v is not None]
    if demographic:
        parts.append("特殊人群：" + "、".join(demographic))
    extra = facts.get("_extra") or {}
    if extra:
        # Findings the governed allowlist does not cover. They belong in a record
        # even though nothing downstream reads them — that is what a record is for.
        parts.append("补充信息：" + "、".join(f"{k}={v}" for k, v in extra.items()))
    return "；".join(parts)


def _examination(facts: dict) -> str:
    parts = []
    if facts.get("vas") is not None:
        parts.append(f"VAS 疼痛评分：{facts['vas']}/10")
    if facts.get("odi") is not None:
        parts.append(f"ODI 功能障碍指数：{facts['odi']}")
    if facts.get("myelopathy_signs"):
        parts.append(f"髓性症状：{facts['myelopathy_signs']}")
    if facts.get("limb_vascular"):
        parts.append(f"肢体血供与肿胀：{facts['limb_vascular']}")
    if not parts:
        return "线上问诊未进行体格检查"
    return "；".join(parts) + "\n（线上问诊，以上为患者自述，非查体所见）"


def _investigations(outputs: dict) -> str:
    parts = []
    findings = outputs.get("image_findings") or {}
    for kind, read in (findings.items() if isinstance(findings, dict) else []):
        if not isinstance(read, dict):
            continue
        observations = "、".join(str(o) for o in (read.get("observations") or [])[:6])
        line = f"{kind} 模型判读所见：{observations or '无明确所见'}"
        if read.get("requires_formal_read"):
            line += "（**非影像报告，需正式阅片**）"
        parts.append(line)
    advice = (outputs.get("biomedical") or {}).get("exam_advice") or []
    if advice:
        parts.append("建议检查：" + "；".join(str(a) for a in advice[:6]))
    return "\n".join(parts)


def _western_diagnosis(outputs: dict) -> str:
    biomedical = outputs.get("biomedical") or {}
    differentials = biomedical.get("differentials") or []
    if not differentials:
        return ""
    lines = ["考虑方向（按可能性排序，线上不能确诊）："]
    lines += [f"{index}. {item}" for index, item in enumerate(differentials[:8], 1)]
    if biomedical.get("reasoning"):
        lines.append(f"依据：{biomedical['reasoning']}")
    return "\n".join(lines)


def _tcm_diagnosis(outputs: dict) -> str:
    pattern = outputs.get("tcm_pattern") or {}
    if not pattern:
        return ""
    parts = []
    if pattern.get("primary_pattern"):
        parts.append(f"证型：{pattern['primary_pattern']}")
    primary = pattern.get("primary_pattern")
    # A differential list that repeats the primary pattern reads as indecision the
    # model did not express.
    candidates = [c for c in (pattern.get("candidate_patterns") or []) if c and c != primary]
    if candidates:
        parts.append("待鉴别证型：" + _phrase(candidates, limit=4))
    if pattern.get("reasoning"):
        parts.append(f"辨证依据：{pattern['reasoning']}")
    expert = _expert_reference(outputs.get("expert_cases") or {})
    if expert:
        parts.append(expert)
    return "；".join(parts)


def _expert_reference(expert: dict) -> str:
    """A one-line summary of the corpus reference, not the whole mined profile.

    Pasting the nested profile dict into a record is worse than omitting it: it
    buries the two numbers a reviewer wants — which pattern, how many cases —
    under a page of serialised Python.
    """
    practice = expert.get("expert_practice")
    if isinstance(practice, str) and practice.strip():
        return f"专家经验参照：{practice.strip()}"
    if not isinstance(practice, dict):
        return ""
    matched = practice.get("matched_pattern") or practice.get("requested_pattern") or ""
    patterns = practice.get("patterns") or []
    entry = next((p for p in patterns if isinstance(p, dict) and p.get("pattern") == matched), None)
    if entry is None and patterns and isinstance(patterns[0], dict):
        entry = patterns[0]
    if entry is None:
        return ""
    core = [h.get("herb") for h in (entry.get("core_herbs") or []) if isinstance(h, dict)]
    bits = [f"专家经验参照：{entry.get('pattern', matched)}"]
    if entry.get("n_cases"):
        bits.append(f"本院语料 {entry['n_cases']} 例")
    if core:
        bits.append("核心药：" + _phrase(core, limit=8))
    if entry.get("counterexample_cases"):
        bits.append(f"反例 {entry['counterexample_cases']} 例")
    if entry.get("reliable") is False:
        bits.append("样本量不足，仅作参考")
    return "，".join(bits)


def _risk_assessment(state: ClinicalRunState, outputs: dict) -> str:
    screening = (outputs.get("intake") or {}).get("screening") or {}
    parts = []
    level = screening.get("triage_level") or state.risk_mode
    by = {"llm": "模型判定", "rule": "规则筛查判定"}.get(screening.get("triage_by"), "")
    parts.append(f"分诊等级：{level}（{by}）" if by else f"分诊等级：{level}")
    if screening.get("triage_reason"):
        parts.append(f"理由：{screening['triage_reason']}")
    hits = sorted({h.get("signal", "") for h in (screening.get("hits") or []) if h.get("signal")})
    if hits:
        parts.append("规则关键词命中：" + "、".join(hits))
    disputed = screening.get("disputed_rule_hits") or []
    if disputed:
        # A disagreement belongs in the record. Whoever reads this later needs to
        # know the screen flagged something and why it was set aside.
        parts.append("模型认为本例不适用的规则命中：" + "；".join(
            f"{d.get('signal')}（{d.get('why', '')}）" for d in disputed))
    urgent = outputs.get("urgent_action_plan") or {}
    if urgent.get("risk_judgement"):
        parts.append(f"急症判断：{urgent['risk_judgement']}")
    return "\n".join(parts)


def _medication_safety(outputs: dict) -> str:
    safety = outputs.get("medication_safety") or {}
    findings = safety.get("findings") or []
    if not findings:
        return safety.get("overall") or ""
    lines = []
    for finding in findings[:8]:
        if not isinstance(finding, dict):
            continue
        lines.append(
            f"[{finding.get('severity', '')}] {finding.get('combination', '')}"
            f" — 机制：{finding.get('mechanism', '')}；处理：{finding.get('what_to_do', '')}")
    return "\n".join(lines)


def _treatment_plan(outputs: dict) -> str:
    parts = []
    urgent = outputs.get("urgent_action_plan") or {}
    if urgent:
        for key, label in (("immediate_action", "立即处置"), ("transport_advice", "转运"),
                           ("do_not", "禁止事项"), ("tell_clinicians", "告知接诊医师")):
            value = _phrase(urgent.get(key), separator="；")
            if value:
                parts.append(f"{label}：{value}")
        return "\n".join(parts)
    formula = outputs.get("formula") or {}
    if formula.get("treatment_principle"):
        parts.append(f"治法：{_phrase(formula['treatment_principle'])}")
    if formula.get("formula_name"):
        parts.append(f"方剂：{formula['formula_name']}")
    herbs = formula.get("herbs") or []
    if herbs:
        # Names only. The doses live in the prescription section, where the
        # signature status sits next to them.
        parts.append("组成（不含剂量）：" + _phrase(herbs, limit=30))
    advice = (outputs.get("biomedical") or {}).get("exam_advice") or []
    if not parts and advice:
        parts.append("以线下评估与检查为下一步，暂不给出用药方案")
    return "\n".join(parts)


def _prescription_text(outputs: dict) -> str:
    draft = outputs.get("prescription_draft")
    if not isinstance(draft, dict):
        return ""
    review = outputs.get("physician_review") or {}
    signed = bool(review.get("approved"))
    lines = [
        f"方名：{draft.get('formula_name', '')}",
        f"审核状态：{'医师已逐味审核签名' if signed else '未签名，仅为医师草案，不得作为处方使用'}",
    ]
    for herb in draft.get("herbs") or []:
        if not isinstance(herb, dict):
            continue
        authorized = herb.get("authorized_range_g")
        range_text = f"授权范围 {authorized[0]}–{authorized[1]}g" if authorized else "无授权范围"
        flags = "、".join(herb.get("risk_flags") or []) or "无"
        lines.append(
            f"· {herb.get('herb_name', '')} {herb.get('dose_value', '')}"
            f"{herb.get('dose_unit', 'g')}（{range_text}；样本 n={herb.get('sample_n')}；"
            f"风险标记：{flags}；状态：{herb.get('physician_status', 'pending')}）")
    decoction = draft.get("decoction") or {}
    if decoction:
        lines.append(
            f"煎服：{decoction.get('frequency', '')}，每日 {decoction.get('times_per_day', '')} 次，"
            f"共 {decoction.get('duration_days', '')} 天")
    if draft.get("prescription_hash"):
        lines.append(f"草案指纹：{draft['prescription_hash']}")
    return "\n".join(lines)


def _advice(outputs: dict) -> str:
    urgent = outputs.get("urgent_action_plan") or {}
    if urgent.get("during_transport"):
        return _phrase(urgent["during_transport"], separator="；")
    parts = []
    review = outputs.get("physician_review") or {}
    if review.get("notes"):
        parts.append(str(review["notes"]))
    formula = outputs.get("formula") or {}
    if formula.get("cautions"):
        parts.append("注意：" + _phrase(formula["cautions"], separator="；", limit=4))
    return "\n".join(parts)


def _followup(outputs: dict) -> str:
    urgent = outputs.get("urgent_action_plan") or {}
    escalate = urgent.get("escalate_if") or []
    if escalate:
        return "出现以下情况立即急诊：" + _phrase(escalate, separator="；")
    expert = outputs.get("expert_cases") or {}
    parts = []
    followup = (expert.get("expert_practice") or {}) if isinstance(expert.get("expert_practice"), dict) else {}
    trajectory = followup.get("followup") or expert.get("followup")
    if isinstance(trajectory, dict) and trajectory.get("median_visits"):
        parts.append(f"语料随访轨迹参照：中位复诊 {trajectory['median_visits']} 次"
                     f"（{trajectory.get('patients_with_followup', 0)}/"
                     f"{trajectory.get('distinct_patients', 0)} 例有复诊记录）")
    parts.append("症状加重、出现大小便异常、下肢进行性无力或发热时立即线下就诊")
    return "\n".join(parts)


def _uncertainty(state: ClinicalRunState, outputs: dict) -> str:
    parts = []
    interview = outputs.get("interview") or {}
    verdict = interview.get("verdict") or {}
    if verdict.get("verdict"):
        parts.append(f"问诊充分性：{verdict['verdict']}（{verdict.get('reason', '')}）")
    blocking = verdict.get("blocking_labels") or []
    if blocking:
        parts.append("必答项仍未闭合：" + _phrase(blocking, limit=8))
    closed = verdict.get("closed_by_reviewer") or []
    if closed:
        parts.append("审核者按病史原话判定为已闭合：" + "；".join(
            f"{c.get('label', c.get('axis_id'))}（依据：{c.get('quote', '')}）" for c in closed))
    for issue in (state.safety_issues or [])[:4]:
        parts.append(f"安全审查：{issue}")
    urgent = outputs.get("urgent_action_plan") or {}
    if urgent.get("uncertainty"):
        parts.append(str(urgent["uncertainty"]))
    if not parts:
        parts.append("线上问诊无法替代查体与影像，本摘要为风险分层与方向性判断")
    return "\n".join(parts)


# -------------------------------------------------------------------- helpers
def _text_or_blank(value: Any) -> str:
    return str(value).strip() if value else ""


def _phrase(value: Any, separator: str = "、", limit: int = 12) -> str:
    """Render a value as prose. Never as a Python repr.

    ``治法：['补益肝肾', '活血通络']`` in a medical record is the kind of detail that
    makes a generated note obviously machine-produced and therefore ignored.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return separator.join(f"{k}={_phrase(v)}" for k, v in list(value.items())[:limit])
    if isinstance(value, (list, tuple, set)):
        return separator.join(_phrase(item) for item in list(value)[:limit] if item not in (None, ""))
    return str(value)


def _status_label(state: ClinicalRunState) -> str:
    if state.release_status == "approved_by_physician":
        return "医师已签名"
    if state.release_status == "urgent_action_plan":
        return "急症处置记录（未签名）"
    return "草案 · 未经医师签名"


def _disclaimer(state: ClinicalRunState) -> str:
    base = (
        "本摘要由 Yaobi 智能体依据线上问诊生成，**不是诊断书、不是处方**，"
        "不能替代查体、影像与执业医师判断。"
    )
    if state.outputs.get("prescription_draft") and state.release_status != "approved_by_physician":
        base += "含剂量内容为医师草案，未经逐味审核签名前不得作为处方使用。"
    return base


def redact_doses(text: str) -> str:
    """Remove gram values from model-written narrative.

    Applied to the sections a model authored, never to
    :func:`_prescription_text` — that one *is* the prescription, its numbers came
    from the deterministic pipeline, and its signature status is printed beside
    them.
    """
    return _DOSE_RE.sub("（剂量见处方草案）", text or "")
