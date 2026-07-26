"""Role-scoped rendering of a finished run.

The internal state carries other patients' (pseudonymised) case records, raw
tool payloads and the full evidence ledger. Dumping all of that to whoever ran
the query ignores purpose limitation, so nothing reaches a caller except
through a view built for their role.
"""

from __future__ import annotations

from typing import Any

from .state import NON_RELEASABLE_LEVELS, ClinicalRunState

DISCLAIMER_PATIENT = "本内容为健康信息参考，不构成诊断或处方；如症状加重或出现急症信号请立即线下就医。"
DISCLAIMER_PHYSICIAN = "本内容为决策支持草案，所有含剂量内容必须由医师逐味审核签名后方可作为处方。"
DISCLAIMER_RESEARCHER = "本视图仅返回聚合统计，不返回个体病例自由文本。"


def render(state: ClinicalRunState, role: str | None = None, *, debug: bool = False) -> dict[str, Any]:
    """Build the caller-facing view for ``role``."""
    if debug:
        return state.to_dict()
    role = role or state.role
    if role == "patient":
        return _patient_view(state)
    if role == "researcher":
        return _researcher_view(state)
    return _physician_view(state)


def _header(state: ClinicalRunState) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "role": state.role,
        "risk_mode": state.risk_mode,
        "release_status": state.release_status,
        "planner_mode": state.planner_mode,
    }


def _patient_view(state: ClinicalRunState) -> dict[str, Any]:
    view: dict[str, Any] = {
        **_header(state),
        "disclaimer": DISCLAIMER_PATIENT,
        "questions_for_you": state.open_questions,
        "safety_notices": state.safety_issues + state.warnings,
    }
    if state.risk_mode == "urgent" and "urgent_action_plan" in state.outputs:
        plan = state.outputs["urgent_action_plan"]
        view["urgent"] = {
            k: plan[k]
            for k in ("risk_judgement", "why_urgent", "immediate_action", "transport_advice",
                      "during_transport", "do_not", "tell_clinicians", "escalate_if", "uncertainty")
            if k in plan
        }
        return view

    biomedical = state.outputs.get("biomedical", {})
    view["what_this_might_be"] = biomedical.get("differentials", [])
    view["what_to_do_next"] = biomedical.get("exam_advice", [])
    view["medication_warnings"] = _patient_medication_warnings(state)
    view["see_a_doctor_if"] = [
        "出现大小便困难或失禁、会阴/肛周麻木",
        "腿越来越无力、走路不稳、抬不起脚",
        "发热寒战、夜间痛醒、体重明显下降",
        "外伤后无法负重或剧痛进行性加重",
    ]
    # Expert case records and the raw evidence ledger are deliberately withheld.
    return view


def _patient_medication_warnings(state: ClinicalRunState) -> list[dict[str, str]]:
    """Plain-language version of the interaction findings, without the mechanism jargon."""
    safety = state.outputs.get("medication_safety", {})
    return [
        {
            "combination": finding.get("title", ""),
            "severity": finding.get("severity", ""),
            "what_to_do": finding.get("management", ""),
        }
        for finding in safety.get("findings", [])
        if finding.get("severity") in ("contraindicated", "major")
    ]


def citation_bundle(state: ClinicalRunState) -> list[dict[str, str]]:
    """Every external source cited by this run, with version and retrieval date.

    A clinical answer has to be able to say which guideline, which label version
    and which pharmacopoeia edition it rests on; this collects them from the
    evidence payloads so any view can show them.
    """
    seen: dict[str, dict[str, str]] = {}
    for evidence in state.evidence.values():
        for citation in evidence.payload.get("citations", []) or []:
            if not isinstance(citation, dict):
                continue
            key = f"{citation.get('source_id')}|{citation.get('version')}|{citation.get('url')}"
            seen.setdefault(key, citation)
        for finding in evidence.payload.get("rule_findings", []) or []:
            if isinstance(finding, dict) and finding.get("source_id"):
                seen.setdefault(f"rule|{finding['source_id']}", {
                    "source_id": finding["source_id"],
                    "source": "Yaobi 骨科相互作用规则包",
                    "license": "MIT (this repository)",
                    "version": finding.get("rule_id", ""),
                })
    return list(seen.values())


def _physician_view(state: ClinicalRunState) -> dict[str, Any]:
    outputs = state.outputs
    return {
        **_header(state),
        "disclaimer": DISCLAIMER_PHYSICIAN,
        "intake": outputs.get("intake"),
        "plan": outputs.get("plan"),
        "timeline": outputs.get("timeline"),
        "biomedical": outputs.get("biomedical"),
        "tcm_pattern": outputs.get("tcm_pattern"),
        "expert_cases": outputs.get("expert_cases"),
        "medication_safety": outputs.get("medication_safety"),
        "citations": citation_bundle(state),
        "formula": outputs.get("formula"),
        "dose_safety": outputs.get("dose_safety"),
        "prescription_draft": outputs.get("prescription_draft"),
        "physician_review": outputs.get("physician_review"),
        "urgent_action_plan": outputs.get("urgent_action_plan"),
        "safety_audit": outputs.get("safety_audit"),
        "safety_issues": state.safety_issues,
        "warnings": state.warnings,
        "open_questions": state.open_questions,
        "missing_information": state.missing_information,
        "claims": [
            {"id": c.claim_id, "kind": c.kind, "text": c.text, "evidence_ids": c.evidence_ids,
             "confidence": c.confidence, "origin": c.origin}
            for c in state.claims
        ],
        "evidence_ledger": [
            {"id": e.evidence_id, "level": e.level, "source": e.source, "summary": e.summary,
             "source_version": e.source_version, "releasable": e.level not in NON_RELEASABLE_LEVELS}
            for e in state.evidence.values()
        ],
        "run_meta": outputs.get("run_meta"),
    }


def _researcher_view(state: ClinicalRunState) -> dict[str, Any]:
    cases = state.outputs.get("expert_cases", {})
    return {
        **_header(state),
        "disclaimer": DISCLAIMER_RESEARCHER,
        "counts": {
            "similar_cases": len(cases.get("similar", [])),
            "counterexamples": len(cases.get("counterexamples", [])),
            "evidence_items": len(state.evidence),
            "claims": len(state.claims),
        },
        "dose_safety": state.outputs.get("dose_safety"),
        "citations": citation_bundle(state),
        "evidence_levels": _level_histogram(state),
        "safety_issues": state.safety_issues,
        "run_meta": state.outputs.get("run_meta"),
    }


def console_payload(state: ClinicalRunState, role: str | None = None) -> dict[str, Any]:
    """Build the operator-console view: what would be delivered, plus the audit.

    This is deliberately *not* what a patient receives. ``delivered`` is the
    role-scoped answer that would actually go out; ``audit`` is the reasoning
    record — plan, tool trace, evidence ledger, safety findings — shown only to
    the operator running the console. Keeping the two separate on the wire is
    what stops the console from quietly becoming a privacy leak.
    """
    role = role or state.role
    return {
        "delivered": render(state, role),
        "audit": {
            "tasks": [
                {
                    "task_id": t.task_id, "agent": t.agent, "objective": t.objective,
                    "status": t.status, "depends_on": t.depends_on,
                    "required_tools": t.required_tools, "origin": t.origin,
                    "repair_reason": t.repair_reason,
                }
                for t in state.tasks
            ],
            "traces": [
                {
                    "agent": t.agent, "action": t.action, "output_summary": t.output_summary,
                    "evidence_ids": t.evidence_ids, "timestamp": t.timestamp,
                }
                for t in state.traces
            ],
            "evidence": [
                {
                    "id": e.evidence_id, "level": e.level, "source": e.source, "summary": e.summary,
                    "source_version": e.source_version,
                    "releasable": e.level not in NON_RELEASABLE_LEVELS,
                    "tool_ok": bool(e.payload.get("tool_ok", True)),
                }
                for e in state.evidence.values()
            ],
            "claims": [
                {"id": c.claim_id, "kind": c.kind, "text": c.text,
                 "evidence_ids": c.evidence_ids, "confidence": c.confidence, "origin": c.origin}
                for c in state.claims
            ],
            # Per-agent record of the model-driven tool loop: which tools it
            # chose, with what arguments, and whether it fell back.
            "autonomy": state.outputs.get("autonomy", {}),
            # History taking: axis coverage, the questions actually asked, and the
            # adequacy verdict. Operator-only — the patient sees the questions, not
            # the judge's reasoning about whether they were enough.
            "interview": state.outputs.get("interview", {}),
            "consult_panel": state.outputs.get("consult_panel", {}),
            "image_findings": state.outputs.get("image_findings", {}),
            "osteoporosis_risk": state.outputs.get("osteoporosis_risk", {}),
            "safety_audit": state.outputs.get("safety_audit", {}),
            "safety_issues": state.safety_issues,
            "warnings": state.warnings,
            "citations": citation_bundle(state),
            "medication_safety": state.outputs.get("medication_safety", {}),
            "dose_safety": state.outputs.get("dose_safety", {}),
            "prescription_draft": state.outputs.get("prescription_draft"),
            "missing_information": state.missing_information,
            "open_questions": state.open_questions,
        },
        "meta": {
            **_header(state),
            **(state.outputs.get("run_meta") or {}),
            "intake": state.outputs.get("intake", {}).get("screening", {}),
        },
    }


def _level_histogram(state: ClinicalRunState) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for evidence in state.evidence.values():
        histogram[evidence.level] = histogram.get(evidence.level, 0) + 1
    return histogram
