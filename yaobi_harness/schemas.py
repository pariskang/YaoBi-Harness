"""Structural contracts for agent outputs.

Skill manifests name an ``output_schema``; this module is what makes that name
mean something. Validation is deliberately shallow — required keys and their
broad types — because the goal is to catch an agent (or an LLM) silently
emitting a differently-shaped payload, not to re-implement a type system.
"""

from __future__ import annotations

from typing import Any

#: schema name -> {field: (required, accepted python types)}
SCHEMAS: dict[str, dict[str, tuple[bool, tuple[type, ...]]]] = {
    "PatientEventGraph": {
        "events": (True, (list,)),
        "limitations": (False, (str,)),
    },
    "IntakeSummary": {
        "risk_mode": (True, (str,)),
        "screening": (True, (dict,)),
        "missing_information": (False, (list,)),
    },
    "TaskPlan": {
        "tasks": (True, (list,)),
        "planner_mode": (True, (str,)),
    },
    "UrgentCarePlan": {
        "risk_judgement": (True, (str,)),
        "immediate_action": (True, (str,)),
        "transport_advice": (True, (str,)),
        "do_not": (True, (list,)),
        "escalate_if": (True, (list,)),
        "uncertainty": (True, (str,)),
    },
    "BiomedicalAssessment": {
        "differentials": (True, (list,)),
        "exam_advice": (True, (list,)),
        "reasoning": (False, (str,)),
        "evidence_note": (False, (str,)),
    },
    "PatternAssessment": {
        "primary_pattern": (True, (str,)),
        "candidate_patterns": (True, (list,)),
        "evidence_for": (False, (list,)),
        "counter_evidence_needed": (False, (list,)),
        "reasoning": (False, (str,)),
    },
    "ExpertCaseEvidence": {
        "similar": (True, (list,)),
        "counterexamples": (True, (list,)),
        "limitation": (True, (str,)),
        # Present when the corpus-mined expert skill is active: the synthesis of
        # what this expert habitually does, rather than raw look-alike cases.
        "expert_practice": (False, (str, dict, list)),
        "reasoning": (False, (str,)),
    },
    "MedicationSafety": {
        "findings": (True, (list,)),
        "medications_reviewed": (True, (list,)),
        "reviewed": (True, (bool,)),
    },
    "FormulaCandidate": {
        "formula_name": (True, (str,)),
        "herbs": (True, (list,)),
        "treatment_principle": (True, (list,)),
        "rationale": (False, (str,)),
        "combination_check": (False, (str,)),
    },
    "PrescriptionDraft": {
        "formula_name": (True, (str,)),
        "herbs": (True, (list,)),
        "requires_physician_approval": (True, (bool,)),
        "dose_safety": (True, (dict,)),
    },
    "PhysicianReview": {
        "approved": (True, (bool,)),
        "problems_checked": (True, (list,)),
    },
    "SafetyAudit": {
        "checks_run": (True, (list,)),
        "issues": (False, (list,)),
    },
    "InterviewProgress": {
        "coverage": (True, (dict,)),
        "questions": (True, (list,)),
        "verdict": (True, (dict,)),
        "rounds_used": (False, (int,)),
    },
    # One consult subagent's opinion. ``urgency`` is required because a member
    # that cannot state an urgency has not done the one job the panel needs.
    "ConsultOpinion": {
        "urgency": (True, (str,)),
        "key_findings": (True, (list,)),
        "concerns": (True, (list,)),
        "recommend_next": (True, (list,)),
        "questions_for_patient": (False, (list,)),
        "dissent": (False, (str,)),
        "evidence_note": (False, (str,)),
    },
    # A vision read. ``requires_formal_read`` is required and must be true for a
    # radiograph or MRI/CT — see the check in ``validate`` below.
    "ImageFindings": {
        "image_kind": (True, (str,)),
        "readable": (True, (bool,)),
        "observations": (True, (list,)),
        "requires_formal_read": (True, (bool,)),
        "not_assessable": (False, (list,)),
        "urgent_signals": (False, (list,)),
        "suggest_ask": (False, (list,)),
        "suggest_exam": (False, (list,)),
        "caveat": (False, (str,)),
    },
    # The narrative sections of the clinical note. Only the sections a model adds
    # value to are required; the structured ones (prescription, evidence,
    # citations) are assembled deterministically and are not the model's to write.
    "ClinicalNoteSections": {
        "chief_complaint": (True, (str,)),
        "present_illness": (True, (str,)),
        "western_diagnosis": (True, (str,)),
        "tcm_diagnosis": (False, (str,)),
        "risk_assessment": (False, (str,)),
        "treatment_plan": (True, (str,)),
        "advice": (False, (str,)),
        "followup": (False, (str,)),
        "uncertainty": (True, (str,)),
        "past_history": (False, (str,)),
        "four_diagnoses": (False, (str,)),
        "examination": (False, (str,)),
        "investigations": (False, (str,)),
        "medication_safety": (False, (str,)),
    },
}

#: Image kinds whose findings may never claim to stand in for a formal report.
_RADIOLOGY_KINDS = {"radiograph", "mri_ct"}


def validate(schema_name: str, payload: Any) -> tuple[bool, list[str]]:
    """Validate ``payload`` against a named schema.

    An unknown schema name is a manifest bug and is reported as a failure
    rather than passing silently.
    """
    if not schema_name:
        return True, []
    schema = SCHEMAS.get(schema_name)
    if schema is None:
        return False, [f"unknown output_schema {schema_name!r}"]
    if not isinstance(payload, dict):
        return False, [f"{schema_name}: payload is not a mapping"]
    problems: list[str] = []
    for field_name, (required, types) in schema.items():
        if field_name not in payload:
            if required:
                problems.append(f"{schema_name}: missing required field {field_name!r}")
            continue
        value = payload[field_name]
        if value is None:
            if required:
                problems.append(f"{schema_name}: field {field_name!r} is null")
            continue
        if not isinstance(value, types):
            expected = "/".join(t.__name__ for t in types)
            problems.append(f"{schema_name}: field {field_name!r} should be {expected}, got {type(value).__name__}")

    # A radiology read that sets ``requires_formal_read: false`` is claiming to
    # be a report. Shape validation would pass it, so it is checked as a value.
    if (
        schema_name == "ImageFindings"
        and str(payload.get("image_kind", "")) in _RADIOLOGY_KINDS
        and payload.get("requires_formal_read") is False
    ):
        problems.append("ImageFindings: 影像判读不得声明 requires_formal_read=false（模型判读不能替代正式阅片）")
    return not problems, problems
