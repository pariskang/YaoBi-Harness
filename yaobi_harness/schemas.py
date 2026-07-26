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
    },
    "PatternAssessment": {
        "primary_pattern": (True, (str,)),
        "candidate_patterns": (True, (list,)),
    },
    "ExpertCaseEvidence": {
        "similar": (True, (list,)),
        "counterexamples": (True, (list,)),
        "limitation": (True, (str,)),
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
}


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
    return not problems, problems
