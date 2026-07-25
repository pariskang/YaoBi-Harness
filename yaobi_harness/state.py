from __future__ import annotations

import time, uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

Role = Literal["patient", "physician", "researcher"]
RiskMode = Literal["routine", "urgent"]
ReleaseStatus = Literal[
    "urgent_action_plan", "needs_more_information", "needs_examination",
    "insufficient_evidence", "treatment_advice_only", "draft_for_physician",
    "approved_by_physician", "blocked", "failed_closed",
]

class EvidenceLevel(str, Enum):
    OBSERVED = "observed_fact"
    EXPERT_CASE = "expert_case"
    GUIDELINE = "guideline_or_standard"
    PHARMACOPEIA = "pharmacopeia_or_formal_norm"
    TOOL = "tool_result"
    MODEL = "model_reasoning"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

@dataclass
class Evidence:
    evidence_id: str
    level: str
    source: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    source_version: str | None = None

@dataclass
class AgentTrace:
    agent: str
    action: str
    input_summary: str = ""
    output_summary: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=now)

@dataclass
class Task:
    task_id: str
    agent: str
    objective: str
    required_tools: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"

@dataclass
class Budget:
    max_loops: int = 3
    max_tool_calls: int = 24
    max_questions: int = 8
    used_tool_calls: int = 0
    loop_counts: dict[str, int] = field(default_factory=dict)

    def reserve_tool(self) -> bool:
        if self.used_tool_calls >= self.max_tool_calls:
            return False
        self.used_tool_calls += 1
        return True

@dataclass
class ClinicalRunState:
    complaint: str
    role: Role = "physician"
    run_id: str = field(default_factory=lambda: f"yaobi_{uuid.uuid4().hex[:10]}")
    risk_mode: RiskMode = "routine"
    facts: dict[str, Any] = field(default_factory=dict)
    missing_information: list[str] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    traces: list[AgentTrace] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    release_status: ReleaseStatus = "needs_more_information"
    safety_issues: list[str] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    requires_physician_approval: bool = True

    def add_evidence(self, level: EvidenceLevel | str, source: str, summary: str, payload: dict[str, Any] | None = None, *, ok: bool = True, error: str | None = None, source_version: str | None = None) -> str:
        eid = f"E{len(self.evidence)+1:04d}"
        data = dict(payload or {})
        data.update({"tool_ok": ok, "tool_error": error})
        evidence_level = "failed_tool_event" if not ok else str(level.value if isinstance(level, EvidenceLevel) else level)
        self.evidence[eid] = Evidence(eid, evidence_level, source, summary, data, source_version)
        return eid

    def fail_closed(self, reason: str) -> None:
        self.release_status = "failed_closed"
        self.safety_issues.append(reason)

    def trace(self, agent: str, action: str, input_summary: str = "", output_summary: str = "", evidence_ids: list[str] | None = None) -> None:
        self.traces.append(AgentTrace(agent, action, input_summary, output_summary, evidence_ids or []))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
