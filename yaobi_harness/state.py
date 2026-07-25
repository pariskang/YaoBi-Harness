"""Run state for the Yaobi clinical harness.

The state is the single source of truth that flows through every graph node.
It is fully serialisable so a run can be checkpointed after each node and
resumed later (see :mod:`yaobi_harness.graph`).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

Role = Literal["patient", "physician", "researcher"]
RiskMode = Literal["routine", "urgent"]
ReleaseStatus = Literal[
    "urgent_action_plan",
    "needs_more_information",
    "needs_examination",
    "insufficient_evidence",
    "treatment_advice_only",
    "draft_for_physician",
    "approved_by_physician",
    "blocked",
    "failed_closed",
]


class EvidenceLevel(str, Enum):
    """Provenance grade of a piece of evidence.

    The grade is decided by the *tool* that produced the evidence, never by the
    agent that consumed it. ``STUB`` exists so placeholder knowledge sources can
    never be laundered into guideline-grade evidence.
    """

    OBSERVED = "observed_fact"
    EXPERT_CASE = "expert_case"
    GUIDELINE = "guideline_or_standard"
    PHARMACOPEIA = "pharmacopeia_or_formal_norm"
    TOOL = "tool_result"
    MODEL = "model_reasoning"
    STUB = "stub_not_for_clinical_use"
    FAILED = "failed_tool_event"


#: Evidence grades that may never on their own justify a released clinical claim.
NON_RELEASABLE_LEVELS = {EvidenceLevel.STUB.value, EvidenceLevel.FAILED.value, EvidenceLevel.MODEL.value}


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
class Claim:
    """A single released assertion bound to the evidence that supports it.

    ``CitationGuard`` (see :mod:`yaobi_harness.agent.agents`) rejects any claim
    whose evidence list is empty or backed only by non-releasable evidence.
    """

    claim_id: str
    kind: str
    text: str
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    origin: str = "rule"


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
    origin: str = "rule"
    #: Set when a critic finding asks this task to be re-run in a later loop.
    repair_reason: str = ""


@dataclass
class Budget:
    """Hard ceilings for a single run.

    Every counter here is enforced; nothing in this dataclass is decorative.
    Tool calls are charged only when a call is actually executed, so a denied
    or policy-blocked call can never drain the budget.
    """

    max_loops: int = 3
    max_tool_calls: int = 24
    max_questions: int = 8
    max_llm_calls: int = 12
    max_llm_tokens: int = 120_000
    used_tool_calls: int = 0
    used_llm_calls: int = 0
    used_llm_tokens: int = 0
    used_questions: int = 0
    loop_counts: dict[str, int] = field(default_factory=dict)

    def can_afford_tool(self) -> bool:
        return self.used_tool_calls < self.max_tool_calls

    def charge_tool(self) -> None:
        """Charge one executed tool call. Call only after the call ran."""
        self.used_tool_calls += 1

    def reserve_llm(self) -> bool:
        if self.used_llm_calls >= self.max_llm_calls or self.used_llm_tokens >= self.max_llm_tokens:
            return False
        self.used_llm_calls += 1
        return True

    def charge_llm_tokens(self, tokens: int) -> None:
        self.used_llm_tokens += max(0, int(tokens))

    def can_loop(self, node: str) -> bool:
        return self.loop_counts.get(node, 0) < self.max_loops

    def count_loop(self, node: str) -> int:
        self.loop_counts[node] = self.loop_counts.get(node, 0) + 1
        return self.loop_counts[node]

    def reserve_questions(self, n: int) -> int:
        allowed = max(0, min(n, self.max_questions - self.used_questions))
        self.used_questions += allowed
        return allowed


@dataclass
class ClinicalRunState:
    complaint: str
    role: Role = "physician"
    run_id: str = field(default_factory=lambda: f"yaobi_{uuid.uuid4().hex[:10]}")
    risk_mode: RiskMode = "routine"
    facts: dict[str, Any] = field(default_factory=dict)
    missing_information: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    claims: list[Claim] = field(default_factory=list)
    traces: list[AgentTrace] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    release_status: ReleaseStatus = "needs_more_information"
    safety_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    requires_physician_approval: bool = True
    planner_mode: str = "rule"
    loop_index: int = 0

    # ---------------------------------------------------------------- evidence
    def add_evidence(
        self,
        level: EvidenceLevel | str,
        source: str,
        summary: str,
        payload: dict[str, Any] | None = None,
        *,
        ok: bool = True,
        error: str | None = None,
        source_version: str | None = None,
    ) -> str:
        eid = f"E{len(self.evidence) + 1:04d}"
        data = dict(payload or {})
        data.update({"tool_ok": ok, "tool_error": error})
        if not ok:
            evidence_level = EvidenceLevel.FAILED.value
        else:
            evidence_level = str(level.value if isinstance(level, EvidenceLevel) else level)
        self.evidence[eid] = Evidence(eid, evidence_level, source, summary, data, source_version)
        return eid

    def releasable_evidence_ids(self) -> set[str]:
        return {eid for eid, e in self.evidence.items() if e.level not in NON_RELEASABLE_LEVELS}

    # ------------------------------------------------------------------ claims
    def add_claim(
        self,
        kind: str,
        text: str,
        evidence_ids: list[str] | None = None,
        *,
        confidence: float = 0.5,
        origin: str = "rule",
    ) -> str:
        claim_id = f"C{len(self.claims) + 1:04d}"
        self.claims.append(Claim(claim_id, kind, text, list(evidence_ids or []), confidence, origin))
        return claim_id

    # ------------------------------------------------------------------ status
    def fail_closed(self, reason: str) -> None:
        self.release_status = "failed_closed"
        self.safety_issues.append(reason)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def trace(
        self,
        agent: str,
        action: str,
        input_summary: str = "",
        output_summary: str = "",
        evidence_ids: list[str] | None = None,
    ) -> None:
        self.traces.append(AgentTrace(agent, action, input_summary, output_summary, evidence_ids or []))

    def task_by_id(self, task_id: str) -> Task | None:
        return next((t for t in self.tasks if t.task_id == task_id), None)

    # ------------------------------------------------------------ (de)serialise
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClinicalRunState":
        """Rebuild a state from :meth:`to_dict`, so runs can be resumed."""
        payload = dict(data)
        budget = Budget(**payload.pop("budget", {}) or {})
        tasks = [Task(**t) for t in payload.pop("tasks", []) or []]
        claims = [Claim(**c) for c in payload.pop("claims", []) or []]
        traces = [AgentTrace(**t) for t in payload.pop("traces", []) or []]
        evidence = {k: Evidence(**v) for k, v in (payload.pop("evidence", {}) or {}).items()}
        known = {f for f in cls.__dataclass_fields__}
        state = cls(**{k: v for k, v in payload.items() if k in known})
        state.budget = budget
        state.tasks = tasks
        state.claims = claims
        state.traces = traces
        state.evidence = evidence
        return state
