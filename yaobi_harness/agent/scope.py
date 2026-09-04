"""Isolated run-state views for concurrent subagents.

Running consult members concurrently is easy; running them concurrently *and*
keeping the audit trail reproducible is the actual problem.

Three things break if members share one :class:`ClinicalRunState`:

* ``state.add_evidence`` allocates ids from ``len(self.evidence)``, so two
  members recording at once can collide, and even when they do not, the id a
  finding gets depends on which HTTP response arrived first. A checkpoint would
  then differ between two runs of the same case — which destroys the one property
  an audit trail exists to provide.
* ``ConsultSubagent`` used to swap ``state.budget`` for the member's slice and
  restore it afterwards. Two members doing that concurrently leave the run with
  whichever slice happened to finish last.
* ``warnings`` and ``traces`` would interleave by completion order.

Locking fixes the corruption but not the nondeterminism. So instead each member
runs against a :class:`MemberScope` — its own evidence dict, warnings, traces,
claims and budget over a shared read-only view of the case — and the results are
merged back **in convened order** once every member has finished. Evidence ids
are allocated during that merge, so they depend on the panel roster rather than
on network timing. Two runs of the same case produce the same ledger.

The cost is that a member cannot see another member's evidence mid-flight. For a
panel that is not a loss: independent opinions are the entire point, and letting
members read each other's findings is how a panel converges on one confident
answer instead of surfacing disagreement.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..state import Budget, ClinicalRunState


@dataclass
class MergeReport:
    """What a merge moved into the parent, for the audit trail."""

    member: str
    evidence_remap: dict[str, str] = field(default_factory=dict)
    warnings: int = 0
    traces: int = 0
    claims: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "member": self.member, "evidence_remap": self.evidence_remap,
            "warnings": self.warnings, "traces": self.traces, "claims": self.claims,
        }


class MemberScope:
    """A private, writable view of a run for one concurrent subagent.

    Reads pass through to a *snapshot* of the parent so a member cannot observe
    another member's mid-flight mutations; writes land in this scope and are
    merged later. Only the fields a subagent legitimately writes are isolated —
    a member has no business setting ``release_status`` or ``risk_mode``, and
    attempting it raises rather than silently doing nothing.
    """

    #: Fields a member may never write. Escalation is the panel's decision, made
    #: after synthesis; a single member reaching in would bypass that.
    FROZEN = ("release_status", "risk_mode", "requires_physician_approval", "tasks", "planner_mode")

    def __init__(self, parent: ClinicalRunState, budget: Budget, *, label: str = "") -> None:
        self._parent = parent
        self.label = label
        # A deep copy of the inputs: a member that mutates a nested fact dict
        # must not have that leak into a sibling running at the same time.
        self.complaint = parent.complaint
        self.role = parent.role
        self.run_id = parent.run_id
        self.facts = copy.deepcopy(parent.facts)
        self.outputs = copy.deepcopy(parent.outputs)
        self.missing_information = list(parent.missing_information)
        self.open_questions = list(parent.open_questions)
        self.images = [dict(image) for image in parent.images]
        self.allow_prescription = parent.allow_prescription
        self.enable_panel = parent.enable_panel
        self.loop_index = parent.loop_index

        # Read-only mirrors of the frozen fields, so a member's prompt sees the
        # real risk posture even though it cannot change it.
        self._risk_mode = parent.risk_mode
        self._release_status = parent.release_status

        # Private, mergeable state.
        self.evidence: dict[str, Any] = {}
        self.claims: list[Any] = []
        self.traces: list[Any] = []
        self.warnings: list[str] = []
        self.safety_issues: list[str] = []
        self.budget = budget

    # ------------------------------------------------------------- read-only
    @property
    def risk_mode(self) -> str:
        return self._risk_mode

    @risk_mode.setter
    def risk_mode(self, value: str) -> None:
        raise PermissionError(
            "会诊子体不能改写 risk_mode：升级由合议后统一决定，单个成员直接改写会绕过最保守优先合议"
        )

    @property
    def release_status(self) -> str:
        return self._release_status

    @release_status.setter
    def release_status(self, value: str) -> None:
        raise PermissionError("会诊子体不能改写 release_status：放行状态只由放行状态机决定")

    # -------------------------------------------------------------- writable
    def add_evidence(
        self,
        level: Any,
        source: str,
        summary: str,
        payload: dict[str, Any] | None = None,
        *,
        ok: bool = True,
        error: str | None = None,
        source_version: str | None = None,
    ) -> str:
        """Record evidence privately, with a scope-local id.

        The id is provisional: :func:`merge_scopes` reassigns it in convened
        order so the parent ledger is deterministic.
        """
        from ..state import Evidence, EvidenceLevel

        local_id = f"{self.label or 'M'}#{len(self.evidence) + 1:03d}"
        data = dict(payload or {})
        data.update({"tool_ok": ok, "tool_error": error})
        resolved = (
            EvidenceLevel.FAILED.value if not ok
            else str(level.value if isinstance(level, EvidenceLevel) else level)
        )
        self.evidence[local_id] = Evidence(local_id, resolved, source, summary, data, source_version)
        return local_id

    def add_claim(
        self,
        kind: str,
        text: str,
        evidence_ids: list[str] | None = None,
        *,
        confidence: float = 0.5,
        origin: str = "rule",
    ) -> str:
        from ..state import Claim

        claim_id = f"{self.label or 'M'}C{len(self.claims) + 1:03d}"
        self.claims.append(Claim(claim_id, kind, text, list(evidence_ids or []), confidence, origin))
        return claim_id

    def trace(
        self,
        agent: str,
        action: str,
        input_summary: str = "",
        output_summary: str = "",
        evidence_ids: list[str] | None = None,
    ) -> None:
        from ..state import AgentTrace

        self.traces.append(AgentTrace(agent, action, input_summary, output_summary, list(evidence_ids or [])))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def releasable_evidence_ids(self) -> set[str]:
        from ..state import NON_RELEASABLE_LEVELS

        return {eid for eid, e in self.evidence.items() if e.level not in NON_RELEASABLE_LEVELS}

    def fail_closed(self, reason: str) -> None:
        """A member cannot fail the run closed; it records the reason instead.

        One specialist's tool outage is not grounds for failing an entire
        consultation. The panel reports the member as having produced no opinion,
        and the run's own critic still sees the recorded issue.
        """
        self.safety_issues.append(f"会诊成员 {self.label or '?'} 报告严重问题: {reason}")

    def __getattr__(self, name: str) -> Any:
        # Anything not isolated above falls through to the parent, read-only.
        # __getattr__ only fires for attributes absent from the instance, so this
        # cannot shadow the isolated fields set in __init__.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._parent, name)


def merge_scopes(
    parent: ClinicalRunState,
    scopes: list[MemberScope],
) -> list[MergeReport]:
    """Fold member scopes into the parent in list order.

    Order is the caller's convened roster, never completion order — that is what
    makes a concurrent panel produce the same ledger as a sequential one.
    Returns per-member remapping so a caller can rewrite the evidence ids its
    opinions cite.
    """
    reports: list[MergeReport] = []
    for scope in scopes:
        remap: dict[str, str] = {}
        for local_id, evidence in scope.evidence.items():
            new_id = parent.add_evidence(
                evidence.level, evidence.source, evidence.summary,
                {k: v for k, v in evidence.payload.items() if k not in ("tool_ok", "tool_error")},
                ok=bool(evidence.payload.get("tool_ok", True)),
                error=evidence.payload.get("tool_error"),
                source_version=evidence.source_version,
            )
            remap[local_id] = new_id

        for claim in scope.claims:
            parent.add_claim(
                claim.kind, claim.text,
                [remap.get(eid, eid) for eid in claim.evidence_ids],
                confidence=claim.confidence, origin=claim.origin,
            )
        for trace in scope.traces:
            trace.evidence_ids = [remap.get(eid, eid) for eid in trace.evidence_ids]
            parent.traces.append(trace)
        parent.warnings.extend(scope.warnings)
        parent.safety_issues.extend(scope.safety_issues)

        reports.append(MergeReport(
            member=scope.label,
            evidence_remap=remap,
            warnings=len(scope.warnings),
            traces=len(scope.traces),
            claims=len(scope.claims),
        ))
    return reports


class TaskScope(MemberScope):
    """A scope for a graph task running beside its independent siblings.

    A consult member and a graph task want almost the same isolation, and differ
    in exactly one place: a member may not touch the run's release status, while
    ``BiomedicalAgent`` escalating to ``needs_examination`` **is** the agent doing
    its job. So the frozen fields are not refused here — they are *recorded*, and
    :func:`merge_task_scopes` replays them in task order.

    Replaying in task order is what makes concurrency invisible in the result.
    Every one of these writes is a conditional escalation of the form "if the
    status is still X, make it Y"; evaluating those conditions against the
    pre-wave value and then applying them in order reproduces what a sequential
    run would have produced, whichever agent's HTTP response happened to land
    first.

    The scope also tracks ``outputs``, ``missing_information``, ``open_questions``
    and ``notes``, which :class:`MemberScope` isolates but never merges back —
    correct for a panel, where a member's private working notes stay private, and
    wrong for a graph task, whose output *is* the point of running it.
    """

    #: The release status is frozen on a panel member and merely *recorded* here.
    #: These two stay genuinely off-limits: a task rewriting the plan mid-wave
    #: would race with the scheduler that is iterating it.
    #:
    #: Enforced, not documented. Without the guard below a write would land as an
    #: instance attribute on the scope, be dropped at merge, and leave no trace —
    #: the failure mode is a task that appears to have rewritten the plan and did
    #: not.
    FROZEN = ("tasks", "planner_mode")

    def __init__(self, parent: ClinicalRunState, budget: Budget, *, label: str = "") -> None:
        super().__init__(parent, budget, label=label)
        self.notes: list[str] = []
        self.status_writes: list[tuple[str, str]] = []
        self._outputs_before = copy.deepcopy(parent.outputs)
        self._missing_before = list(parent.missing_information)
        self._questions_before = list(parent.open_questions)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.FROZEN:
            raise PermissionError(
                f"并行任务不能改写 {name}：计划正在被调度器遍历，任务改写它会与遍历竞争"
            )
        super().__setattr__(name, value)

    # `MemberScope` raises on these; a task legitimately sets them.
    @property
    def risk_mode(self) -> str:
        return self._risk_mode

    @risk_mode.setter
    def risk_mode(self, value: str) -> None:
        self._risk_mode = str(value)
        self.status_writes.append(("risk_mode", str(value)))

    @property
    def release_status(self) -> str:
        return self._release_status

    @release_status.setter
    def release_status(self, value: str) -> None:
        self._release_status = str(value)
        self.status_writes.append(("release_status", str(value)))

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    def fail_closed(self, reason: str) -> None:
        """A task *may* fail the run closed; it is recorded and replayed in order."""
        self.safety_issues.append(reason)
        self.release_status = "failed_closed"

    def new_outputs(self) -> dict[str, Any]:
        """Output keys this task wrote or changed."""
        return {k: v for k, v in self.outputs.items()
                if k not in self._outputs_before or self._outputs_before[k] != v}

    def new_missing(self) -> list[str]:
        return [m for m in self.missing_information if m not in self._missing_before]

    def new_questions(self) -> list[str]:
        return [q for q in self.open_questions if q not in self._questions_before]


def merge_task_scopes(parent: ClinicalRunState, scopes: list["TaskScope"]) -> list[MergeReport]:
    """Fold concurrent graph tasks into the parent, in task order.

    Order is the plan's order, never completion order — the same guarantee
    :func:`merge_scopes` gives a panel, for the same reason.
    """
    reports = merge_scopes(parent, scopes)
    for scope in scopes:
        for key, value in scope.new_outputs().items():
            # Dict-valued outputs merge key-by-key rather than replace. Most
            # output keys have exactly one writer, for which the two are the same
            # thing — but ``outputs["autonomy"]`` is a shared ledger every
            # autonomous agent adds its own entry to, and wholesale replacement
            # silently kept only whichever member of the wave merged last. The
            # run then reported two of its four agents as never having run
            # autonomously, which is a false audit trail, not a slow one.
            existing = parent.outputs.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                parent.outputs[key] = {**existing, **value}
            else:
                parent.outputs[key] = value
        parent.missing_information = list(dict.fromkeys(
            parent.missing_information + scope.new_missing()))
        parent.open_questions = list(dict.fromkeys(
            parent.open_questions + scope.new_questions()))
        for message in scope.notes:
            parent.note(message)
        for field_name, value in scope.status_writes:
            setattr(parent, field_name, value)
    return reports
