"""When may the interview stop?

A model that decides for itself when it has asked enough will stop early — that
is the single most predictable failure of autonomous history-taking. So the
model's claim "问诊已充分" is treated the way Grok Build treats a goal-completion
claim: as a *proposal* that an independent verifier judges, with an explicit
non-progress detector so the loop cannot spin forever either.

Five verdicts, mirroring that design and adapted for the clinical stakes:

``ACHIEVED``      the verifier agrees. Proceed.
``NOT_ACHIEVED``  named gaps remain. Ask again with those gaps fed back.
``STALLED``       two consecutive rounds surfaced the *same* gaps, so more
                  asking is not producing answers. Proceed with the deficit
                  recorded, rather than trapping the patient in a loop.
``CAP_REACHED``   the round cap fired. Same handling as ``STALLED``.
``BLOCKED``       a required red-flag axis is still open. Never waivable.

The clinical inversion of the software original is ``BLOCKED``. Grok's
classifier fails *open* on infrastructure error — a stuck goal is worse than a
wrongly-completed one. Here the asymmetry runs the other way for red flags: if
cauda-equina screening never got an answer, no verdict, no cap, no classifier
outage and no model opinion may release a prescriptive result. Everything else
fails open, because a patient stuck in an endless questionnaire is also a
clinical harm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMError
from .axes import AXES_BY_ID, coverage, required_open_axes

#: How many ask-rounds one interview may take before ``CAP_REACHED``.
MAX_ROUNDS = 6

#: Consecutive rounds with an identical open-gap set before ``STALLED``.
#:
#: Three, not two. Two consecutive identical rounds is only *one* unproductive
#: exchange, and patients routinely answer one thing at a time — volunteering
#: their age while ignoring the bowel-and-bladder question is normal cooperation,
#: not a stall. At two the judge declared ``blocked`` on the second turn of an
#: ordinary conversation.
STALL_THRESHOLD = 3

ACHIEVED = "achieved"
NOT_ACHIEVED = "not_achieved"
STALLED = "stalled"
CAP_REACHED = "cap_reached"
BLOCKED = "blocked"

VERIFIER_SYSTEM_PROMPT = """你是骨科门诊的**问诊充分性审核者**，不是问诊者。

你的任务只有一个：判断目前采集到的病史，是否足以支撑下一步临床决策。请以一位**苛刻的骨科主任医师**
的标准审核——你的默认立场是"还不够"，只有在确实没有影响决策的缺口时才判定充分。

审核要点：
1. 危险信号是否**逐条问过并得到明确回答**（不是"没提到"就算阴性）。
2. 定位信息是否足以形成解剖学假设（节段、侧别、放射范围）。
3. 用药与特殊人群信息是否足以做相互作用与剂量安全判断。
4. 若下一步涉及中医处方，四诊是否有客观锚点（舌脉），而非仅凭主诉推测。
5. 病史中是否存在**互相矛盾**之处需要澄清。

只输出 JSON：
{{"adequate": true/false,
  "missing_axes": ["axis_id", ...],
  "reason": "一句话说明依据",
  "contradictions": ["如有矛盾，逐条列出"]}}

`missing_axes` 只能取自这些 axis_id：{axis_ids}
不要输出诊断、治疗建议或任何剂量。"""


@dataclass
class AdequacyVerdict:
    verdict: str
    reason: str = ""
    missing_axes: list[str] = field(default_factory=list)
    #: Required axes still open. These are the ones that cannot be waived.
    blocking_axes: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    rounds_used: int = 0
    #: ``rule`` when no model ran, ``llm`` when the verifier answered,
    #: ``llm_failed_open`` when it errored and the rule verdict stood in.
    judged_by: str = "rule"

    @property
    def may_proceed(self) -> bool:
        """Whether the interview may hand off, deficit or not."""
        return self.verdict in (ACHIEVED, STALLED, CAP_REACHED)

    @property
    def deficit(self) -> bool:
        """True when we proceed *despite* an incomplete history."""
        return self.verdict in (STALLED, CAP_REACHED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict, "reason": self.reason,
            "missing_axes": self.missing_axes, "blocking_axes": self.blocking_axes,
            "contradictions": self.contradictions, "rounds_used": self.rounds_used,
            "judged_by": self.judged_by,
            "missing_labels": [AXES_BY_ID[a].label for a in self.missing_axes if a in AXES_BY_ID],
            "blocking_labels": [AXES_BY_ID[a].label for a in self.blocking_axes if a in AXES_BY_ID],
        }


class AdequacyJudge:
    """Judges whether history-taking may stop, and detects non-progress.

    One instance per interview: it remembers the gap set of each round, which is
    what makes stall detection possible.
    """

    def __init__(
        self,
        llm: Any | None = None,
        *,
        max_rounds: int = MAX_ROUNDS,
        stall_threshold: int = STALL_THRESHOLD,
    ) -> None:
        self.llm = llm
        self.max_rounds = max_rounds
        self.stall_threshold = stall_threshold
        self.history: list[frozenset[str]] = []

    # --------------------------------------------------------------- judging
    def judge(
        self,
        facts: dict[str, Any],
        complaint: str,
        *,
        role: str = "patient",
        risk_mode: str = "routine",
        prescriptive: bool = False,
        rounds_used: int = 0,
        budget: Any | None = None,
    ) -> AdequacyVerdict:
        """Return the verdict for the current state of the history."""
        blocking = [a.axis_id for a in required_open_axes(
            facts, complaint, role=role, risk_mode=risk_mode, prescriptive=prescriptive)]

        rule_missing = list(blocking)
        verdict = self._ask_model(facts, complaint, role=role, budget=budget)
        if verdict is None:
            missing, reason, contradictions, judged_by = rule_missing, "规则审核：按必答轴判定", [], "rule"
        else:
            model_missing, reason, contradictions = verdict
            # The model may *add* gaps, never remove a rule-required one. Same
            # containment as red flags: cognition widens caution, never narrows it.
            missing = sorted(set(rule_missing) | set(model_missing))
            judged_by = "llm"

        signature = frozenset(missing)
        self.history.append(signature)

        # Blocking axes come first: nothing below can clear them.
        if blocking:
            if self._stalled(signature):
                return AdequacyVerdict(
                    BLOCKED, "反复追问后必答项仍未获答复，不能进入含剂量或处方环节",
                    missing, blocking, contradictions, rounds_used, judged_by)
            if rounds_used >= self.max_rounds:
                return AdequacyVerdict(
                    BLOCKED, f"已达最大追问轮次({self.max_rounds})，必答项仍未闭合",
                    missing, blocking, contradictions, rounds_used, judged_by)
            return AdequacyVerdict(NOT_ACHIEVED, reason, missing, blocking, contradictions, rounds_used, judged_by)

        if not missing and not contradictions:
            return AdequacyVerdict(ACHIEVED, reason or "问诊充分", [], [], [], rounds_used, judged_by)
        if self._stalled(signature):
            return AdequacyVerdict(
                STALLED, "连续两轮追问未取得新信息，带缺口继续并记录在案",
                missing, [], contradictions, rounds_used, judged_by)
        if rounds_used >= self.max_rounds:
            return AdequacyVerdict(
                CAP_REACHED, f"已达最大追问轮次({self.max_rounds})，带缺口继续并记录在案",
                missing, [], contradictions, rounds_used, judged_by)
        return AdequacyVerdict(NOT_ACHIEVED, reason, missing, [], contradictions, rounds_used, judged_by)

    # -------------------------------------------------------------- internals
    def _stalled(self, signature: frozenset[str]) -> bool:
        """True when the last ``stall_threshold`` rounds had identical gaps.

        An empty gap set is never a stall — that is success, and treating it as
        non-progress would be the wrong verdict for the right reason.
        """
        if not signature or len(self.history) < self.stall_threshold:
            return False
        recent = self.history[-self.stall_threshold :]
        return all(entry == signature for entry in recent)

    def _ask_model(
        self,
        facts: dict[str, Any],
        complaint: str,
        *,
        role: str,
        budget: Any | None,
    ) -> tuple[list[str], str, list[str]] | None:
        """Run the adversarial verifier. ``None`` means it did not run."""
        if self.llm is None or not getattr(self.llm, "available", False):
            return None
        if budget is not None and not budget.reserve_llm():
            return None

        report = coverage(facts, complaint, role=role)
        catalogue = {
            axis_id: {"label": AXES_BY_ID[axis_id].label, "tier": AXES_BY_ID[axis_id].tier}
            for axis_id in report["open"]
            if axis_id in AXES_BY_ID
        }
        try:
            response = self.llm.chat(
                [
                    {"role": "system", "content": VERIFIER_SYSTEM_PROMPT.format(
                        axis_ids=", ".join(sorted(AXES_BY_ID)))},
                    {"role": "user", "content": json.dumps(
                        {
                            "narrative": complaint[:4000],
                            "collected_facts": _redact(facts),
                            "axes_answered": report["answered"],
                            "axes_open": catalogue,
                        },
                        ensure_ascii=False,
                    )},
                ],
                temperature=0.0, max_tokens=700, response_format_json=True,
            )
            if budget is not None:
                budget.charge_llm_tokens(response.total_tokens)
        except (LLMError, Exception):  # noqa: BLE001 - fail open to the rule verdict
            return None

        payload = response.json(None)
        if not isinstance(payload, dict):
            return None
        missing = [a for a in (payload.get("missing_axes") or []) if isinstance(a, str) and a in AXES_BY_ID]
        contradictions = [str(c) for c in (payload.get("contradictions") or []) if str(c).strip()][:5]
        reason = str(payload.get("reason") or "")[:300]
        # An explicit `adequate: false` with no named axis is still a gap signal;
        # keep it visible rather than silently reading it as "adequate".
        if payload.get("adequate") is False and not missing and not contradictions:
            contradictions = ["审核者认为病史不足但未指明具体缺口"]
        return missing, reason, contradictions


def _redact(facts: dict[str, Any]) -> dict[str, Any]:
    """Drop anything the verifier has no business seeing.

    ``physician_review`` carries a signature; a prescription draft carries doses.
    Neither belongs in an adequacy prompt, and excluding them here means a
    prompt-injected reply cannot echo them back into the run.
    """
    return {k: v for k, v in facts.items() if k not in ("physician_review", "prescription_draft", "signature")}
