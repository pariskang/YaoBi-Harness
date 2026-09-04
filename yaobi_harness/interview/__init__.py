"""Model-driven history taking: 十问歌 + orthopaedic specialty enquiry.

The model composes the questions; rule-derived axes decide what must be covered;
an independent judge decides when asking may stop. See :mod:`.axes`,
:mod:`.loop` and :mod:`.adequacy`.
"""

from .adequacy import (
    ACHIEVED, BLOCKED, CAP_REACHED, NOT_ACHIEVED, STALLED,
    AdequacyJudge, AdequacyVerdict,
)
from .axes import (
    AXES, AXES_BY_ID, TIERS, Axis, coverage, open_axes, plan_next,
    relevant_axes, required_open_axes,
)
from .loop import ASK_TOOL, InterviewLoop, InterviewQuestion, InterviewRound

__all__ = [
    "ACHIEVED", "AXES", "AXES_BY_ID", "ASK_TOOL", "BLOCKED", "CAP_REACHED",
    "NOT_ACHIEVED", "STALLED", "TIERS", "AdequacyJudge", "AdequacyVerdict",
    "Axis", "InterviewLoop", "InterviewQuestion", "InterviewRound",
    "coverage", "open_axes", "plan_next", "relevant_axes", "required_open_axes",
]
