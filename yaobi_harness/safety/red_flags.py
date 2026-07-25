"""Red-flag screening for orthopaedic and adjacent emergencies.

Design rules, in priority order:

1. **Asymmetric cost.** A missed cauda equina syndrome or acute coronary
   syndrome is catastrophic; an unnecessary emergency-department referral is
   not. Screening therefore *defaults to escalation* whenever context is
   ambiguous, and suppression requires positive evidence.
2. **Clause-scoped suppression.** A negation, family-history or past-history
   cue only suppresses terms inside the *same clause*. The previous
   character-window approach let ``既往体健，现突发胸痛`` suppress a live chest
   pain, which is exactly the failure mode this module exists to prevent.
3. **Two tiers.** ``HARD`` signals escalate on their own. ``SOFT`` signals
   (e.g. isolated night pain) escalate only with corroboration, and otherwise
   downgrade to "needs examination" rather than being discarded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Emergencies that must be excluded before any remote treatment advice.
#: Categories cover spinal, limb, infective, oncological, vascular and
#: cardiopulmonary causes seen in an orthopaedic front door.
HARD_RED_FLAGS: dict[str, list[str]] = {
    "cauda_equina": [
        "尿潴留", "不能排尿", "尿不出来", "尿不出", "解不出小便", "小便解不出来", "排尿困难",
        "大小便失禁", "小便失禁", "大便失禁", "尿失禁", "尿憋不住", "憋不住尿", "控制不住大小便",
        "会阴麻木", "鞍区麻木", "下身迟钝", "肛周麻木", "屁股麻木", "私处麻木", "性功能突然减退",
    ],
    "cardiopulmonary": [
        "胸痛", "胸闷", "胸口疼", "心前区疼", "大汗", "冷汗", "呼吸困难", "喘不上气",
        "气促", "脸色苍白", "面色苍白", "晕厥", "昏倒", "意识不清", "咯血",
    ],
    "vascular_dvt_pe": [
        "小腿肿胀", "单侧腿肿", "下肢肿胀", "腿突然肿", "深静脉血栓", "肺栓塞", "小腿压痛肿胀",
    ],
    "infection_or_tumor": [
        "发热", "寒战", "高烧", "体重下降", "消瘦", "肿瘤", "癌", "结核",
        "静脉吸毒", "免疫抑制", "长期激素", "化疗",
    ],
    "septic_joint_or_osteomyelitis": [
        "关节红肿热痛", "关节剧痛不能动", "关节化脓", "伤口流脓", "术后伤口红肿", "骨髓炎",
    ],
    "fracture": [
        "外伤", "跌倒", "摔倒", "骨折", "车祸", "高处坠落", "撞击后剧痛", "无法负重", "不能站立",
    ],
    "progressive_neuro": [
        "肌力下降", "足下垂", "进行性麻木", "越来越无力", "双腿越来越无力", "抬不起脚",
        "拿不住东西", "走路踩棉花", "行走不稳", "四肢无力",
    ],
    "cervical_myelopathy": [
        "手笨拙", "系扣子困难", "写字变差", "踩棉花感", "束带感", "颈部外伤后麻木",
    ],
    "compartment_syndrome": [
        "石膏后剧痛", "被动牵拉痛", "肢体张力高", "肢端发凉苍白", "疼痛与体征不符", "剧痛进行性加重",
    ],
}

#: Signals that are concerning but non-specific on their own.
SOFT_RED_FLAGS: dict[str, list[str]] = {
    "night_pain": ["夜间痛", "夜间腰痛", "夜间疼痛", "夜里痛醒", "痛醒"],
    "chronic_bone_fragility": ["骨质疏松", "骨量减少"],
    "age_extremes": ["高龄", "老年人首次腰痛"],
}

#: Corroborating findings that promote a soft signal to a hard one.
SOFT_CORROBORATION = [
    "新发", "突然", "加重", "发热", "外伤", "跌倒", "肿瘤", "癌", "体重下降", "消瘦",
    "夜间加重", "静息痛", "无法缓解", "进行性",
]

NEGATION_CUES = ["无", "否认", "未见", "未出现", "没有", "不伴", "无明显", "排除", "不存在", "未诉"]
THIRD_PARTY_CUES = ["父亲", "母亲", "家族", "亲属", "家人", "朋友", "别人", "他人", "同事"]
HYPOTHETICAL_CUES = ["如果", "假如", "万一", "担心", "害怕", "会不会", "怎么办", "是否会"]
HISTORY_CUES = ["既往", "去年", "多年前", "以前", "曾经", "曾", "陈旧", "已痊愈", "已恢复", "慢性", "旧伤"]
#: Cues that re-anchor a clause to the present and therefore veto history cues.
PRESENT_CUES = [
    "现", "目前", "今日", "今天", "近日", "刚", "刚才", "突然", "突发", "新发", "此次", "本次",
    "近期", "这两天", "最近", "正在",
]
CONTRAST_CUES = ["但", "但是", "然而", "不过", "却"]

CLAUSE_SPLIT_RE = re.compile(r"[，,。；;、\n\r\t ]+")


@dataclass
class RedFlagHit:
    signal: str
    term: str
    tier: str
    clause: str
    reason: str
    source: str = "contextual_red_flag_screen"

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "term": self.term,
            "tier": self.tier,
            "clause": self.clause,
            "reason": self.reason,
            "source": self.source,
        }


@dataclass
class ScreenResult:
    hits: list[RedFlagHit] = field(default_factory=list)
    soft_hits: list[RedFlagHit] = field(default_factory=list)
    suppressed: list[dict[str, str]] = field(default_factory=list)
    ambiguous: bool = False

    @property
    def urgent(self) -> bool:
        return bool(self.hits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": [h.to_dict() for h in self.hits],
            "soft_hits": [h.to_dict() for h in self.soft_hits],
            "suppressed": self.suppressed,
            "ambiguous": self.ambiguous,
        }


def split_clauses(text: str) -> list[str]:
    """Split free text into clauses; negation scope never crosses a boundary."""
    return [c for c in CLAUSE_SPLIT_RE.split(text or "") if c]


def _contains(clause: str, cues: list[str]) -> str | None:
    return next((c for c in cues if c in clause), None)


def _is_negated(clause: str, term: str) -> bool:
    """True when a negation cue governs ``term`` inside this clause.

    The cue must appear *before* the term and must not be separated from it by
    a contrast marker (``无高血压病史但今日胸痛`` keeps the chest pain).
    """
    idx = clause.find(term)
    if idx < 0:
        return False
    left = clause[:idx]
    cue_pos = max((left.rfind(c) for c in NEGATION_CUES if c in left), default=-1)
    if cue_pos < 0:
        return False
    between = left[cue_pos:]
    if any(c in between for c in CONTRAST_CUES):
        return False
    # A long gap between cue and term usually means the cue governs something else.
    return len(between) <= 12


def _classify_clause(clause: str, term: str) -> tuple[bool, str]:
    """Return ``(suppressed, reason)`` for ``term`` inside ``clause``."""
    if _is_negated(clause, term):
        return True, "negated_in_clause"
    third = _contains(clause, THIRD_PARTY_CUES)
    if third:
        return True, f"third_party:{third}"
    hypo = _contains(clause, HYPOTHETICAL_CUES)
    if hypo:
        return True, f"hypothetical:{hypo}"
    history = _contains(clause, HISTORY_CUES)
    present = _contains(clause, PRESENT_CUES)
    if history and not present:
        return True, f"historical:{history}"
    return False, "current_patient_symptom"


def screen(text: str) -> ScreenResult:
    """Screen ``text`` for emergency signals.

    Suppression is clause-local and requires an explicit cue; anything else is
    treated as a live symptom of the patient in front of us.
    """
    result = ScreenResult()
    clauses = split_clauses(text)
    corroborated = any(c in (text or "") for c in SOFT_CORROBORATION)

    for clause in clauses:
        for signal, terms in HARD_RED_FLAGS.items():
            for term in terms:
                if term not in clause:
                    continue
                suppressed, reason = _classify_clause(clause, term)
                if suppressed:
                    result.suppressed.append({"signal": signal, "term": term, "reason": reason, "clause": clause})
                else:
                    result.hits.append(RedFlagHit(signal, term, "hard", clause, reason))

        for signal, terms in SOFT_RED_FLAGS.items():
            for term in terms:
                if term not in clause:
                    continue
                suppressed, reason = _classify_clause(clause, term)
                if suppressed:
                    result.suppressed.append({"signal": signal, "term": term, "reason": reason, "clause": clause})
                elif corroborated:
                    result.hits.append(RedFlagHit(signal, term, "soft_corroborated", clause, "corroborated_by_context"))
                else:
                    result.soft_hits.append(RedFlagHit(signal, term, "soft", clause, reason))

    # Deduplicate while keeping order.
    seen: set[tuple[str, str]] = set()
    deduped: list[RedFlagHit] = []
    for hit in result.hits:
        key = (hit.signal, hit.term)
        if key not in seen:
            seen.add(key)
            deduped.append(hit)
    result.hits = deduped
    result.ambiguous = bool(result.soft_hits) and not result.hits
    return result


def merge_llm_hits(result: ScreenResult, llm_signals: list[dict[str, Any]] | None) -> ScreenResult:
    """Union semantic screening results into the rule-based result.

    The LLM may only **add** signals. It can never clear a rule-based hit, so a
    hallucinating or prompt-injected model cannot downgrade a triage decision.
    """
    for raw in llm_signals or []:
        signal = str(raw.get("signal") or "llm_semantic_risk")
        term = str(raw.get("term") or raw.get("quote") or "")[:40]
        if any(h.signal == signal and h.term == term for h in result.hits):
            continue
        result.hits.append(
            RedFlagHit(signal, term, "llm_semantic", str(raw.get("clause") or "")[:80], "llm_semantic_escalation", source="llm_screen")
        )
    return result
