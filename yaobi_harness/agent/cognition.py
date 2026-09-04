"""LLM-backed reasoning helpers.

The division of labour: **rules gather signals, the model decides what they
mean.** A keyword screen is good at noticing "尿不出来" in a sentence and bad at
knowing whether one month of back pain with fatigue is an emergency. So the
screen's hits and the model's own semantic findings are both handed to the model
as *material*, and the model returns the triage decision. With no model
configured the rule result stands, unchanged — that is the deterministic
fallback, not the normal path.

What the model still cannot do is unlock a capability or emit a dose: those are
enforced by the capability broker and the dose pipeline, not by second-guessing
the model's clinical reasoning here.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm.base import LLMError
from ..state import ClinicalRunState

RED_FLAG_SYSTEM_PROMPT = """你是骨科急诊分诊医师。规则引擎已经做过关键词筛查，它的命中只是**线索**，
不是结论——关键词筛查会把"乏力""劳累"这类非特异症状读成危险信号。**分诊结论由你做。**

你要做两件事：

1. 找出规则引擎**可能漏掉**的急症信号（口语化表达、方言、隐含描述、症状组合）。
   只报告"当前这位患者本人"的症状；家属病史、假设性提问、已痊愈的既往史不报。
2. 给出**本例的分诊等级**，并说明理由。判断标准是骨科主任在急诊会做的判断：
   - `emergency`：需要数小时内处理（马尾综合征、骨筋膜室、化脓性关节炎、疑似 PE/ACS、
     进行性运动缺损）。
   - `urgent`：需要尽快线下评估但不必叫救护车（明确的感染/肿瘤线索、外伤后不能负重、
     新发神经根性肌力下降）。
   - `routine`：门诊节奏即可（绝大多数慢性腰痛，包括伴乏力、疲劳、睡眠差的）。

**不要因为规则引擎报了某个信号就抬高等级。** 如果它命中的词在本例语境里并不危险，
把它列进 `rule_hits_you_disagree_with` 并说明为什么——你的判断会被采纳，
但会与规则结论一并记入审计台账。反之，规则没报而你认为危险，直接写进 signals。

只输出 JSON：{"triage": "emergency|urgent|routine", "triage_reason": "一到两句，写给同行看",
"signals": [{"signal": "cauda_equina|cardiopulmonary|vascular_dvt_pe|infection_or_tumor|
septic_joint_or_osteomyelitis|fracture|progressive_neuro|cervical_myelopathy|compartment_syndrome|other",
"term": "原文片段", "why": "一句话理由", "certainty": "confirmed|suspected|cannot_exclude"}],
"rule_hits_you_disagree_with": [{"signal": "...", "why": "为什么本例不危险"}]}"""

QUESTION_SYSTEM_PROMPT = """你是骨科问诊助手。根据主诉和已知信息缺口，生成最能改变处置决策的追问。
优先问能排除急症、能决定是否需要线下检查的问题。语言与患者角色匹配：patient 用通俗中文，physician 可用专业术语。
只输出 JSON：{"questions": ["...", "..."]}"""

CRITIC_SYSTEM_PROMPT = """你是临床安全审查员，立场是**证伪**。给定本次运行的结论、证据台账和安全状态，
找出可能导致患者受伤害的问题：证据不支持的断言、被忽略的鉴别诊断、剂量与人群不匹配、遗漏的红旗、
角色越权、引用与结论不符。

只输出 JSON：{"issues": [{"severity": "block|warn", "issue": "问题", "evidence": "依据"}]}
没有问题时输出 {"issues": []}。不要为了凑数编造问题。"""


def _ask(state: ClinicalRunState, llm: Any, messages: list[dict[str, str]], *, max_tokens: int = 800) -> Any:
    """Run one advisory LLM call, charging budget and swallowing all failures."""
    if llm is None or not getattr(llm, "available", False):
        return None
    if not state.budget.reserve_llm():
        state.warn("LLM 预算已用尽，本次咨询回退到规则结果")
        return None
    try:
        response = llm.chat(messages, temperature=0.0, max_tokens=max_tokens, response_format_json=True)
    except (LLMError, Exception) as exc:  # noqa: BLE001 - advisory calls never break the run
        state.warn(f"LLM 咨询失败({type(exc).__name__})，已回退规则结果")
        return None
    state.budget.charge_llm_tokens(response.total_tokens)
    from ..llm.base import extract_json_with_repairs

    payload, repairs = extract_json_with_repairs(response.text, None)
    if repairs:
        # Recorded as a note rather than swallowed. A model whose every answer needs
        # repairing is a prompt problem, and "unclosed" specifically means the reply
        # hit ``max_tokens`` — neither is visible from a successful-looking result.
        state.note("模型输出经 JSON 修复后才可解析: " + "、".join(repairs))
    return payload


#: Triage levels the model may return, in ascending urgency. Anything else is
#: treated as "the model did not answer", so a typo cannot silently downgrade.
TRIAGE_LEVELS = ("routine", "urgent", "emergency")


def triage(
    state: ClinicalRunState,
    llm: Any,
    rule_hits: list[dict[str, Any]],
    soft_hits: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Ask the model to triage this case, given the screen's hits as material.

    Returns ``None`` when no model answered, which is the signal to keep the
    rule-derived risk mode. A returned dict always carries a valid ``triage``
    level, the model's reasoning, the signals it added, and any rule hit it
    thinks does not apply here — all of which are recorded, including the
    disagreements, because "the screen said infection, the model said no, and
    here is why" is exactly what an auditor needs to see.
    """
    payload = _ask(
        state,
        llm,
        [
            {"role": "system", "content": RED_FLAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "text": state.complaint,
                        "facts": state.facts,
                        "rule_engine_hits": rule_hits,
                        "rule_engine_soft_hits": soft_hits or [],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        max_tokens=800,
    )
    if not isinstance(payload, dict):
        return None
    level = str(payload.get("triage") or "").strip().lower()
    signals = payload.get("signals")
    disputed = payload.get("rule_hits_you_disagree_with")
    return {
        "triage": level if level in TRIAGE_LEVELS else "",
        "reason": str(payload.get("triage_reason") or "")[:400],
        "signals": [s for s in signals if isinstance(s, dict)] if isinstance(signals, list) else [],
        "disputed": [d for d in disputed if isinstance(d, dict)] if isinstance(disputed, list) else [],
    }


def semantic_red_flag_signals(state: ClinicalRunState, llm: Any, rule_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Signals only, for callers that do not want a triage decision."""
    result = triage(state, llm, rule_hits)
    return result["signals"] if result else []


def followup_questions(state: ClinicalRunState, llm: Any, fallback: list[str], limit: int) -> list[str]:
    """Generate follow-up questions, falling back to the rule-based list."""
    if limit <= 0:
        return []
    payload = _ask(
        state,
        llm,
        [
            {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "chief_complaint": state.complaint,
                        "role": state.role,
                        "risk_mode": state.risk_mode,
                        "missing_information": state.missing_information,
                        "known_facts": state.facts,
                        "max_questions": limit,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        max_tokens=500,
    )
    questions = []
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        questions = [str(q) for q in payload["questions"] if isinstance(q, str) and q.strip()]
    return (questions or fallback)[:limit]


def adversarial_issues(state: ClinicalRunState, llm: Any) -> list[dict[str, str]]:
    """Ask the model to falsify the run's own conclusions."""
    ledger = [
        {"id": eid, "level": e.level, "source": e.source, "summary": e.summary}
        for eid, e in state.evidence.items()
    ]
    payload = _ask(
        state,
        llm,
        [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "role": state.role,
                        "risk_mode": state.risk_mode,
                        "release_status": state.release_status,
                        "outputs": _trim(state.outputs),
                        "claims": [{"kind": c.kind, "text": c.text, "evidence_ids": c.evidence_ids} for c in state.claims],
                        "evidence_ledger": ledger,
                        "existing_safety_issues": state.safety_issues,
                    },
                    ensure_ascii=False,
                )[:12000],
            },
        ],
        max_tokens=900,
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("issues"), list):
        return []
    issues = []
    for item in payload["issues"]:
        if not isinstance(item, dict) or not item.get("issue"):
            continue
        severity = str(item.get("severity", "warn")).lower()
        issues.append(
            {
                "severity": "block" if severity == "block" else "warn",
                "issue": str(item["issue"])[:300],
                "evidence": str(item.get("evidence", ""))[:300],
            }
        )
    return issues[:10]


def _trim(outputs: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky retrieval payloads before sending context to the model."""
    trimmed = dict(outputs)
    cases = trimmed.get("expert_cases")
    if isinstance(cases, dict):
        trimmed["expert_cases"] = {
            "similar_count": len(cases.get("similar", [])),
            "counterexample_count": len(cases.get("counterexamples", [])),
            "limitation": cases.get("limitation", ""),
        }
    return trimmed
