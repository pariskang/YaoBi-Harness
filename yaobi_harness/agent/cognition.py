"""LLM-backed advisory helpers.

Every function here is *additive only*: it can raise a risk, add a question or
add a safety objection, but it can never clear a rule-based finding, unlock a
capability or produce a dose. If the model is unavailable, errors, exceeds
budget or returns nonsense, the caller keeps its deterministic result. That
asymmetry is what makes it safe to put a third-party model in the loop.
"""

from __future__ import annotations

import json
from typing import Any

from ..llm.base import LLMError
from ..state import ClinicalRunState

RED_FLAG_SYSTEM_PROMPT = """你是骨科急诊分诊的安全网。规则引擎已经做过关键词筛查，你的唯一职责是
找出规则引擎**可能漏掉**的当前患者急症信号（口语化表达、方言、隐含描述、症状组合）。

原则：
- 只做加法。不要评价或推翻规则引擎已命中的信号。
- 宁可多报不可漏报：不确定时报出来，由下游人工判断。
- 只报告"当前这位患者本人"的症状；家属病史、假设性提问、已痊愈的既往史不报。

只输出 JSON：{"signals": [{"signal": "cauda_equina|cardiopulmonary|vascular_dvt_pe|infection_or_tumor|
septic_joint_or_osteomyelitis|fracture|progressive_neuro|cervical_myelopathy|compartment_syndrome|other",
"term": "原文片段", "why": "一句话理由"}]}"""

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
    return response.json(None)


def semantic_red_flag_signals(state: ClinicalRunState, llm: Any, rule_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ask the model for red flags the keyword screen may have missed."""
    payload = _ask(
        state,
        llm,
        [
            {"role": "system", "content": RED_FLAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"text": state.complaint, "facts": state.facts, "rule_engine_hits": rule_hits},
                    ensure_ascii=False,
                ),
            },
        ],
        max_tokens=600,
    )
    if not isinstance(payload, dict):
        return []
    signals = payload.get("signals")
    return [s for s in signals if isinstance(s, dict)] if isinstance(signals, list) else []


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
