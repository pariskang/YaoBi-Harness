"""Model-driven tool-calling loop, contained by the same control plane.

This is what makes execution autonomous rather than merely planned: the model
sees the tool schemas its *skill* permits, chooses which to call and with what
arguments, reads the observations, and decides when it has enough to answer.

The containment is what makes that acceptable in a clinical system:

* **Only the skill's tools are visible.** Schemas are filtered to
  ``allowed_tools`` minus ``forbidden_tools``, so the model cannot even name a
  tool it may not use — and every call still passes the broker, which would
  deny it anyway.
* **Observations are evidence.** Each tool result is recorded in the ledger at
  the grade the tool declares, and its evidence id is handed back to the model
  so its answer can cite it.
* **The output is a contract.** The final message must validate against the
  skill's declared schema; anything else is a failure, and the caller falls
  back to its deterministic body.
* **No doses, ever.** Dose generation is not reachable from any loop-enabled
  skill; it stays entirely in the deterministic pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .. import schemas
from ..llm.base import LLMError, ToolSpec
from ..state import ClinicalRunState

MAX_STEPS = 6
MAX_OBSERVATION_CHARS = 4000

SYSTEM_TEMPLATE = """你是骨科临床决策支持系统中的 **{agent}**。

本次运行上下文：交付对象={role}，风险模式={risk_mode}。

## 你的技能：{skill_id}
{skill_description}

{skill_instructions}

## 可用工具
你只能调用下面列出的工具；系统会在执行前再次校验权限，越权调用会被拒绝并计入安全审查。
必要时先调用工具取证，再作答；不要凭空断言。

## 硬性约束
1. **不得输出任何药物克数或剂量数值。** 剂量由独立的确定性链路处理。
2. 每一条结论都要能落到你实际调用过的工具返回的证据上；工具返回里带有 `evidence_id`，
   请在 `citations` 字段中引用它们。
3. 证据不足时如实说明不足，不要用推测填充。
4. 中文作答，面向{role}的表达水平。

## 输出格式
当你取证完毕后，**不要再调用工具**，直接输出一个 JSON 对象（不要加代码块围栏）：
{schema_hint}

其中 `citations` 是你引用的 evidence_id 数组。"""


@dataclass
class LoopStep:
    """One observable move in the loop, kept for the audit trail."""

    step: int
    kind: str  # tool_call | final | error
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    summary: str = ""
    evidence_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step, "kind": self.kind, "tool": self.tool,
            "arguments": self.arguments, "ok": self.ok,
            "summary": self.summary, "evidence_id": self.evidence_id,
        }


@dataclass
class ToolLoopResult:
    ok: bool = False
    output: dict[str, Any] | None = None
    steps: list[LoopStep] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    mode: str = "not_run"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "mode": self.mode, "error": self.error,
            "steps": [s.to_dict() for s in self.steps],
            "evidence_ids": self.evidence_ids, "citations": self.citations,
        }


def serialise_observation(observation: dict[str, Any]) -> str:
    """Serialise a tool observation, shrinking it *structurally* if it is large.

    Slicing the JSON string would hand the model malformed JSON — a retrieval
    result of a few hundred cases is easily past any sane limit, and a cut in
    the middle of a string is unparseable for both the model and anything that
    reads the transcript back. Bulky payloads are therefore summarised into
    counts and previews so the result stays valid JSON.
    """
    text = json.dumps(observation, ensure_ascii=False)
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text

    trimmed = dict(observation)
    trimmed["data"] = _shrink(observation.get("data"), MAX_OBSERVATION_CHARS // 2)
    trimmed["_truncated"] = "过长的检索结果已按结构压缩为计数与样例"
    text = json.dumps(trimmed, ensure_ascii=False)
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    return json.dumps(
        {
            "evidence_id": observation.get("evidence_id"),
            "ok": observation.get("ok", True),
            "summary": str(observation.get("summary", ""))[:400],
            "_truncated": "结果过大，仅保留摘要；如需细节请缩小查询范围",
        },
        ensure_ascii=False,
    )


def _shrink(value: Any, budget: int) -> Any:
    """Recursively replace bulky structures with counts and short previews."""
    if isinstance(value, list):
        preview = [_shrink(v, budget // 4) for v in value[:2]]
        return {"count": len(value), "preview": preview} if len(value) > 2 else preview
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if len(json.dumps(out, ensure_ascii=False)) > budget:
                out["_omitted"] = "字段过多，已省略"
                break
            out[key] = _shrink(item, budget // 2)
        return out
    if isinstance(value, str) and len(value) > 300:
        return value[:300] + "…"
    return value


def schema_hint(schema_name: str) -> str:
    """Describe a named output schema in a form a model can follow."""
    schema = schemas.SCHEMAS.get(schema_name) or {}
    fields = {
        name: ("<必填 " + "/".join(t.__name__ for t in types) + ">") if required
        else ("<可选 " + "/".join(t.__name__ for t in types) + ">")
        for name, (required, types) in schema.items()
    }
    fields["citations"] = "<list[str] 证据ID>"
    return json.dumps(fields, ensure_ascii=False, indent=2)


class ToolLoop:
    """Runs one agent's turn as a bounded ReAct loop."""

    def __init__(
        self,
        llm: Any,
        tools: Any,
        broker: Any,
        state: ClinicalRunState,
        *,
        agent_name: str,
        skill_id: str,
        skill_spec: Any | None = None,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.broker = broker
        self.state = state
        self.agent_name = agent_name
        self.skill_id = skill_id
        self.skill_spec = skill_spec
        self.max_steps = max_steps

    # ------------------------------------------------------------- availability
    @property
    def available(self) -> bool:
        return self.llm is not None and bool(getattr(self.llm, "available", False))

    def allowed_tool_specs(self) -> list[ToolSpec]:
        """Tool schemas the model is allowed to see for this skill."""
        from ..tools import tool_specs

        if self.skill_spec is None:
            return []
        allowed = set(self.skill_spec.allowed_tools) - set(self.skill_spec.forbidden_tools)
        return [spec for spec in tool_specs() if spec.name in allowed]

    # --------------------------------------------------------------------- run
    def run(self, objective: str, context: dict[str, Any], schema_name: str) -> ToolLoopResult:
        result = ToolLoopResult()
        specs = self.allowed_tool_specs()
        if not self.available:
            result.mode, result.error = "llm_unavailable", "no model configured"
            return result
        if not specs:
            result.mode, result.error = "no_tools_for_skill", f"{self.skill_id} grants no tools"
            return result

        messages = [
            {"role": "system", "content": self._system_prompt(schema_name)},
            {"role": "user", "content": json.dumps(
                {"objective": objective, **context}, ensure_ascii=False)[:12000]},
        ]

        for step in range(1, self.max_steps + 1):
            if not self.state.budget.reserve_llm():
                result.mode, result.error = "llm_budget_exhausted", "LLM 预算耗尽"
                return self._finish(result)
            try:
                response = self.llm.chat(messages, tools=specs, temperature=0.0, max_tokens=1600)
            except (LLMError, Exception) as exc:  # noqa: BLE001 - never break the run
                result.steps.append(LoopStep(step, "error", summary=f"{type(exc).__name__}: {exc}"[:200], ok=False))
                result.mode, result.error = "llm_error", f"{type(exc).__name__}"
                return self._finish(result)
            self.state.budget.charge_llm_tokens(response.total_tokens)

            if not response.tool_calls:
                return self._finalize(result, response.text, schema_name, step)

            messages.append(self._assistant_message(response))
            for call in response.tool_calls:
                observation, loop_step = self._execute(call, step)
                result.steps.append(loop_step)
                if loop_step.evidence_id:
                    result.evidence_ids.append(loop_step.evidence_id)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id or f"call_{step}",
                    "name": call.name,
                    "content": serialise_observation(observation),
                })

        result.mode, result.error = "max_steps_exceeded", f"未在 {self.max_steps} 步内给出结论"
        return self._finish(result)

    # ----------------------------------------------------------------- helpers
    def _system_prompt(self, schema_name: str) -> str:
        spec = self.skill_spec
        return SYSTEM_TEMPLATE.format(
            agent=self.agent_name,
            role=self.state.role,
            risk_mode=self.state.risk_mode,
            skill_id=self.skill_id,
            skill_description=getattr(spec, "description", "") or "(未提供描述)",
            skill_instructions=getattr(spec, "instructions", "") or "",
            schema_hint=schema_hint(schema_name),
        )

    @staticmethod
    def _assistant_message(response: Any) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": response.text or None,
            "tool_calls": [
                {
                    "id": call.id or f"call_{index}",
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                }
                for index, call in enumerate(response.tool_calls)
            ],
        }

    def _execute(self, call: Any, step: int) -> tuple[dict[str, Any], LoopStep]:
        """Run one model-chosen tool call through the broker."""
        from ..tools import TOOL_NAMES

        arguments = call.arguments if isinstance(call.arguments, dict) else {}
        if call.name not in TOOL_NAMES:
            # Hand the error back so the model can correct itself instead of
            # the run dying on a hallucinated tool name.
            observation = {"error": f"unknown tool {call.name!r}", "available": sorted(
                s.name for s in self.allowed_tool_specs())}
            return observation, LoopStep(step, "tool_call", call.name, arguments, False, "unknown_tool")

        from .agents import record_tool

        result = self.tools.call(self.broker, call.name, **arguments)
        if result.recoverable:
            # The model called a real tool the wrong way. That is a prompting
            # problem, not a clinical event: return the schema so it can retry,
            # and keep it out of the evidence ledger entirely — otherwise a
            # single malformed argument would count as a failed tool and block
            # the whole run.
            spec = next((s for s in self.allowed_tool_specs() if s.name == call.name), None)
            observation = {
                "ok": False,
                "recoverable": True,
                "error": result.error or result.summary,
                "hint": "参数不正确，请按 parameters 重新调用",
                "parameters": spec.parameters if spec else {},
            }
            return observation, LoopStep(step, "tool_call", call.name, arguments, False,
                                         f"参数错误，可重试: {result.summary}")

        evidence_id = record_tool(self.state, result)
        observation = {
            "evidence_id": evidence_id,
            "ok": result.ok,
            "summary": result.summary,
            "evidence_level": result.resolved_level(),
            "data": result.data,
        }
        if not result.ok:
            observation["error"] = result.error or result.summary
        return observation, LoopStep(step, "tool_call", call.name, arguments, result.ok,
                                     result.summary[:160], evidence_id)

    def _finalize(self, result: ToolLoopResult, text: str, schema_name: str, step: int) -> ToolLoopResult:
        from ..llm.base import extract_json

        payload = extract_json(text, None)
        if not isinstance(payload, dict):
            result.steps.append(LoopStep(step, "final", ok=False, summary="输出不是 JSON 对象"))
            result.mode, result.error = "invalid_output", "final message was not a JSON object"
            return self._finish(result)

        citations = [c for c in (payload.pop("citations", None) or []) if isinstance(c, str)]
        known = set(result.evidence_ids)
        result.citations = [c for c in citations if c in known]

        ok, problems = schemas.validate(schema_name, payload)
        if not ok:
            result.steps.append(LoopStep(step, "final", ok=False, summary="; ".join(problems)[:200]))
            result.mode, result.error = "schema_violation", "; ".join(problems)
            return self._finish(result)

        if self._mentions_dose(payload):
            result.steps.append(LoopStep(step, "final", ok=False, summary="输出中出现剂量数值"))
            result.mode, result.error = "dose_in_output", "model emitted a dose value"
            return self._finish(result)

        result.output = payload
        result.ok = True
        result.mode = "llm_tool_loop"
        result.steps.append(LoopStep(step, "final", ok=True, summary=f"{len(result.evidence_ids)}次取证后作答"))
        return result

    @staticmethod
    def _mentions_dose(payload: dict[str, Any]) -> bool:
        """Reject any model output carrying gram values.

        Doses are produced only by the deterministic pipeline, which checks them
        against expert-case support and an authorised pharmacopoeia range. A
        number that slipped through here would carry none of that.
        """
        import re

        blob = json.dumps(payload, ensure_ascii=False)
        return bool(re.search(r"\d+(?:\.\d+)?\s*(?:克|g\b|G\b|毫克|mg\b)", blob))

    def _finish(self, result: ToolLoopResult) -> ToolLoopResult:
        if result.mode not in ("llm_tool_loop",):
            self.state.warn(f"{self.agent_name}: 模型自主执行未成功({result.mode})，已回退确定性逻辑")
        return result
