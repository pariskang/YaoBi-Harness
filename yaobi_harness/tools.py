"""Tools, capability brokering and the de-identified expert-case store.

Three invariants hold here regardless of which agent (rule-based or LLM-driven)
is calling:

* **The broker is the only door to a tool.** Role, risk mode, skill policy,
  circuit-breaker health and budget are all checked before execution, and the
  budget is charged only for calls that actually run.
* **Tools declare their own evidence grade.** A placeholder knowledge source
  returns ``STUB`` and no consumer can promote it to guideline grade.
* **Dose safety is checked against the proposed dose**, not merely against the
  existence of an authorised range.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import statistics
import threading
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .knowledge import ortho_interactions
from .knowledge.store import KnowledgeStore
from .llm.base import ToolSpec
from .safety import incompatibility as incompat
from .safety import red_flags
from .state import EvidenceLevel

DIRECT_IDENTIFIER_FIELDS = {"姓名", "病案号", "地址", "医师工号", "医师姓名", "科室代码", "就诊序号", "身份证号", "电话", "联系方式"}
DATE_FIELDS = {"就诊日期"}
AUTHORIZED_FIELDS = {
    "性别", "年龄", "就诊月份", "主诉", "现病史", "既往史", "过敏史", "手术史",
    "中医四诊", "辅助检查", "中医诊断", "西医诊断", "治疗方法", "西药", "中药", "治疗",
}
HERB_ITEM_RE = re.compile(
    r"(?:^|[,，\n])\s*\d*\s*/?\s*(?:\[[^\]]+\])?\s*\*?\s*([^*/，,\n]+?)\s*\*?\s*1\s*"
    r"(?:克|g|G|mg|m1g)\s*/\s*(\d+(?:\.\d+)?)\s*(?:克|g|G)\s*/\s*用法[:：]\s*([^/，,\n]*)"
)
DLP_PATTERNS = [
    re.compile(r"1[3-9]\d{9}"),                                  # mainland mobile numbers
    re.compile(r"\b\d{17}[\dXx]\b"),                             # 18-digit national ID
    re.compile(r"\b\d{15}\b"),                                   # legacy 15-digit national ID
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                      # e-mail
    re.compile(r"[一-鿿]{2,8}(?:街道|社区|小区|村|路|号楼|单元|镇|乡)"),
    re.compile(r"(?:20|19)\d{2}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?"),  # full dates
]

#: Re-exported for backwards compatibility; the authoritative table lives in
#: :mod:`yaobi_harness.safety.incompatibility`.
RISK_HERBS = incompat.RISK_HERBS
RED_FLAGS = red_flags.HARD_RED_FLAGS

HERB_ALIASES = {
    "杜仲": ["杜仲", "盐杜仲"], "甘草": ["甘草", "甘草片", "炙甘草"], "桃仁": ["桃仁", "燀山桃仁"],
    "延胡索": ["延胡索", "醋延胡索"], "白芍": ["白芍", "麸白芍"], "牛膝": ["牛膝", "川牛膝"],
    "党参": ["党参", "炒党参"], "茯苓": ["茯苓"], "独活": ["独活"], "桑寄生": ["桑寄生"],
    "当归": ["当归"], "川芎": ["川芎"], "熟地黄": ["熟地黄"], "红花": ["红花"],
}

#: Minimum expert cases required before a median dose may be proposed.
MIN_DOSE_SAMPLE_N = 3
#: Absolute sanity ceiling used only when no authorised range exists.
ABSOLUTE_DOSE_CEILING_G = 60.0


class DeidentificationKeyError(RuntimeError):
    """Raised when a stable pseudonymisation key is missing.

    Falling back to a random key would silently produce a different research ID
    for the same patient on every call, breaking longitudinal retrieval while
    looking like it works.
    """


@dataclass
class ToolResult:
    tool: str
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    evidence_level: str = EvidenceLevel.TOOL.value
    error: str | None = None
    source_version: str | None = None
    #: True when backed by placeholder data that must never be released.
    is_stub: bool = False
    #: True for transport-style failures that are worth retrying.
    retryable: bool = False
    #: True when the *caller* got the call wrong (bad or missing arguments,
    #: unknown tool name). A model can correct these on its next turn, so they
    #: must not be recorded as tool failures, must not trip the circuit breaker,
    #: and must not block the run.
    recoverable: bool = False

    def resolved_level(self) -> str:
        if not self.ok:
            return EvidenceLevel.FAILED.value
        if self.is_stub:
            return EvidenceLevel.STUB.value
        return self.evidence_level


def _tool_result_to_dict(result: ToolResult) -> dict[str, Any]:
    """Serialise a tool result for the journal."""
    return {
        "tool": result.tool, "ok": result.ok, "summary": result.summary, "data": result.data,
        "evidence_level": result.evidence_level, "error": result.error,
        "source_version": result.source_version, "is_stub": result.is_stub,
        "retryable": result.retryable, "recoverable": result.recoverable,
    }


def _tool_result_from_dict(name: str, payload: Any) -> ToolResult:
    """Rebuild a tool result from the journal, defensively.

    A journal is an input like any other, so a malformed record produces a failed
    result rather than an exception — the run then treats it as a tool failure and
    fails closed, which is the correct response to unusable evidence.
    """
    if not isinstance(payload, dict):
        return ToolResult(name, False, "journal_replay_malformed",
                          error="journal entry was not an object", retryable=False)
    return ToolResult(
        tool=str(payload.get("tool") or name),
        ok=bool(payload.get("ok")),
        summary=str(payload.get("summary") or ""),
        data=payload.get("data") if isinstance(payload.get("data"), dict) else {},
        evidence_level=str(payload.get("evidence_level") or EvidenceLevel.TOOL.value),
        error=payload.get("error"),
        source_version=payload.get("source_version"),
        is_stub=bool(payload.get("is_stub")),
        retryable=bool(payload.get("retryable")),
        recoverable=bool(payload.get("recoverable")),
    )


class ToolHealth:
    """Circuit breaker shared by every broker within one run.

    ``record_failure`` is a read-modify-write on a per-tool counter, and consult
    panel members run concurrently against one breaker, so every method holds a
    lock. Without it two members failing the same tool at once can each read 0
    and each write 1, leaving the circuit closed after two failures.
    """

    def __init__(self, failure_threshold: int = 2) -> None:
        self.failure_threshold = failure_threshold
        self.consecutive_failures: dict[str, int] = {}
        self.open_circuits: set[str] = set()
        self._lock = threading.RLock()

    def is_healthy(self, tool: str) -> bool:
        with self._lock:
            return tool not in self.open_circuits

    def record_success(self, tool: str) -> None:
        with self._lock:
            self.consecutive_failures.pop(tool, None)
            self.open_circuits.discard(tool)

    def record_failure(self, tool: str) -> None:
        with self._lock:
            count = self.consecutive_failures.get(tool, 0) + 1
            self.consecutive_failures[tool] = count
            if count >= self.failure_threshold:
                self.open_circuits.add(tool)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "open_circuits": sorted(self.open_circuits),
                "consecutive_failures": dict(self.consecutive_failures),
            }


class CapabilityBroker:
    """Single choke point for tool authorisation.

    Check order matters: policy first, budget last, so a denied call can never
    drain the run budget.
    """

    URGENT_FORBIDDEN = {"formula_composition_search", "herb_dose_distribution", "physician_review_submit"}
    PATIENT_FORBIDDEN = URGENT_FORBIDDEN

    def __init__(
        self,
        role: str,
        risk_mode: str,
        budget: Any | None = None,
        skill_registry: Any | None = None,
        active_skill: str | None = None,
        health: ToolHealth | None = None,
        *,
        require_skill: bool = True,
        journal: Any | None = None,
    ) -> None:
        self.role = role
        self.risk_mode = risk_mode
        self.budget = budget
        self.skill_registry = skill_registry
        self.active_skill = active_skill
        self.health = health or ToolHealth()
        self.require_skill = require_skill
        #: Optional call journal. It rides on the broker because the broker is
        #: already the single door every tool call passes through, and it is
        #: constructed once per run — the tool registry is shared across runs.
        self.journal = journal

    def allow(self, tool: str) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` without consuming any budget."""
        if not self.health.is_healthy(tool):
            return False, "tool_unhealthy_circuit_open"
        if self.risk_mode == "urgent" and tool in self.URGENT_FORBIDDEN:
            return False, "urgent_mode_forbids_prescription_or_dose"
        if self.role == "patient" and tool in self.PATIENT_FORBIDDEN:
            return False, "patient_role_forbids_formula_or_dose_tools"
        if self.skill_registry is not None:
            # Fail closed: an agent with no declared skill has no tool rights.
            if not self.active_skill:
                if self.require_skill:
                    return False, "skill_policy_denied:no_active_skill_declared_for_agent"
            else:
                ok, problems = self.skill_registry.enforce(self.active_skill, self.role, [tool])
                if not ok:
                    return False, "skill_policy_denied:" + ";".join(problems)
        if self.budget is not None and not self.budget.can_afford_tool():
            return False, "budget_exhausted"
        return True, "allowed"

    def charge(self) -> None:
        if self.budget is not None:
            self.budget.charge_tool()


class ToolRegistry:
    """Executable tool implementations plus their LLM-facing schemas."""

    def __init__(
        self,
        xlsx_path: str | Path | None = None,
        records: list[dict[str, Any]] | None = None,
        failing_tools: set[str] | None = None,
        authorized_ranges: dict[str, tuple[float, float]] | None = None,
        *,
        deid_key: str | None = None,
        max_attempts: int = 2,
        knowledge: KnowledgeStore | None = None,
        drug_normalizer: Any | None = None,
        vision: Any | None = None,
    ) -> None:
        if records is not None:
            self.case_store = ExpertCaseStore.from_records(records, deid_key=deid_key)
        elif xlsx_path:
            self.case_store = ExpertCaseStore(xlsx_path, deid_key=deid_key)
        else:
            self.case_store = ExpertCaseStore.empty()
        self.failing_tools = set(failing_tools or set())
        self.authorized_ranges = dict(authorized_ranges or {})
        self.max_attempts = max(1, max_attempts)
        #: Optional licensed knowledge store (guidelines, labels, dose ranges, DDIs).
        self.knowledge = knowledge
        #: Optional live RxNorm connector for medication name resolution.
        self.drug_normalizer = drug_normalizer
        #: Optional multimodal client. Absent means the image tools report
        #: themselves unavailable rather than the harness failing to start.
        self.vision = vision
        self.tools: dict[str, Callable[..., ToolResult]] = {
            "red_flag_evidence_search": self.red_flag_evidence_search,
            "similar_case_search": self.similar_case_search,
            "counterexample_case_search": self.counterexample_case_search,
            "herb_dose_distribution": self.herb_dose_distribution,
            "tcm_pattern_knowledge_search": self.tcm_pattern_knowledge_search,
            "clinical_guideline_search": self.clinical_guideline_search,
            "pharmacopeia_check": self.pharmacopeia_check,
            "interaction_check": self.interaction_check,
            "special_population_check": self.special_population_check,
            "emergency_resource_lookup": self.emergency_resource_lookup,
            "formula_composition_search": self.formula_composition_search,
            "physician_review_submit": self.physician_review_submit,
            "patient_timeline_search": self.patient_timeline_search,
            "drug_interaction_check": self.drug_interaction_check,
            "drug_label_lookup": self.drug_label_lookup,
            "drug_normalize": self.drug_normalize,
            "expert_practice_profile": self.expert_practice_profile,
            "interview_axis_lookup": self.interview_axis_lookup,
            "medical_image_read": self.medical_image_read,
        }
        self._profile = None
        self._profile_lock = threading.RLock()

    # ------------------------------------------------------------- dispatching
    def call(self, broker: CapabilityBroker, name: str, **kwargs: Any) -> ToolResult:
        """Authorise, execute (with bounded retry) and account for one tool call.

        Authorisation runs *before* the journal is consulted. A replay therefore
        re-derives every policy decision live: a journal recorded as a physician
        cannot hand a patient-role replay a formula result, because the broker
        denies the call before the recorded result is ever reached. The journal
        supplies data, never permission.
        """
        allowed, reason = broker.allow(name)
        if not allowed:
            # Policy denials and budget exhaustion are *not* tool failures: they
            # must not trip the circuit breaker or consume budget.
            return ToolResult(name, False, reason, {"denied_reason": reason}, error=reason)
        if name not in self.tools:
            return ToolResult(name, False, "unknown_tool", error="unknown_tool", recoverable=True)

        journal = getattr(broker, "journal", None)
        if journal is not None:
            hit, recorded = journal.next_result("tool", name, kwargs)
            if hit:
                broker.charge()  # a replayed call still consumes the run's budget
                replayed = _tool_result_from_dict(name, recorded)
                if replayed.ok:
                    broker.health.record_success(name)
                elif not replayed.recoverable:
                    broker.health.record_failure(name)
                return replayed

        final = self._execute_with_retry(broker, name, kwargs)
        if journal is not None:
            # Recorded once, on every exit path, and only the *final* result: a
            # replay reproduces the outcome the run acted on, not the transient
            # failures on the way to it.
            journal.record("tool", name, kwargs, _tool_result_to_dict(final))
        return final

    def _execute_with_retry(self, broker: CapabilityBroker, name: str, kwargs: dict[str, Any]) -> ToolResult:
        """Run one tool, retrying transport-style failures within the budget."""
        result: ToolResult | None = None
        for attempt in range(self.max_attempts):
            broker.charge()
            if name in self.failing_tools:
                result = ToolResult(name, False, "tool_failure", error="injected_tool_failure", retryable=True)
            else:
                try:
                    result = self.tools[name](**kwargs)
                except TypeError as exc:
                    # A caller error, not a tool failure: hand it back so the
                    # caller (often a model) can fix its arguments and retry.
                    result = ToolResult(name, False, "tool_bad_arguments", error=repr(exc), recoverable=True)
                except Exception as exc:  # noqa: BLE001 - surfaced as evidence, never swallowed
                    result = ToolResult(name, False, "tool_exception", error=repr(exc), retryable=True)
            if result.ok:
                broker.health.record_success(name)
                return result
            if result.recoverable:
                # The tool itself is fine; do not open its circuit.
                return result
            broker.health.record_failure(name)
            if not result.retryable or attempt == self.max_attempts - 1:
                break
            if broker.budget is not None and not broker.budget.can_afford_tool():
                break
        return result or ToolResult(name, False, "tool_failure", error="unknown")

    # ------------------------------------------------------------------ triage
    def red_flag_evidence_search(self, text: str) -> ToolResult:
        screen = red_flags.screen(text)
        return ToolResult(
            "red_flag_evidence_search",
            True,
            f"命中{len(screen.hits)}个当前本人硬性风险信号，{len(screen.soft_hits)}个待证实弱信号",
            screen.to_dict(),
            source_version="red_flags.clause_scoped.v4",
        )

    def emergency_resource_lookup(self, location: str = "中国大陆") -> ToolResult:
        phone = "120" if "中国" in location else None
        return ToolResult(
            "emergency_resource_lookup",
            True,
            "急救资源",
            {
                "emergency_phone": phone,
                "advice": "如所在地急救号码未知，请使用当地官方急救电话；中国大陆为120",
            },
        )

    # --------------------------------------------------------------- knowledge
    def clinical_guideline_search(self, topic: str, limit: int = 5) -> ToolResult:
        """Search the licensed guideline store, falling back to a labelled stub.

        A hit from the store is genuine guideline-grade evidence and carries the
        guideline name, version, publication date and evidence grade. With no
        store configured — or no match — the result is explicitly a stub and is
        recorded as ``stub_not_for_clinical_use``.
        """
        if self.knowledge is not None:
            hits = self.knowledge.search_guidelines(topic, limit=limit)
            if hits:
                return ToolResult(
                    "clinical_guideline_search",
                    True,
                    f"授权指南库命中{len(hits)}条: {hits[0]['title']}",
                    {
                        "topic": topic,
                        "guidelines": hits,
                        "citations": [h["provenance"] for h in hits],
                        "points": [r for h in hits for r in h.get("recommendations", [])][:12],
                    },
                    evidence_level=EvidenceLevel.GUIDELINE.value,
                    source_version=hits[0]["provenance"].get("version") or hits[0]["provenance"].get("source_id"),
                )
        return ToolResult(
            "clinical_guideline_search",
            True,
            "本地占位指南摘要（未配置授权指南数据源）",
            {
                "topic": topic,
                "guideline_id": "local_stub.not_for_clinical_release",
                "guidelines": [],
                "points": [
                    "先筛查马尾综合征、感染、肿瘤、骨折、进行性神经缺损、骨筋膜室综合征及非腰痛急症",
                    "红旗或持续/进展神经根症状需线下评估与影像",
                ],
                "how_to_fix": "运行 `python -m yaobi_harness knowledge build` 并按授权摄取 NICE/VA-DoD/中华医学会 指南",
            },
            evidence_level=EvidenceLevel.GUIDELINE.value,
            source_version="stub-guideline-v3",
            is_stub=True,
        )

    def tcm_pattern_knowledge_search(self, text: str) -> ToolResult:
        patterns = []
        if any(x in text for x in ["刺痛", "固定", "麻木", "久坐"]):
            patterns.append("气滞血瘀证")
        if any(x in text for x in ["乏力", "酸软", "久病", "劳累"]):
            patterns.append("气血痹阻证")
        if any(x in text for x in ["冷痛", "畏寒", "怕冷"]):
            patterns.append("寒湿痹阻证")
        return ToolResult(
            "tcm_pattern_knowledge_search",
            True,
            "证候知识匹配（规则提示）",
            {"patterns": patterns or ["待辨证"], "limits": "规则提示仅作证据检索入口，不替代四诊合参"},
            is_stub=True,
        )

    def formula_composition_search(self, pattern: str) -> ToolResult:
        base = ["独活", "桑寄生", "杜仲", "牛膝", "当归", "川芎", "白芍", "熟地黄", "党参", "茯苓", "甘草"]
        if "瘀" in pattern:
            base += ["桃仁", "红花", "延胡索"]
        return ToolResult(
            "formula_composition_search",
            True,
            "候选方群组成",
            {"formula_name": "独活寄生汤加减候选", "herbs": base, "pattern": pattern, "requires_dose_evidence": True},
            is_stub=True,
        )

    # ------------------------------------------------------------ expert cases
    def similar_case_search(self, query: str, limit: int = 5) -> ToolResult:
        return ToolResult(
            "similar_case_search",
            True,
            "假名化相似病例检索",
            {"cases": self.case_store.search(query, limit), "privacy": "pseudonymized_structured_fields_dlp_checked"},
            evidence_level=EvidenceLevel.EXPERT_CASE.value,
        )

    def counterexample_case_search(self, query: str, limit: int = 3) -> ToolResult:
        cases = [
            c for c in self.case_store.search(query, limit * 3)
            if any(w in " ".join(map(str, c.values())) for w in ["加重", "无效", "未缓解", "复发"])
        ]
        return ToolResult(
            "counterexample_case_search",
            True,
            "假名化反例检索",
            {"cases": cases[:limit]},
            evidence_level=EvidenceLevel.EXPERT_CASE.value,
        )

    def expert_profile(self):
        """Lazily mine the expert practice profile from the loaded corpus.

        Guarded because consult members call this concurrently: without the lock
        five members racing a cold cache each mine the whole corpus, which is the
        most expensive computation in the registry.
        """
        with self._profile_lock:
            if self._profile is None:
                from .expert.profile import build_profile

                self._profile = build_profile(self.case_store.records)
            return self._profile

    def expert_practice_profile(self, pattern: str | None = None) -> ToolResult:
        """Aggregate expert habits — core herbs, treatments, investigations.

        This is what turns a case corpus into transferable experience: instead
        of five raw look-alike cases, an agent gets "in N cases of this pattern
        the expert used X in M of them, ordered Y, and Z cases worsened".
        """
        profile = self.expert_profile()
        if not profile.total_cases:
            return ToolResult(
                "expert_practice_profile", True, "未加载专家病例库",
                {"total_cases": 0, "patterns": [], "how_to_fix": "运行时传入 --xlsx 或 ToolRegistry(records=...)"},
                is_stub=True,
            )
        brief = profile.brief(pattern)
        matched = profile.pattern_for(pattern) if pattern else None
        return ToolResult(
            "expert_practice_profile",
            True,
            f"专家经验画像: {profile.total_cases}例/{len(profile.patterns)}证型"
            + (f"; 命中证型 {matched.pattern}({matched.n_cases}例)" if matched else ""),
            {**brief, "requested_pattern": pattern, "matched_pattern": matched.pattern if matched else None},
            evidence_level=EvidenceLevel.EXPERT_CASE.value,
        )

    def interview_axis_lookup(self, axis_id: str | None = None, tier: str | None = None) -> ToolResult:
        """Look up history-taking axes: what to ask and why it matters.

        Exposed as a tool rather than baked into a prompt so an autonomous agent
        can pull the professional rationale for an axis on demand, and so the
        lookup lands in the evidence ledger like any other retrieval. The axis
        table is authored content, not a placeholder, so it is graded
        ``guideline_or_standard`` rather than ``stub``.
        """
        from .interview.axes import AXES, AXES_BY_ID, TIERS

        if axis_id:
            axis = AXES_BY_ID.get(axis_id)
            if axis is None:
                return ToolResult(
                    "interview_axis_lookup", False, f"未知问诊轴 {axis_id!r}",
                    {"available": sorted(AXES_BY_ID)}, error="unknown_axis", recoverable=True,
                )
            return ToolResult(
                "interview_axis_lookup", True, f"问诊轴: {axis.label}({axis.tier})",
                {"axis": axis.to_dict()}, evidence_level=EvidenceLevel.GUIDELINE.value,
            )

        if tier:
            if tier not in TIERS:
                return ToolResult(
                    "interview_axis_lookup", False, f"未知层级 {tier!r}",
                    {"available": list(TIERS)}, error="unknown_tier", recoverable=True,
                )
            selected = [a for a in AXES if a.tier == tier]
        else:
            selected = list(AXES)
        return ToolResult(
            "interview_axis_lookup", True,
            f"返回 {len(selected)} 条问诊轴" + (f"（{tier}）" if tier else "（十问歌+骨科专科全表）"),
            {
                "axes": [
                    {"axis_id": a.axis_id, "label": a.label, "tier": a.tier,
                     "tradition": a.tradition, "rationale": a.rationale,
                     "probes": list(a.probes)}
                    for a in selected
                ],
                "tiers": list(TIERS),
            },
            evidence_level=EvidenceLevel.GUIDELINE.value,
        )

    def medical_image_read(
        self,
        image: str,
        kind: str = "other",
        context: str = "",
        deidentified: bool = False,
    ) -> ToolResult:
        """Read one clinical image through the vision model — never diagnostically.

        Two refusals come before any reading happens:

        * **No attestation, no read.** The caller must assert the image is
          de-identified. That is a deliberate speed bump on the one input channel
          that most easily carries a name and a hospital number.
        * **Identifiers found, read discarded.** The vision client's PHI
          pre-check runs first; when it fires, the findings are thrown away and
          the warning is returned in their place.

        The result is graded ``model_reasoning``, which is in
        ``NON_RELEASABLE_LEVELS`` — so an image finding can raise a red flag and
        suggest an examination, but can never on its own justify a released claim.
        """
        from .vision.client import VisionError

        if self.vision is None or not getattr(self.vision, "available", False):
            return ToolResult(
                "medical_image_read", True, "未配置视觉模型，跳过影像判读",
                {"configured": False,
                 "how_to_fix": "设置 YAOBI_VISION_PROVIDER=poe、POE_API_KEY 与 YAOBI_VISION_MODEL=Gemini-3.1-Pro"},
                is_stub=True,
            )
        if not deidentified:
            return ToolResult(
                "medical_image_read", False,
                "缺少去标识化声明：请先遮盖姓名/ID/日期/条码/人脸，并显式声明 deidentified=true",
                {"required": "deidentified=true"}, error="deidentification_not_attested", recoverable=True,
            )
        try:
            read = self.vision.read(image, kind=kind, context=context)
        except VisionError as exc:
            return ToolResult(
                "medical_image_read", False, f"影像判读失败: {exc}",
                {"kind": kind}, error=str(exc), retryable=True,
            )

        payload = read.to_dict()
        if read.phi_detected:
            # Not a tool failure — the tool did exactly its job. It is returned
            # as a successful *refusal* so the run records why no findings exist.
            return ToolResult(
                "medical_image_read", True, "图片含可识别身份信息，已拒绝判读",
                payload, evidence_level=EvidenceLevel.MODEL.value,
            )
        summary = f"{read.image_kind} 视觉所见 {len(read.observations)} 条"
        if read.urgent_signals:
            summary += f"；急症外观信号 {len(read.urgent_signals)} 条"
        return ToolResult(
            "medical_image_read", True, summary + "（模型判读，不能替代正式阅片）",
            payload, evidence_level=EvidenceLevel.MODEL.value,
        )

    def patient_timeline_search(self, research_patient_id: str) -> ToolResult:
        visits = [r for r in self.case_store.records if r.get("research_patient_id") == research_patient_id]
        return ToolResult(
            "patient_timeline_search",
            True,
            "患者脱敏时间线",
            {"research_patient_id": research_patient_id, "visits": visits, "visit_count": len(visits)},
            evidence_level=EvidenceLevel.EXPERT_CASE.value,
        )

    def herb_dose_distribution(self, herbs: list[str], pattern: str | None = None, age: int | None = None) -> ToolResult:
        distributions = self.case_store.dose_distribution(
            herbs, pattern=pattern, age=age, authorized_ranges=self.authorized_ranges
        )
        return ToolResult(
            "herb_dose_distribution",
            True,
            "按证型/年龄/别名分层的专家剂量分布",
            {"distributions": distributions, "min_sample_n": MIN_DOSE_SAMPLE_N},
            evidence_level=EvidenceLevel.EXPERT_CASE.value,
        )

    # -------------------------------------------------------------- dose safety
    def pharmacopeia_check(self, herbs: list[str], doses: dict[str, float] | None = None) -> ToolResult:
        """Validate each herb *and its proposed dose* against authorised ranges.

        ``doses`` maps herb name to the gram value the harness intends to draft.
        When it is supplied, a dose outside the authorised range fails the check
        — the previous implementation only verified that a range existed, so an
        out-of-range dose could still be drafted.
        """
        doses = doses or {}
        checked: list[dict[str, Any]] = []
        for herb in herbs:
            rng, provenance = self._authorized_range(herb)
            proposed = doses.get(herb)
            entry: dict[str, Any] = {
                "herb": herb,
                "authorized_range_available": bool(rng),
                "range_g": list(rng) if rng else None,
                "range_source": provenance,
                "proposed_dose_g": proposed,
                "dose_checked": proposed is not None,
                "dose_within_range": None,
                "risk_flags": ["risk_herb_requires_special_review"] if herb in RISK_HERBS else [],
            }
            if rng and proposed is not None:
                entry["dose_within_range"] = bool(rng[0] <= float(proposed) <= rng[1])
                if not entry["dose_within_range"]:
                    entry["risk_flags"].append(
                        f"dose_out_of_authorized_range:{proposed}g_not_in_{rng[0]}-{rng[1]}g"
                    )
            entry["ok"] = bool(
                rng
                and herb not in RISK_HERBS
                and (proposed is None or entry["dose_within_range"] is True)
            )
            checked.append(entry)
        unverified = [c["herb"] for c in checked if c["dose_checked"] is False]
        from_store = [c for c in checked if (c.get("range_source") or {}).get("source_id")]
        return ToolResult(
            "pharmacopeia_check",
            True,
            "药典/正式范围与拟用剂量校验",
            {
                "version": self._pharmacopeia_version(),
                "checked": checked,
                "citations": [c["range_source"] for c in from_store],
                "doses_not_submitted_for_check": unverified,
                "pass": all(c["ok"] for c in checked) and not unverified and bool(checked),
            },
            # Only a licensed pharmacopoeia earns pharmacopoeia-grade evidence;
            # a locally configured range table is an ordinary tool result.
            evidence_level=EvidenceLevel.PHARMACOPEIA.value if from_store else EvidenceLevel.TOOL.value,
            is_stub=not from_store and not self.authorized_ranges,
        )

    def _authorized_range(self, substance: str) -> tuple[tuple[float, float] | None, dict[str, Any] | None]:
        """Prefer a licensed pharmacopoeia range over a locally configured one."""
        if self.knowledge is not None:
            entry = self.knowledge.dose_range(substance)
            if entry:
                return (float(entry["min_value"]), float(entry["max_value"])), {
                    **entry["provenance"], "basis": entry.get("basis", ""), "unit": entry.get("unit", "g"),
                }
        rng = self.authorized_ranges.get(substance)
        return (rng, {"source_id": "", "source": "locally_configured_ranges"} if rng else None)

    def _pharmacopeia_version(self) -> str:
        if self.knowledge is not None and self.knowledge.enabled_sources():
            return "knowledge_store:" + ",".join(self.knowledge.enabled_sources())
        return "authorized_ranges_stub" if self.authorized_ranges else "no_authorized_pharmacopeia_dataset"

    # ------------------------------------------------------------ medications
    def drug_interaction_check(
        self,
        medications: list[str],
        conditions: list[str] | None = None,
        include_herbs: list[str] | None = None,
    ) -> ToolResult:
        """Screen a medication list for orthopaedic drug-drug interactions.

        Combines the in-repo orthopaedic rule pack with any licensed
        interaction data in the knowledge store. Findings are returned most
        severe first, each with mechanism, management and its source.
        """
        meds = [str(m) for m in (medications or []) if str(m).strip()]
        rule_hits = ortho_interactions.evaluate(meds, conditions or [])
        store_hits = self.knowledge.interactions_for(meds) if self.knowledge is not None else []
        blocking = ortho_interactions.blocking(rule_hits) + [
            h for h in store_hits if h.get("severity") in ortho_interactions.BLOCKING_SEVERITIES
        ]
        herb_violations = incompat.check_combination(include_herbs or [])
        return ToolResult(
            "drug_interaction_check",
            True,
            f"药物相互作用筛查: {len(rule_hits)}条规则命中, {len(store_hits)}条数据库命中",
            {
                "medications": meds,
                "conditions": sorted({str(c) for c in (conditions or [])}),
                "rule_findings": rule_hits,
                "database_findings": store_hits,
                "herb_combination_violations": herb_violations,
                "blocking": blocking,
                "pass": not blocking and not herb_violations,
                "rule_pack": ortho_interactions.rule_pack_summary(),
                "coverage_note": (
                    "内置规则包只覆盖骨科高频高危组合；未接入授权 DDI 数据库时不能视为完整相互作用审查"
                    if not store_hits else "内置规则包 + 授权相互作用数据库"
                ),
            },
            evidence_level=EvidenceLevel.TOOL.value,
            is_stub=not store_hits and not rule_hits,
        )

    def drug_label_lookup(self, ingredient: str, sections: list[str] | None = None) -> ToolResult:
        """Return authorised label sections (interactions, contraindications, dosing)."""
        if self.knowledge is None:
            return ToolResult(
                "drug_label_lookup", True, "未配置说明书知识库",
                {"ingredient": ingredient, "sections": [], "how_to_fix": "运行 knowledge build 摄取 openFDA/DailyMed/NMPA 说明书"},
                is_stub=True,
            )
        from .knowledge.store import LABEL_SECTIONS

        found = self.knowledge.label_sections(ingredient, sections or LABEL_SECTIONS)
        return ToolResult(
            "drug_label_lookup",
            True,
            f"{ingredient}: 命中{len(found)}个说明书章节",
            {
                "ingredient": ingredient,
                "sections": found,
                "citations": [f["provenance"] for f in found],
            },
            evidence_level=EvidenceLevel.PHARMACOPEIA.value if found else EvidenceLevel.TOOL.value,
            is_stub=not found,
        )

    def drug_normalize(self, name: str) -> ToolResult:
        """Resolve a free-text medication name to RxCUI + ATC classes.

        Requires a configured RxNorm connector; without one the harness falls
        back to the rule pack's own bilingual class matching, which needs no
        network access.
        """
        classes = ortho_interactions.classify(name)
        if self.drug_normalizer is None:
            return ToolResult(
                "drug_normalize", True, f"{name}: 本地类别匹配",
                {"input": name, "rxcui": None, "atc": [], "local_classes": classes, "mode": "offline_class_match"},
            )
        resolved = self.drug_normalizer.normalize(name)
        return ToolResult(
            "drug_normalize", True, f"{name}: RxNorm 标准化",
            {**resolved, "local_classes": classes, "mode": "rxnorm"},
        )

    def interaction_check(
        self,
        herbs: list[str],
        medications: list[str] | None = None,
        allergies: list[str] | None = None,
        medications_confirmed: bool = False,
        allergies_confirmed: bool = False,
        pregnancy: bool | None = None,
    ) -> ToolResult:
        """Herb-herb, herb-drug, allergy and pregnancy screening.

        Herb-herb screening applies 十八反/十九畏, which the previous version
        omitted entirely — the most basic combination safety rule in Chinese
        herbal prescribing.
        """
        flags: list[str] = []
        meds = medications or []
        allergy_list = allergies or []

        if not medications_confirmed:
            flags.append("current_medications_unknown")
        if not allergies_confirmed:
            flags.append("allergies_unknown")
        if any("抗凝" in m or "华法林" in m or "阿司匹林" in m or "氯吡格雷" in m for m in meds):
            flags.append("活血药与抗凝/抗血小板药需专项审查")
        for herb in herbs:
            if herb in allergy_list:
                flags.append(f"过敏史包含{herb}")

        combination_violations = incompat.check_combination(herbs)
        flags += [v["detail"] for v in combination_violations]
        pregnancy_violations = incompat.check_pregnancy(herbs) if pregnancy else []
        flags += [v["detail"] for v in pregnancy_violations]

        return ToolResult(
            "interaction_check",
            True,
            "配伍禁忌/相互作用/过敏/妊娠筛查",
            {
                "risk_flags": flags,
                "combination_violations": combination_violations,
                "pregnancy_violations": pregnancy_violations,
                "pass": not flags,
            },
        )

    def special_population_check(
        self,
        pregnancy: bool | None = None,
        age: int | None = None,
        renal: str | None = None,
        liver: str | None = None,
        **_ignored: Any,
    ) -> ToolResult:
        missing = []
        if pregnancy is None:
            missing.append("pregnancy")
        if age is None:
            missing.append("age")
        if renal in (None, "unknown", "未知", ""):
            missing.append("renal")
        if liver in (None, "unknown", "未知", ""):
            missing.append("liver")
        flags = []
        if pregnancy is True:
            flags.append("pregnancy_requires_no_remote_herbal_draft")
        if isinstance(age, int) and (age < 18 or age >= 75):
            flags.append("age_requires_special_dose_review")
        return ToolResult(
            "special_population_check",
            True,
            "特殊人群信息审查",
            {"missing": missing, "risk_flags": flags, "pass": not missing and not flags},
        )

    def physician_review_submit(
        self,
        prescription: dict[str, Any],
        approvals: dict[str, bool],
        physician_id: str | None = None,
        signature: str | None = None,
    ) -> ToolResult:
        problems: list[str] = []
        herbs = prescription.get("herbs") or []
        if not physician_id or not signature:
            problems.append("missing_physician_identity_or_signature")
        if not prescription.get("prescription_hash"):
            problems.append("missing_prescription_hash")
        for herb in herbs:
            name = herb.get("herb_name")
            for required in ("dose_value", "dose_unit", "processing", "dose_evidence_ids"):
                if not herb.get(required):
                    problems.append(f"{name}:missing_{required}")
            if name in RISK_HERBS and not herb.get("special_review"):
                problems.append(f"{name}:risk_herb_special_review_missing")
            if not approvals.get(name):
                problems.append(f"{name}:not_approved")
        violations = incompat.check_combination([h.get("herb_name", "") for h in herbs])
        problems += [f"combination:{v['detail']}" for v in violations]
        ok = bool(herbs) and not problems
        return ToolResult(
            "physician_review_submit",
            ok,
            "医师逐味审核通过" if ok else "医师审核未通过",
            {"approved": ok, "problems": problems},
            error=None if ok else "review_incomplete",
        )


class ExpertCaseStore:
    """De-identified, pseudonymised view over the expert's case records."""

    def __init__(self, xlsx_path: str | Path, deid_key: str | None = None) -> None:
        self.path = Path(xlsx_path)
        self.deid_key = resolve_deid_key(deid_key)
        self.records = self._load_xlsx_structured(self.path)

    @classmethod
    def empty(cls) -> "ExpertCaseStore":
        store = cls.__new__(cls)
        store.path = None
        store.deid_key = None
        store.records = []
        return store

    @classmethod
    def from_records(cls, records: list[dict[str, Any]], deid_key: str | None = None) -> "ExpertCaseStore":
        store = cls.empty()
        store.deid_key = resolve_deid_key(deid_key)
        store.records = [store._deidentify_record(r) for r in records]
        return store

    def _research_id(self, source: str) -> str:
        """Stable HMAC pseudonym; the key is resolved once per store."""
        return "YP" + hmac.new(self.deid_key.encode(), source.encode(), hashlib.sha256).hexdigest()[:12]

    def _deidentify_record(self, row: dict[str, Any]) -> dict[str, Any]:
        source = str(row.get("病案号") or row.get("case_id") or row.get("就诊序号") or sorted(row.items()))
        record: dict[str, Any] = {}
        for key, value in row.items():
            if key in DIRECT_IDENTIFIER_FIELDS or value in (None, ""):
                continue
            field_name = "就诊月份" if key in DATE_FIELDS else key
            if field_name in AUTHORIZED_FIELDS:
                record[field_name] = sanitize_text(generalize_date(value) if key in DATE_FIELDS else value)
        record["research_patient_id"] = self._research_id(source)
        record["herbs"] = parse_herbs(str(row.get("中药", "")))
        record["text_index"] = " ".join(
            str(record.get(k, ""))
            for k in ["性别", "年龄", "主诉", "现病史", "既往史", "中医四诊", "中医诊断", "西医诊断", "治疗方法", "治疗"]
        )
        return record

    def _load_xlsx_structured(self, path: Path) -> list[dict[str, Any]]:
        rows = self._xlsx_rows(path)
        if not rows:
            return []
        headers = [str(x).strip() for x in rows[0]]
        return [
            self._deidentify_record({headers[i]: r[i] if i < len(r) else "" for i in range(len(headers))})
            for r in rows[1:]
            if any(r)
        ]

    def _xlsx_rows(self, path: Path) -> list[list[str]]:
        with zipfile.ZipFile(path) as archive:
            shared: list[str] = []
            ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                for si in root.findall("a:si", ns):
                    shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
            sheet = next(n for n in archive.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
            root = ET.fromstring(archive.read(sheet))
            out: list[list[str]] = []
            for row in root.findall(".//a:row", ns):
                cells: dict[int, str] = {}
                for cell in row.findall("a:c", ns):
                    ref = cell.attrib.get("r", "")
                    idx = col_to_index(re.sub(r"\d+", "", ref))
                    value = cell.find("a:v", ns)
                    text = "" if value is None else value.text or ""
                    if cell.attrib.get("t") == "s" and text.isdigit() and int(text) < len(shared):
                        text = shared[int(text)]
                    cells[idx] = text
                out.append([cells.get(i, "") for i in range(max(cells.keys(), default=-1) + 1)])
            return out

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        """Tokens plus CJK bigrams.

        Chinese has no whitespace, so whole-phrase tokens such as ``腰痛3月``
        almost never occur verbatim in a record. Bigrams give the retrieval a
        usable recall floor without pulling in a segmentation dependency.
        """
        tokens = [t for t in re.split(r"\W+", query or "") if t]
        cjk = re.sub(r"[^一-鿿]", "", query or "")
        bigrams = [cjk[i : i + 2] for i in range(len(cjk) - 1)]
        return tokens + bigrams

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        terms = self._query_terms(query)
        scored = []
        for record in self.records:
            score = sum(str(record.get("text_index", "")).count(t) for t in terms)
            if score:
                scored.append((score, record))
        return [
            {k: v for k, v in record.items() if k not in DIRECT_IDENTIFIER_FIELDS and k not in ("text_index", "中药")}
            for _, record in sorted(scored, key=lambda x: -x[0])[:limit]
        ]

    def dose_distribution(
        self,
        herbs: list[str],
        pattern: str | None = None,
        age: int | None = None,
        authorized_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, Any]:
        """Stratified dose statistics per herb.

        ``outlier_flag`` is judged against the authorised range when one exists,
        so a 30 g median for a 3–9 g herb is flagged instead of sliding under a
        single global ceiling.
        """
        ranges = authorized_ranges or {}
        out: dict[str, Any] = {}
        for herb in herbs:
            aliases = set(HERB_ALIASES.get(herb, [herb]))
            doses: list[float] = []
            for record in self.records:
                if pattern and pattern not in str(record.get("中医诊断", "")):
                    continue
                record_age = str(record.get("年龄", "")).rstrip("岁")
                if age and record_age.isdigit() and abs(int(record_age) - age) > 15:
                    continue
                doses += [x["dose_g"] for x in record.get("herbs", []) if x["herb_name"] in aliases]

            median = statistics.median(doses) if doses else None
            rng = ranges.get(herb)
            if not doses:
                outlier = False
            elif rng:
                outlier = bool(max(doses) > rng[1] * 1.5 or min(doses) <= 0 or (median is not None and not rng[0] <= median <= rng[1]))
            else:
                outlier = bool(max(doses) > ABSOLUTE_DOSE_CEILING_G or min(doses) <= 0)
            p25, p75 = percentile(doses, 0.25), percentile(doses, 0.75)
            dispersion = None
            if p25 and p75 and p25 > 0:
                dispersion = round((p75 - p25) / p25, 3)
            out[herb] = {
                "n": len(doses),
                "median_g": median,
                "p25_g": p25,
                "p75_g": p75,
                "iqr_ratio": dispersion,
                "evidence": "stratified_expert_case_distribution" if doses else "missing",
                "meets_min_n": len(doses) >= MIN_DOSE_SAMPLE_N,
                "outlier_flag": outlier,
                "authorized_range_g": list(rng) if rng else None,
                "wide_dispersion": bool(dispersion is not None and dispersion > 1.0),
            }
        return out


# ------------------------------------------------------------------ LLM schemas

def tool_specs() -> list[ToolSpec]:
    """OpenAI-style schemas so a planner LLM can reason about the tool surface.

    These are descriptions only. Selecting a tool never bypasses the broker —
    an LLM-proposed call is authorised exactly like a rule-based one.
    """
    herb_array = {"type": "array", "items": {"type": "string"}, "description": "中药名列表"}
    return [
        ToolSpec("red_flag_evidence_search", "对主诉做急症红旗筛查，返回硬信号/弱信号/被抑制项",
                 {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
        ToolSpec("emergency_resource_lookup", "查询当地急救资源与急救电话",
                 {"type": "object", "properties": {"location": {"type": "string"}}}),
        ToolSpec("clinical_guideline_search", "检索指南要点（当前为占位数据源，证据等级为 stub）",
                 {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}),
        ToolSpec("tcm_pattern_knowledge_search", "根据症状文本匹配候选证型",
                 {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
        ToolSpec("similar_case_search", "检索假名化的相似专家病例",
                 {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}),
        ToolSpec("counterexample_case_search", "检索疗效不佳/加重/复发的反例病例",
                 {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}),
        ToolSpec("patient_timeline_search", "按研究假名ID检索同一患者的历次就诊",
                 {"type": "object", "properties": {"research_patient_id": {"type": "string"}}, "required": ["research_patient_id"]}),
        ToolSpec("expert_practice_profile", "查询该专家在某证型下的核心用药、治法、常做检查与随访倾向（聚合统计，不含个体文本）",
                 {"type": "object", "properties": {"pattern": {"type": "string", "description": "证型名，留空则返回语料概览"}}}),
        ToolSpec("formula_composition_search", "按证型检索候选方组成（仅医师角色、非急症）",
                 {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}),
        ToolSpec("herb_dose_distribution", "按证型/年龄分层的专家剂量分布（仅医师角色、非急症）",
                 {"type": "object", "properties": {"herbs": herb_array, "pattern": {"type": "string"}, "age": {"type": "integer"}}, "required": ["herbs"]}),
        ToolSpec("pharmacopeia_check", "校验药材与拟用剂量是否落在授权药典范围内",
                 {"type": "object", "properties": {"herbs": herb_array, "doses": {"type": "object", "description": "药名->拟用克数"}}, "required": ["herbs"]}),
        ToolSpec("interaction_check", "十八反/十九畏、中西药相互作用、过敏与妊娠禁忌筛查",
                 {"type": "object", "properties": {"herbs": herb_array, "medications": {"type": "array", "items": {"type": "string"}},
                                                    "allergies": {"type": "array", "items": {"type": "string"}},
                                                    "medications_confirmed": {"type": "boolean"},
                                                    "allergies_confirmed": {"type": "boolean"},
                                                    "pregnancy": {"type": "boolean"}}, "required": ["herbs"]}),
        ToolSpec("special_population_check", "妊娠/年龄/肝肾功能等特殊人群信息完备性与风险审查",
                 {"type": "object", "properties": {"pregnancy": {"type": "boolean"}, "age": {"type": "integer"},
                                                    "renal": {"type": "string"}, "liver": {"type": "string"}}}),
        ToolSpec("drug_interaction_check", "骨科药物相互作用筛查（内置规则包 + 授权 DDI 库）",
                 {"type": "object", "properties": {"medications": {"type": "array", "items": {"type": "string"}},
                                                    "conditions": {"type": "array", "items": {"type": "string"},
                                                                   "description": "如 renal_impairment, planned_neuraxial_anesthesia"},
                                                    "include_herbs": herb_array}, "required": ["medications"]}),
        ToolSpec("drug_label_lookup", "查询授权说明书章节（相互作用/禁忌/剂量/特殊人群）",
                 {"type": "object", "properties": {"ingredient": {"type": "string"},
                                                    "sections": {"type": "array", "items": {"type": "string"}}},
                  "required": ["ingredient"]}),
        ToolSpec("drug_normalize", "将自由文本药名标准化为 RxCUI 与 ATC 分类",
                 {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}),
        ToolSpec("interview_axis_lookup", "查询问诊轴（十问歌 + 骨科专科）：该轴要问什么、为什么问、属于哪个层级",
                 {"type": "object", "properties": {
                     "axis_id": {"type": "string", "description": "指定轴 id；留空则返回全表"},
                     "tier": {"type": "string", "enum": ["RED_FLAG", "CORE", "SPECIALTY", "TCM", "CONTEXT"],
                              "description": "按层级筛选"}}}),
        ToolSpec("medical_image_read",
                 "判读一张临床图片（X线/MRI-CT 翻拍、舌象、体态、肢体外观、报告单）。"
                 "结果为模型视觉所见，证据等级为 model_reasoning，**不能替代正式阅片、不能作为诊断依据**。"
                 "调用前必须确认图片已去标识化。",
                 {"type": "object", "properties": {
                     "image": {"type": "string", "description": "本地文件路径或 data: URI"},
                     "kind": {"type": "string",
                              "enum": ["radiograph", "mri_ct", "tongue", "posture_gait",
                                       "limb_surface", "report_document", "other"]},
                     "context": {"type": "string", "description": "临床背景，帮助模型聚焦"},
                     "deidentified": {"type": "boolean",
                                      "description": "必须为 true：声明已遮盖姓名/ID/日期/条码/人脸"}},
                  "required": ["image", "deidentified"]}),
        ToolSpec("physician_review_submit", "提交处方草案给医师逐味审核签名",
                 {"type": "object", "properties": {"prescription": {"type": "object"}, "approvals": {"type": "object"},
                                                    "physician_id": {"type": "string"}, "signature": {"type": "string"}},
                  "required": ["prescription", "approvals"]}),
    ]


TOOL_NAMES = {spec.name for spec in tool_specs()}


# ---------------------------------------------------------------- module utils

def resolve_deid_key(explicit: str | None = None) -> str:
    """Return the pseudonymisation key, or fail loudly.

    Set ``YAOBI_DEID_KEY`` (or pass ``deid_key=``) so the same patient always
    maps to the same research ID. ``YAOBI_ALLOW_EPHEMERAL_DEID_KEY=1`` opts into
    a throwaway key for one-off local inspection only.
    """
    key = explicit or os.environ.get("YAOBI_DEID_KEY")
    if key:
        return key
    if os.environ.get("YAOBI_ALLOW_EPHEMERAL_DEID_KEY") == "1":
        return "ephemeral:" + os.urandom(16).hex()
    raise DeidentificationKeyError(
        "缺少稳定假名化密钥：请设置 YAOBI_DEID_KEY 或传入 deid_key。"
        "随机密钥会让同一患者每次得到不同的 research_patient_id，纵向检索将静默失效。"
    )


def parse_herbs(text: str) -> list[dict[str, Any]]:
    herbs = []
    for name, dose, method in HERB_ITEM_RE.findall(text):
        clean = name.strip().lstrip("*").strip()
        herbs.append(
            {
                "herb_name": clean,
                "dose_g": float(dose),
                "administration": method.strip() or "未注明",
                "risk_flags": ["risk_herb"] if clean in RISK_HERBS else [],
            }
        )
    return herbs


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] if lo == hi else values[lo] * (hi - idx) + values[hi] * (idx - lo)


def col_to_index(col: str) -> int:
    n = 0
    for ch in col:
        n = n * 26 + ord(ch.upper()) - 64
    return n - 1


def sanitize_text(value: Any) -> str:
    text = str(value)
    for pattern in DLP_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def generalize_date(value: Any) -> str:
    text = str(value)
    match = re.match(r"(20\d{2})[-/.年](\d{1,2})", text)
    return f"{match.group(1)}-{int(match.group(2)):02d}" if match else "date_generalized"


def is_current_patient_symptom(text: str, term: str) -> bool:
    """Backwards-compatible wrapper over the clause-scoped screener."""
    result = red_flags.screen(text)
    return any(hit.term == term for hit in result.hits) or any(hit.term == term for hit in result.soft_hits)


def herbs_in(items: Iterable[Any]) -> list[str]:
    """Normalise a formula herb list that may contain strings or dicts."""
    out = []
    for item in items or []:
        out.append(item if isinstance(item, str) else str(item.get("herb_name", "")))
    return [h for h in out if h]
