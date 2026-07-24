from __future__ import annotations

import re, statistics, zipfile, xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

HERB_RE = re.compile(r"([\u4e00-\u9fff]{1,8})\s*(\d+(?:\.\d+)?)\s*(g|克|G)")
RED_FLAGS = {
    "cauda_equina": ["尿潴留", "大小便失禁", "会阴麻木", "鞍区麻木"],
    "infection_or_tumor": ["发热", "夜间痛", "体重下降", "肿瘤", "癌"],
    "fracture": ["外伤", "跌倒", "骨质疏松"],
    "progressive_neuro": ["肌力下降", "足下垂", "进行性麻木"],
}

@dataclass
class ToolResult:
    tool: str
    ok: bool
    summary: str
    data: dict[str, Any]
    evidence_level: str = "tool_result"

class CapabilityBroker:
    def __init__(self, role: str, risk_mode: str, prescription_approved: bool = False):
        self.role = role
        self.risk_mode = risk_mode
        self.prescription_approved = prescription_approved
        self.health: dict[str, bool] = defaultdict(lambda: True)

    def allow(self, tool: str) -> tuple[bool, str]:
        urgent_forbidden = {"formula_composition_search", "herb_dose_distribution", "physician_review_submit"}
        patient_forbidden = {"herb_dose_distribution", "physician_review_submit"}
        if not self.health[tool]:
            return False, "tool_unhealthy"
        if self.risk_mode == "urgent" and tool in urgent_forbidden:
            return False, "urgent_mode_forbids_prescription_or_dose"
        if self.role == "patient" and tool in patient_forbidden:
            return False, "patient_role_forbids_dose_or_review_tools"
        return True, "allowed"

class ToolRegistry:
    def __init__(self, xlsx_path: str | Path | None = None):
        self.case_store = ExpertCaseStore(xlsx_path) if xlsx_path else ExpertCaseStore.empty()
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
        }

    def call(self, broker: CapabilityBroker, name: str, **kwargs: Any) -> ToolResult:
        ok, reason = broker.allow(name)
        if not ok:
            return ToolResult(name, False, reason, {"denied_reason": reason})
        if name not in self.tools:
            return ToolResult(name, False, "unknown_tool", {})
        return self.tools[name](**kwargs)

    def red_flag_evidence_search(self, text: str) -> ToolResult:
        hits=[]
        for k, words in RED_FLAGS.items():
            for w in words:
                if w in text:
                    hits.append({"signal": k, "term": w, "source": "structured_red_flag_screen"})
        return ToolResult("red_flag_evidence_search", True, f"发现{len(hits)}个风险信号", {"hits": hits})

    def similar_case_search(self, query: str, limit: int = 5) -> ToolResult:
        return ToolResult("similar_case_search", True, "相似专家病例检索", {"cases": self.case_store.search(query, limit)})

    def counterexample_case_search(self, query: str, limit: int = 3) -> ToolResult:
        cases = [c for c in self.case_store.search(query, limit * 2) if "加重" in c.get("text", "") or "无效" in c.get("text", "")][:limit]
        return ToolResult("counterexample_case_search", True, "反例/冲突病例检索", {"cases": cases})

    def herb_dose_distribution(self, herbs: list[str]) -> ToolResult:
        return ToolResult("herb_dose_distribution", True, "专家病例剂量分布", {"distributions": self.case_store.dose_distribution(herbs)})

    def tcm_pattern_knowledge_search(self, text: str) -> ToolResult:
        patterns=[]
        if any(x in text for x in ["刺痛", "固定", "麻木", "久坐"]): patterns.append("气滞血瘀证")
        if any(x in text for x in ["乏力", "酸软", "久病", "劳累"]): patterns.append("气血痹阻证")
        if any(x in text for x in ["冷痛", "畏寒"]): patterns.append("寒湿痹阻证")
        return ToolResult("tcm_pattern_knowledge_search", True, "证候知识匹配", {"patterns": patterns or ["待辨证"], "limits": "规则提示仅作证据检索入口"})

    def clinical_guideline_search(self, topic: str) -> ToolResult:
        return ToolResult("clinical_guideline_search", True, "腰痛指南要点", {"points": ["先筛查马尾综合征、感染、肿瘤、骨折、进行性神经缺损等红旗", "非特异性腰痛优先教育、保持活动、分层康复；神经根症状或红旗需线下评估影像/专科"]})

    def pharmacopeia_check(self, herbs: list[str]) -> ToolResult:
        return ToolResult("pharmacopeia_check", True, "药典/规范占位校验", {"version": "requires_authorized_pharmacopeia_dataset", "checked": [{"herb": h, "ok": True, "needs_authorized_range": True} for h in herbs]})

    def interaction_check(self, herbs: list[str], medications: list[str] | None = None) -> ToolResult:
        meds=medications or []
        flags=[]
        if meds and any("抗凝" in m or "华法林" in m for m in meds): flags.append("活血药与抗凝/抗血小板药需医师专项审查")
        return ToolResult("interaction_check", True, "相互作用初筛", {"risk_flags": flags})

    def special_population_check(self, pregnancy: bool | None = None, age: int | None = None, renal: str | None = None, liver: str | None = None) -> ToolResult:
        missing=[k for k,v in {"pregnancy":pregnancy,"age":age,"renal":renal,"liver":liver}.items() if v is None]
        return ToolResult("special_population_check", True, "特殊人群信息审查", {"missing": missing, "pass": not missing})

    def emergency_resource_lookup(self, location: str = "中国大陆") -> ToolResult:
        phone = "120" if "中国" in location else "当地急救电话"
        return ToolResult("emergency_resource_lookup", True, "急救资源", {"emergency_phone": phone, "advice": "疑似马尾综合征/严重神经缺损时优先急救转运"})

    def formula_composition_search(self, pattern: str) -> ToolResult:
        base = ["独活", "桑寄生", "杜仲", "牛膝", "当归", "川芎", "白芍", "熟地黄", "党参", "茯苓", "甘草"]
        if "瘀" in pattern: base += ["桃仁", "红花", "延胡索"]
        return ToolResult("formula_composition_search", True, "候选方群组成", {"formula_name": "独活寄生汤加减候选", "herbs": base})

class ExpertCaseStore:
    def __init__(self, xlsx_path: str | Path):
        self.path = Path(xlsx_path); self.records = self._load_xlsx_text(self.path)
    @classmethod
    def empty(cls):
        o=cls.__new__(cls); o.path=None; o.records=[]; return o
    def _load_xlsx_text(self, path: Path) -> list[dict[str, Any]]:
        # dependency-free extraction of visible cell text from first sheet
        try:
            with zipfile.ZipFile(path) as z:
                shared=[]
                if "xl/sharedStrings.xml" in z.namelist():
                    root=ET.fromstring(z.read("xl/sharedStrings.xml")); ns={"a":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                    for si in root.findall("a:si", ns): shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
                sheet=next(n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
                root=ET.fromstring(z.read(sheet)); ns={"a":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                rows=[]
                for r in root.findall(".//a:row", ns):
                    vals=[]
                    for c in r.findall("a:c", ns):
                        v=c.find("a:v", ns); txt="" if v is None else v.text or ""
                        if c.attrib.get("t")=="s" and txt.isdigit() and int(txt)<len(shared): txt=shared[int(txt)]
                        vals.append(txt)
                    if vals: rows.append({"case_id": f"X{len(rows)+1:04d}", "text": " | ".join(vals)})
                return rows
        except Exception:
            return []
    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        terms=[t for t in re.split(r"\W+", query) if t]
        scored=[]
        for r in self.records:
            score=sum(r["text"].count(t) for t in terms)
            if score: scored.append((score,r))
        return [r for _,r in sorted(scored, key=lambda x:-x[0])[:limit]]
    def dose_distribution(self, herbs: list[str]) -> dict[str, Any]:
        vals=defaultdict(list)
        for r in self.records:
            for herb,dose,unit in HERB_RE.findall(r["text"]):
                if herb in herbs: vals[herb].append(float(dose))
        out={}
        for h in herbs:
            ds=vals.get(h,[])
            out[h]={"n":len(ds), "median_g": statistics.median(ds) if ds else None, "evidence":"expert_case_distribution" if ds else "missing"}
        return out
