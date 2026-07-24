from __future__ import annotations

import hashlib, re, statistics, zipfile, xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DIRECT_IDENTIFIER_FIELDS = {"姓名", "病案号", "地址", "医师工号", "医师姓名", "科室代码"}
AUTHORIZED_FIELDS = {"就诊序号", "性别", "年龄", "就诊日期", "主诉", "现病史", "既往史", "过敏史", "手术史", "中医四诊", "辅助检查", "中医诊断", "西医诊断", "治疗方法", "西药", "中药", "治疗"}
HERB_ITEM_RE = re.compile(r"(?:^|[,，\n])\s*\d+\s*/\s*\*?\s*([^*/，,\n]+?)\s*\*\s*1\s*(?:克|g|G)\s*/\s*(\d+(?:\.\d+)?)\s*(?:克|g|G)\s*/\s*用法[:：]\s*([^/，,\n]*)")
RED_FLAGS = {
    "cauda_equina": ["尿潴留", "大小便失禁", "会阴麻木", "鞍区麻木"],
    "cardiopulmonary": ["胸痛", "大汗", "呼吸困难", "晕厥"],
    "infection_or_tumor": ["发热", "夜间痛", "体重下降", "肿瘤", "癌"],
    "fracture": ["外伤", "跌倒", "骨质疏松", "骨折"],
    "progressive_neuro": ["肌力下降", "足下垂", "进行性麻木"],
}
RISK_HERBS = {"附片", "制川乌", "川乌", "草乌", "细辛", "麻黄", "全蝎", "蜈蚣"}

@dataclass
class ToolResult:
    tool: str
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    evidence_level: str = "tool_result"
    error: str | None = None
    source_version: str | None = None

class CapabilityBroker:
    def __init__(self, role: str, risk_mode: str, budget: Any | None = None, prescription_approved: bool = False):
        self.role = role; self.risk_mode = risk_mode; self.budget = budget; self.prescription_approved = prescription_approved
        self.health: dict[str, bool] = defaultdict(lambda: True)

    def allow(self, tool: str) -> tuple[bool, str]:
        urgent_forbidden = {"formula_composition_search", "herb_dose_distribution", "physician_review_submit"}
        patient_forbidden = {"formula_composition_search", "herb_dose_distribution", "physician_review_submit"}
        if self.budget is not None and not self.budget.reserve_tool():
            return False, "budget_exhausted"
        if not self.health[tool]:
            return False, "tool_unhealthy"
        if self.risk_mode == "urgent" and tool in urgent_forbidden:
            return False, "urgent_mode_forbids_prescription_or_dose"
        if self.role == "patient" and tool in patient_forbidden:
            return False, "patient_role_forbids_formula_or_dose_tools"
        return True, "allowed"

class ToolRegistry:
    critical_tools = {"red_flag_evidence_search", "clinical_guideline_search", "emergency_resource_lookup", "special_population_check", "interaction_check", "pharmacopeia_check"}
    def __init__(self, xlsx_path: str | Path | None = None, records: list[dict[str, Any]] | None = None, failing_tools: set[str] | None = None):
        self.case_store = ExpertCaseStore.from_records(records) if records is not None else (ExpertCaseStore(xlsx_path) if xlsx_path else ExpertCaseStore.empty())
        self.failing_tools = failing_tools or set()
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
        }

    def call(self, broker: CapabilityBroker, name: str, **kwargs: Any) -> ToolResult:
        ok, reason = broker.allow(name)
        if not ok: return ToolResult(name, False, reason, {"denied_reason": reason}, error=reason)
        if name in self.failing_tools: return ToolResult(name, False, "tool_failure", error="injected_tool_failure")
        if name not in self.tools: return ToolResult(name, False, "unknown_tool", error="unknown_tool")
        try: return self.tools[name](**kwargs)
        except Exception as exc: return ToolResult(name, False, "tool_exception", error=repr(exc))

    def red_flag_evidence_search(self, text: str) -> ToolResult:
        hits=[]
        for kind, terms in RED_FLAGS.items():
            for term in terms:
                if term in text and not is_negated(text, term):
                    hits.append({"signal": kind, "term": term, "source": "structured_red_flag_screen"})
        return ToolResult("red_flag_evidence_search", True, f"发现{len(hits)}个非否定风险信号", {"hits": hits}, source_version="red_flags.v2")

    def similar_case_search(self, query: str, limit: int = 5) -> ToolResult:
        return ToolResult("similar_case_search", True, "脱敏相似专家病例检索", {"cases": self.case_store.search(query, limit), "privacy": "deidentified_structured_fields_only"}, "expert_case")

    def counterexample_case_search(self, query: str, limit: int = 3) -> ToolResult:
        cases=[c for c in self.case_store.search(query, limit * 3) if any(w in " ".join(map(str,c.values())) for w in ["加重", "无效", "未缓解", "复发"])]
        return ToolResult("counterexample_case_search", True, "脱敏反例/冲突病例检索", {"cases": cases[:limit]}, "expert_case")

    def herb_dose_distribution(self, herbs: list[str], pattern: str | None = None, age: int | None = None) -> ToolResult:
        return ToolResult("herb_dose_distribution", True, "按证型/年龄分层的专家病例剂量分布", {"distributions": self.case_store.dose_distribution(herbs, pattern=pattern, age=age)}, "expert_case")

    def tcm_pattern_knowledge_search(self, text: str) -> ToolResult:
        patterns=[]
        if any(x in text for x in ["刺痛", "固定", "麻木", "久坐"]): patterns.append("气滞血瘀证")
        if any(x in text for x in ["乏力", "酸软", "久病", "劳累"]): patterns.append("气血痹阻证")
        if any(x in text for x in ["冷痛", "畏寒", "怕冷"]): patterns.append("寒湿痹阻证")
        return ToolResult("tcm_pattern_knowledge_search", True, "证候知识匹配", {"patterns": patterns or ["待辨证"], "limits": "规则提示仅作证据检索入口"})

    def clinical_guideline_search(self, topic: str) -> ToolResult:
        return ToolResult("clinical_guideline_search", True, "腰痛指南要点", {"guideline_id": "low_back_pain.red_flags.local_stub", "points": ["先筛查马尾综合征、感染、肿瘤、骨折、进行性神经缺损及非腰痛急症", "红旗或持续/进展神经根症状需线下评估，影像由临床医师按适应证选择"]}, "guideline_or_standard", source_version="stub-guideline-v1")

    def pharmacopeia_check(self, herbs: list[str]) -> ToolResult:
        checked=[]
        for h in herbs:
            checked.append({"herb": h, "ok": h not in RISK_HERBS, "risk_flags": ["risk_herb_requires_special_review"] if h in RISK_HERBS else [], "authorized_range_available": False})
        return ToolResult("pharmacopeia_check", True, "药典授权数据未接入，仅做风险药阻断", {"version": "no_authorized_pharmacopeia_dataset", "checked": checked}, "tool_result")

    def interaction_check(self, herbs: list[str], medications: list[str] | None = None, allergies: list[str] | None = None) -> ToolResult:
        flags=[]; meds=medications or []; allergies=allergies or []
        if any("抗凝" in m or "华法林" in m for m in meds): flags.append("活血药与抗凝/抗血小板药需医师专项审查")
        for h in herbs:
            if h in allergies: flags.append(f"过敏史包含{h}")
        return ToolResult("interaction_check", True, "相互作用/过敏初筛", {"risk_flags": flags, "pass": not flags})

    def special_population_check(self, pregnancy: bool | None = None, age: int | None = None, renal: str | None = None, liver: str | None = None) -> ToolResult:
        missing=[]
        if pregnancy is None: missing.append("pregnancy")
        if age is None: missing.append("age")
        if renal in (None, "unknown", "未知", ""): missing.append("renal")
        if liver in (None, "unknown", "未知", ""): missing.append("liver")
        flags=[]
        if pregnancy is True: flags.append("pregnancy_requires_no_remote_herbal_draft")
        if isinstance(age, int) and (age < 18 or age >= 75): flags.append("age_requires_special_dose_review")
        return ToolResult("special_population_check", True, "特殊人群信息审查", {"missing": missing, "risk_flags": flags, "pass": not missing and not flags})

    def emergency_resource_lookup(self, location: str = "中国大陆") -> ToolResult:
        phone = "120" if "中国" in location else "当地急救电话"
        return ToolResult("emergency_resource_lookup", True, "急救资源", {"emergency_phone": phone, "advice": "疑似马尾综合征/严重神经缺损/胸痛呼吸困难时优先急救转运"})

    def formula_composition_search(self, pattern: str) -> ToolResult:
        base=["独活","桑寄生","杜仲","牛膝","当归","川芎","白芍","熟地黄","党参","茯苓","甘草"]
        if "瘀" in pattern: base += ["桃仁", "红花", "延胡索"]
        return ToolResult("formula_composition_search", True, "候选方群组成", {"formula_name":"独活寄生汤加减候选","herbs":base, "requires_dose_evidence": True})

    def physician_review_submit(self, prescription: dict[str, Any], approvals: dict[str, bool]) -> ToolResult:
        herbs=prescription.get("herbs", [])
        all_ok=bool(herbs) and all(approvals.get(h.get("herb_name")) for h in herbs)
        return ToolResult("physician_review_submit", all_ok, "医师逐味审核" if all_ok else "医师审核未完成", {"approved": all_ok, "pending": [h.get("herb_name") for h in herbs if not approvals.get(h.get("herb_name"))]})

class ExpertCaseStore:
    def __init__(self, xlsx_path: str | Path):
        self.path=Path(xlsx_path); self.records=self._load_xlsx_structured(self.path)
    @classmethod
    def empty(cls): o=cls.__new__(cls); o.path=None; o.records=[]; return o
    @classmethod
    def from_records(cls, records: list[dict[str, Any]]): o=cls.empty(); o.records=[cls._deidentify_record(r) for r in records]; return o
    @staticmethod
    def _research_id(source: str) -> str: return "YP" + hashlib.sha256(source.encode()).hexdigest()[:10]
    @classmethod
    def _deidentify_record(cls, row: dict[str, Any]) -> dict[str, Any]:
        source=str(row.get("病案号") or row.get("就诊序号") or row.get("case_id") or row)
        rec={k:v for k,v in row.items() if k in AUTHORIZED_FIELDS and k not in DIRECT_IDENTIFIER_FIELDS and v not in (None, "")}
        rec["research_patient_id"] = cls._research_id(source)
        rec["herbs"] = parse_herbs(str(row.get("中药", "")))
        rec["text_index"] = " ".join(str(rec.get(k,"")) for k in ["性别","年龄","主诉","现病史","既往史","中医四诊","中医诊断","西医诊断","治疗方法","治疗"])
        return rec
    def _load_xlsx_structured(self, path: Path) -> list[dict[str, Any]]:
        rows=self._xlsx_rows(path)
        if not rows: return []
        headers=[str(x).strip() for x in rows[0]]
        return [self._deidentify_record(dict(zip(headers, r))) for r in rows[1:] if any(r)]
    def _xlsx_rows(self, path: Path) -> list[list[str]]:
        with zipfile.ZipFile(path) as z:
            shared=[]
            if "xl/sharedStrings.xml" in z.namelist():
                root=ET.fromstring(z.read("xl/sharedStrings.xml")); ns={"a":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for si in root.findall("a:si", ns): shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
            sheet=next(n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
            root=ET.fromstring(z.read(sheet)); ns={"a":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            out=[]
            for r in root.findall(".//a:row", ns):
                vals=[]
                for c in r.findall("a:c", ns):
                    v=c.find("a:v", ns); txt="" if v is None else v.text or ""
                    if c.attrib.get("t")=="s" and txt.isdigit() and int(txt)<len(shared): txt=shared[int(txt)]
                    vals.append(txt)
                out.append(vals)
            return out
    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        terms=[t for t in re.split(r"\W+", query) if t]; scored=[]
        for r in self.records:
            text=r.get("text_index", "")
            score=sum(text.count(t) for t in terms)
            if score: scored.append((score, r))
        return [{k:v for k,v in r.items() if k not in DIRECT_IDENTIFIER_FIELDS and k != "text_index"} for _,r in sorted(scored, key=lambda x:-x[0])[:limit]]
    def dose_distribution(self, herbs: list[str], pattern: str | None = None, age: int | None = None) -> dict[str, Any]:
        out={}
        for h in herbs:
            ds=[]
            for r in self.records:
                if pattern and pattern not in str(r.get("中医诊断", "")): continue
                if age and str(r.get("年龄", "")).rstrip("岁").isdigit() and abs(int(str(r.get("年龄")).rstrip("岁"))-age) > 15: continue
                ds += [x["dose_g"] for x in r.get("herbs", []) if x["herb_name"] == h]
            out[h]={"n":len(ds), "median_g": statistics.median(ds) if ds else None, "p25_g": percentile(ds, .25), "p75_g": percentile(ds, .75), "evidence":"stratified_expert_case_distribution" if ds else "missing"}
        return out

def parse_herbs(text: str) -> list[dict[str, Any]]:
    herbs=[]
    for name, dose, method in HERB_ITEM_RE.findall(text):
        clean=name.strip().lstrip("*").strip()
        herbs.append({"herb_name": clean, "dose_g": float(dose), "administration": method.strip() or "未注明", "risk_flags": ["risk_herb"] if clean in RISK_HERBS else []})
    return herbs

def percentile(vals: list[float], q: float) -> float | None:
    if not vals: return None
    vals=sorted(vals); idx=(len(vals)-1)*q; lo=int(idx); hi=min(lo+1, len(vals)-1)
    return vals[lo] if lo==hi else vals[lo]*(hi-idx)+vals[hi]*(idx-lo)


def is_negated(text: str, term: str) -> bool:
    pattern = r"(无|否认|未见|没有|不伴|无明显)([^。；;，,]{0,12})" + re.escape(term)
    return re.search(pattern, text) is not None
