from __future__ import annotations

import hashlib, hmac, os, re, statistics, zipfile, xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DIRECT_IDENTIFIER_FIELDS = {"姓名", "病案号", "地址", "医师工号", "医师姓名", "科室代码", "就诊序号"}
DATE_FIELDS = {"就诊日期"}
AUTHORIZED_FIELDS = {"性别", "年龄", "就诊月份", "主诉", "现病史", "既往史", "过敏史", "手术史", "中医四诊", "辅助检查", "中医诊断", "西医诊断", "治疗方法", "西药", "中药", "治疗"}
HERB_ITEM_RE = re.compile(r"(?:^|[,，\n])\s*\d*\s*/?\s*(?:\[[^\]]+\])?\s*\*?\s*([^*/，,\n]+?)\s*\*?\s*1\s*(?:克|g|G|mg|m1g)\s*/\s*(\d+(?:\.\d+)?)\s*(?:克|g|G)\s*/\s*用法[:：]\s*([^/，,\n]*)")
DLP_PATTERNS = [re.compile(r"1[3-9]\d{9}"), re.compile(r"[\u4e00-\u9fff]{2,8}(?:街道|社区|小区|村|路|号楼|单元)")]
RED_FLAGS = {
    "cauda_equina": ["尿潴留", "不能排尿", "尿不出来", "大小便失禁", "会阴麻木", "鞍区麻木", "下身迟钝"],
    "cardiopulmonary": ["胸痛", "胸闷", "大汗", "呼吸困难", "喘不上气", "脸色苍白", "晕厥"],
    "infection_or_tumor": ["发热", "寒战", "夜间痛", "体重下降", "肿瘤", "癌"],
    "fracture": ["外伤", "跌倒", "骨折"],
    "progressive_neuro": ["肌力下降", "足下垂", "进行性麻木", "越来越无力", "双腿越来越无力"],
}
RISK_HERBS = {"附片", "制川乌", "川乌", "草乌", "细辛", "麻黄", "全蝎", "蜈蚣"}
HERB_ALIASES = {"杜仲": ["杜仲", "盐杜仲"], "甘草": ["甘草", "甘草片", "炙甘草"], "桃仁": ["桃仁", "燀山桃仁"], "延胡索": ["延胡索", "醋延胡索"], "白芍": ["白芍", "麸白芍"], "牛膝": ["牛膝", "川牛膝"], "党参": ["党参", "炒党参"], "茯苓": ["茯苓"], "独活": ["独活"], "桑寄生": ["桑寄生"], "当归": ["当归"], "川芎": ["川芎"], "熟地黄": ["熟地黄"]}

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
    def __init__(self, role: str, risk_mode: str, budget: Any | None = None, skill_registry: Any | None = None, active_skill: str | None = None):
        self.role=role; self.risk_mode=risk_mode; self.budget=budget; self.skill_registry=skill_registry; self.active_skill=active_skill
        self.health: dict[str, bool] = defaultdict(lambda: True)
    def allow(self, tool: str) -> tuple[bool, str]:
        urgent_forbidden={"formula_composition_search","herb_dose_distribution","physician_review_submit"}; patient_forbidden=urgent_forbidden
        if self.budget is not None and not self.budget.reserve_tool(): return False,"budget_exhausted"
        if not self.health[tool]: return False,"tool_unhealthy"
        if self.risk_mode=="urgent" and tool in urgent_forbidden: return False,"urgent_mode_forbids_prescription_or_dose"
        if self.role=="patient" and tool in patient_forbidden: return False,"patient_role_forbids_formula_or_dose_tools"
        if self.skill_registry and self.active_skill:
            ok, problems = self.skill_registry.enforce(self.active_skill, self.role, [tool])
            if not ok: return False, "skill_policy_denied:" + ";".join(problems)
        return True,"allowed"

class ToolRegistry:
    def __init__(self, xlsx_path: str | Path | None = None, records: list[dict[str, Any]] | None = None, failing_tools: set[str] | None = None, authorized_ranges: dict[str, tuple[float, float]] | None = None):
        self.case_store=ExpertCaseStore.from_records(records) if records is not None else (ExpertCaseStore(xlsx_path) if xlsx_path else ExpertCaseStore.empty())
        self.failing_tools=failing_tools or set(); self.authorized_ranges=authorized_ranges or {}
        self.tools: dict[str, Callable[..., ToolResult]]={"red_flag_evidence_search":self.red_flag_evidence_search,"similar_case_search":self.similar_case_search,"counterexample_case_search":self.counterexample_case_search,"herb_dose_distribution":self.herb_dose_distribution,"tcm_pattern_knowledge_search":self.tcm_pattern_knowledge_search,"clinical_guideline_search":self.clinical_guideline_search,"pharmacopeia_check":self.pharmacopeia_check,"interaction_check":self.interaction_check,"special_population_check":self.special_population_check,"emergency_resource_lookup":self.emergency_resource_lookup,"formula_composition_search":self.formula_composition_search,"physician_review_submit":self.physician_review_submit,"patient_timeline_search":self.patient_timeline_search}
    def call(self, broker: CapabilityBroker, name: str, **kwargs: Any) -> ToolResult:
        ok, reason=broker.allow(name)
        if not ok: return ToolResult(name,False,reason,{"denied_reason":reason},error=reason)
        if name in self.failing_tools: return ToolResult(name,False,"tool_failure",error="injected_tool_failure")
        if name not in self.tools: return ToolResult(name,False,"unknown_tool",error="unknown_tool")
        try: return self.tools[name](**kwargs)
        except Exception as exc: return ToolResult(name,False,"tool_exception",error=repr(exc))
    def red_flag_evidence_search(self, text: str) -> ToolResult:
        hits=[]
        for kind, terms in RED_FLAGS.items():
            for term in terms:
                if term in text and is_current_patient_symptom(text, term): hits.append({"signal":kind,"term":term,"source":"contextual_red_flag_screen"})
        return ToolResult("red_flag_evidence_search",True,f"发现{len(hits)}个当前本人非否定风险信号",{"hits":hits},source_version="red_flags.context.v3")
    def similar_case_search(self, query: str, limit: int = 5) -> ToolResult:
        return ToolResult("similar_case_search",True,"初步假名化相似病例检索",{"cases":self.case_store.search(query,limit),"privacy":"pseudonymized_structured_fields_dlp_checked"},"expert_case")
    def counterexample_case_search(self, query: str, limit: int = 3) -> ToolResult:
        cases=[c for c in self.case_store.search(query,limit*3) if any(w in " ".join(map(str,c.values())) for w in ["加重","无效","未缓解","复发"])]
        return ToolResult("counterexample_case_search",True,"初步假名化反例检索",{"cases":cases[:limit]},"expert_case")
    def herb_dose_distribution(self, herbs: list[str], pattern: str | None = None, age: int | None = None) -> ToolResult:
        return ToolResult("herb_dose_distribution",True,"按证型/年龄/别名分层的专家剂量分布",{"distributions":self.case_store.dose_distribution(herbs,pattern=pattern,age=age)},"expert_case")
    def patient_timeline_search(self, research_patient_id: str) -> ToolResult:
        visits=[r for r in self.case_store.records if r.get("research_patient_id")==research_patient_id]
        return ToolResult("patient_timeline_search",True,"患者脱敏时间线",{"visits":visits},"expert_case")
    def tcm_pattern_knowledge_search(self, text: str) -> ToolResult:
        patterns=[]
        if any(x in text for x in ["刺痛","固定","麻木","久坐"]): patterns.append("气滞血瘀证")
        if any(x in text for x in ["乏力","酸软","久病","劳累"]): patterns.append("气血痹阻证")
        if any(x in text for x in ["冷痛","畏寒","怕冷"]): patterns.append("寒湿痹阻证")
        return ToolResult("tcm_pattern_knowledge_search",True,"证候知识匹配",{"patterns":patterns or ["待辨证"],"limits":"规则提示仅作证据检索入口"})
    def clinical_guideline_search(self, topic: str) -> ToolResult:
        return ToolResult("clinical_guideline_search",True,"本地占位指南摘要",{"guideline_id":"local_stub.not_for_clinical_release","points":["先筛查马尾综合征、感染、肿瘤、骨折、进行性神经缺损及非腰痛急症","红旗或持续/进展神经根症状需线下评估"]},"tool_result",source_version="stub-guideline-v2")
    def pharmacopeia_check(self, herbs: list[str]) -> ToolResult:
        checked=[]
        for h in herbs:
            rng=self.authorized_ranges.get(h); ok=bool(rng) and h not in RISK_HERBS
            checked.append({"herb":h,"ok":ok,"authorized_range_available":bool(rng),"range_g":rng,"risk_flags":["risk_herb_requires_special_review"] if h in RISK_HERBS else []})
        return ToolResult("pharmacopeia_check",True,"药典/正式范围校验",{"version":"authorized_ranges_stub" if self.authorized_ranges else "no_authorized_pharmacopeia_dataset","checked":checked},"tool_result")
    def interaction_check(self, herbs: list[str], medications: list[str] | None = None, allergies: list[str] | None = None, medications_confirmed: bool = False, allergies_confirmed: bool = False) -> ToolResult:
        flags=[]; meds=medications or []; allergies=allergies or []
        if not medications_confirmed: flags.append("current_medications_unknown")
        if not allergies_confirmed: flags.append("allergies_unknown")
        if any("抗凝" in m or "华法林" in m for m in meds): flags.append("活血药与抗凝/抗血小板药需专项审查")
        for h in herbs:
            if h in allergies: flags.append(f"过敏史包含{h}")
        return ToolResult("interaction_check",True,"相互作用/过敏初筛",{"risk_flags":flags,"pass":not flags})
    def special_population_check(self, pregnancy: bool | None = None, age: int | None = None, renal: str | None = None, liver: str | None = None) -> ToolResult:
        missing=[]
        if pregnancy is None: missing.append("pregnancy")
        if age is None: missing.append("age")
        if renal in (None,"unknown","未知",""): missing.append("renal")
        if liver in (None,"unknown","未知",""): missing.append("liver")
        flags=[]
        if pregnancy is True: flags.append("pregnancy_requires_no_remote_herbal_draft")
        if isinstance(age,int) and (age<18 or age>=75): flags.append("age_requires_special_dose_review")
        return ToolResult("special_population_check",True,"特殊人群信息审查",{"missing":missing,"risk_flags":flags,"pass":not missing and not flags})
    def emergency_resource_lookup(self, location: str = "中国大陆") -> ToolResult:
        phone="120" if "中国" in location else None
        return ToolResult("emergency_resource_lookup",True,"急救资源",{"emergency_phone":phone,"advice":"如所在地急救号码未知，请使用当地官方急救电话；中国大陆为120"})
    def formula_composition_search(self, pattern: str) -> ToolResult:
        base=["独活","桑寄生","杜仲","牛膝","当归","川芎","白芍","熟地黄","党参","茯苓","甘草"]
        if "瘀" in pattern: base += ["桃仁","红花","延胡索"]
        return ToolResult("formula_composition_search",True,"候选方群组成",{"formula_name":"独活寄生汤加减候选","herbs":base,"requires_dose_evidence":True})
    def physician_review_submit(self, prescription: dict[str, Any], approvals: dict[str, bool], physician_id: str | None = None, signature: str | None = None) -> ToolResult:
        problems=[]; herbs=prescription.get("herbs") or []
        if not physician_id or not signature: problems.append("missing_physician_identity_or_signature")
        if not prescription.get("prescription_hash"): problems.append("missing_prescription_hash")
        for h in herbs:
            name=h.get("herb_name")
            for field in ["dose_value","dose_unit","processing","dose_evidence_ids"]:
                if not h.get(field): problems.append(f"{name}:missing_{field}")
            if name in RISK_HERBS and not h.get("special_review"): problems.append(f"{name}:risk_herb_special_review_missing")
            if not approvals.get(name): problems.append(f"{name}:not_approved")
        ok=bool(herbs) and not problems
        return ToolResult("physician_review_submit",ok,"医师逐味审核通过" if ok else "医师审核未通过",{"approved":ok,"problems":problems},error=None if ok else "review_incomplete")

class ExpertCaseStore:
    def __init__(self, xlsx_path: str | Path): self.path=Path(xlsx_path); self.records=self._load_xlsx_structured(self.path)
    @classmethod
    def empty(cls): o=cls.__new__(cls); o.path=None; o.records=[]; return o
    @classmethod
    def from_records(cls, records: list[dict[str, Any]]): o=cls.empty(); o.records=[cls._deidentify_record(r) for r in records]; return o
    @staticmethod
    def _research_id(source: str) -> str:
        key=os.environ.get("YAOBI_DEID_KEY") or os.urandom(32).hex()
        return "YP" + hmac.new(key.encode(), source.encode(), hashlib.sha256).hexdigest()[:12]
    @classmethod
    def _deidentify_record(cls, row: dict[str, Any]) -> dict[str, Any]:
        source=str(row.get("病案号") or row.get("case_id") or row.get("就诊序号") or row)
        rec={}
        for k,v in row.items():
            if k in DIRECT_IDENTIFIER_FIELDS or v in (None,""): continue
            key="就诊月份" if k in DATE_FIELDS else k
            if key in AUTHORIZED_FIELDS: rec[key]=sanitize_text(generalize_date(v) if k in DATE_FIELDS else v)
        rec["research_patient_id"]=cls._research_id(source); rec["herbs"]=parse_herbs(str(row.get("中药","")))
        rec["text_index"]=" ".join(str(rec.get(k,"")) for k in ["性别","年龄","主诉","现病史","既往史","中医四诊","中医诊断","西医诊断","治疗方法","治疗"])
        return rec
    def _load_xlsx_structured(self, path: Path) -> list[dict[str, Any]]:
        rows=self._xlsx_rows(path)
        if not rows: return []
        headers=[str(x).strip() for x in rows[0]]
        return [self._deidentify_record({headers[i]: r[i] if i < len(r) else "" for i in range(len(headers))}) for r in rows[1:] if any(r)]
    def _xlsx_rows(self, path: Path) -> list[list[str]]:
        with zipfile.ZipFile(path) as z:
            shared=[]; ns={"a":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            if "xl/sharedStrings.xml" in z.namelist():
                root=ET.fromstring(z.read("xl/sharedStrings.xml"))
                for si in root.findall("a:si",ns): shared.append("".join(t.text or "" for t in si.findall(".//a:t",ns)))
            sheet=next(n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
            root=ET.fromstring(z.read(sheet)); out=[]
            for row in root.findall(".//a:row",ns):
                cells={}
                for c in row.findall("a:c",ns):
                    ref=c.attrib.get("r",""); idx=col_to_index(re.sub(r"\d+","",ref))
                    v=c.find("a:v",ns); txt="" if v is None else v.text or ""
                    if c.attrib.get("t")=="s" and txt.isdigit() and int(txt)<len(shared): txt=shared[int(txt)]
                    cells[idx]=txt
                out.append([cells.get(i,"") for i in range(max(cells.keys(), default=-1)+1)])
            return out
    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        terms=[t for t in re.split(r"\W+",query) if t]; scored=[]
        for r in self.records:
            score=sum(str(r.get("text_index","")).count(t) for t in terms)
            if score: scored.append((score,r))
        return [{k:v for k,v in r.items() if k not in DIRECT_IDENTIFIER_FIELDS and k!="text_index" and k!="中药"} for _,r in sorted(scored,key=lambda x:-x[0])[:limit]]
    def dose_distribution(self, herbs: list[str], pattern: str | None = None, age: int | None = None) -> dict[str, Any]:
        out={}
        for h in herbs:
            aliases=set(HERB_ALIASES.get(h,[h])); ds=[]
            for r in self.records:
                if pattern and pattern not in str(r.get("中医诊断","")): continue
                if age and str(r.get("年龄","")).rstrip("岁").isdigit() and abs(int(str(r.get("年龄")).rstrip("岁"))-age)>15: continue
                ds += [x["dose_g"] for x in r.get("herbs",[]) if x["herb_name"] in aliases]
            out[h]={"n":len(ds),"median_g":statistics.median(ds) if ds else None,"p25_g":percentile(ds,.25),"p75_g":percentile(ds,.75),"evidence":"stratified_expert_case_distribution" if ds else "missing","meets_min_n":len(ds)>=3,"outlier_flag":bool(ds and (max(ds)>60 or min(ds)<=0))}
        return out

def parse_herbs(text: str) -> list[dict[str, Any]]:
    herbs=[]
    for name,dose,method in HERB_ITEM_RE.findall(text):
        clean=name.strip().lstrip("*").strip(); val=float(dose)
        herbs.append({"herb_name":clean,"dose_g":val,"administration":method.strip() or "未注明","risk_flags":["risk_herb"] if clean in RISK_HERBS else []})
    return herbs

def percentile(vals: list[float], q: float) -> float | None:
    if not vals: return None
    vals=sorted(vals); idx=(len(vals)-1)*q; lo=int(idx); hi=min(lo+1,len(vals)-1)
    return vals[lo] if lo==hi else vals[lo]*(hi-idx)+vals[hi]*(idx-lo)

def col_to_index(col: str) -> int:
    n=0
    for ch in col: n=n*26+ord(ch.upper())-64
    return n-1

def sanitize_text(value: Any) -> str:
    text=str(value)
    for pat in DLP_PATTERNS: text=pat.sub("[REDACTED]",text)
    return text

def generalize_date(value: Any) -> str:
    text=str(value)
    m=re.match(r"(20\d{2})[-/.年](\d{1,2})", text)
    return f"{m.group(1)}-{int(m.group(2)):02d}" if m else "date_generalized"

def is_current_patient_symptom(text: str, term: str) -> bool:
    idx=text.find(term)
    if idx < 0: return False
    window=text[max(0,idx-18): min(len(text), idx+18)]
    if re.search(r"(无|否认|未见|没有|不伴|无明显)([^。；;，,]{0,12})"+re.escape(term), window): return False
    if re.search(r"(父亲|母亲|家族|亲属|别人|如果|担心|害怕|以后|既往|去年|多年前|已痊愈|恢复)", window): return False
    if term in {"骨质疏松", "夜间痛"} and not re.search(r"(新发|突然|加重|发热|外伤|跌倒|肿瘤|癌|体重下降)", text): return False
    return True
