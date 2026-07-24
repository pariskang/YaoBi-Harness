from __future__ import annotations

from typing import Any
from ..state import ClinicalRunState, EvidenceLevel, Task
from ..tools import CapabilityBroker, ToolRegistry

MINIMUM_INFO = ["起病时间", "疼痛部位/放射", "神经症状", "大小便/会阴感觉", "发热外伤肿瘤史", "妊娠/年龄/肝肾功能", "当前用药/过敏", "舌脉"]

class PlannerAgent:
    def run(self, state: ClinicalRunState) -> ClinicalRunState:
        tasks = [Task("T1", "TimelineAgent", "标准化病历与时间线"), Task("T2", "IntakeAgent", "识别信息缺口和红旗")]
        if state.risk_mode == "urgent":
            tasks += [Task("U1", "UrgentCareAgent", "急症行动与转运", ["red_flag_evidence_search", "emergency_resource_lookup", "clinical_guideline_search"]), Task("U2", "CriticAgent", "急症遗漏复核")]
        else:
            tasks += [
                Task("N1", "BiomedicalAgent", "西医鉴别", ["clinical_guideline_search"]),
                Task("N2", "TCMPatternAgent", "辨证论治", ["tcm_pattern_knowledge_search"]),
                Task("N3", "ExpertCaseAgent", "相似与反例病例", ["similar_case_search", "counterexample_case_search"]),
                Task("N4", "FormulaAgent", "候选治法方剂", ["formula_composition_search"], ["N1", "N2", "N3"]),
                Task("N5", "DoseAgent", "逐味剂量", ["herb_dose_distribution", "pharmacopeia_check", "interaction_check", "special_population_check"], ["N4"]),
                Task("N6", "CriticAgent", "反证与安全审查", ["interaction_check", "special_population_check"], ["N5"]),
            ]
        state.tasks = tasks; state.trace("Planner", "plan", state.complaint, f"生成{len(tasks)}个任务")
        return state

class IntakeAgent:
    def run(self, state: ClinicalRunState, tools: ToolRegistry, broker: CapabilityBroker) -> ClinicalRunState:
        red = tools.call(broker, "red_flag_evidence_search", text=state.complaint)
        eid = state.add_evidence(EvidenceLevel.TOOL, red.tool, red.summary, red.data)
        if red.data.get("hits"):
            state.risk_mode = "urgent"
        state.missing_information = [x for x in MINIMUM_INFO if x not in state.facts]
        state.trace("IntakeAgent", "screen_and_gap", output_summary=f"risk={state.risk_mode}; missing={len(state.missing_information)}", evidence_ids=[eid])
        return state

class UrgentCareAgent:
    def run(self, state: ClinicalRunState, tools: ToolRegistry, broker: CapabilityBroker) -> ClinicalRunState:
        g = tools.call(broker, "clinical_guideline_search", topic="acute low back pain red flags")
        e = tools.call(broker, "emergency_resource_lookup", location=state.facts.get("location", "中国大陆"))
        ids=[state.add_evidence(EvidenceLevel.GUIDELINE, g.tool, g.summary, g.data), state.add_evidence(EvidenceLevel.TOOL, e.tool, e.summary, e.data)]
        state.outputs["urgent_action_plan"] = {
            "risk_judgement": "存在腰痛红旗风险，需优先排除马尾综合征、感染/肿瘤、骨折或进行性神经损害。",
            "immediate_action": f"若有尿潴留/大小便失禁/会阴麻木/进行性无力，请立即拨打{e.data.get('emergency_phone')}或由家属陪同急诊。",
            "do_not": ["不要等待完整线上问诊后再就医", "不要自行推拿、正骨、牵引", "不要自行服用镇静止痛药掩盖病情"],
            "tell_clinicians": ["起病时间", "神经症状进展", "大小便和会阴感觉变化", "肿瘤/感染/外伤/抗凝用药史"],
            "possible_exams": ["神经系统查体", "腰椎MRI/CT按急诊医师判断", "血常规、炎症指标等按疑点选择"],
            "key_questions": state.missing_information[:6],
            "uncertainty": "线上不能远程排除严重疾病；行动建议基于风险信号而非确诊。",
        }
        state.release_status = "urgent_action_plan"
        state.trace("UrgentCareAgent", "urgent_plan", evidence_ids=ids, output_summary="生成急症行动计划并收回处方能力")
        return state

class BiomedicalAgent:
    def run(self, state, tools, broker):
        r=tools.call(broker,"clinical_guideline_search",topic="low back pain differential")
        eid=state.add_evidence(EvidenceLevel.GUIDELINE,r.tool,r.summary,r.data)
        state.outputs["biomedical"]={"differentials":["非特异性腰痛/腰肌劳损","腰椎间盘突出伴神经根病","腰椎管狭窄","脊柱感染/肿瘤/骨折需按红旗排除"],"exam_advice":["神经定位体检","红旗或持续神经根症状时线下影像"]}
        state.trace("BiomedicalAgent","differential",evidence_ids=[eid]); return state

class TCMPatternAgent:
    def run(self,state,tools,broker):
        r=tools.call(broker,"tcm_pattern_knowledge_search",text=state.complaint)
        eid=state.add_evidence(EvidenceLevel.TOOL,r.tool,r.summary,r.data)
        state.outputs["tcm_pattern"]={"primary_pattern":r.data["patterns"][0],"candidate_patterns":r.data["patterns"],"counter_evidence_needed":["寒热表现","舌脉","疼痛固定或游走","乏力与夜痛"]}
        state.trace("TCMPatternAgent","pattern",evidence_ids=[eid]); return state

class ExpertCaseAgent:
    def run(self,state,tools,broker):
        a=tools.call(broker,"similar_case_search",query=state.complaint); b=tools.call(broker,"counterexample_case_search",query=state.complaint)
        ids=[state.add_evidence(EvidenceLevel.EXPERT_CASE,a.tool,a.summary,a.data),state.add_evidence(EvidenceLevel.EXPERT_CASE,b.tool,b.summary,b.data)]
        state.outputs["expert_cases"]={"similar":a.data.get("cases",[]),"counterexamples":b.data.get("cases",[]),"limitation":"单一专家经验库，不代表因果疗效证据"}
        state.trace("ExpertCaseAgent","retrieve_cases",evidence_ids=ids); return state

class FormulaAgent:
    def run(self,state,tools,broker):
        pat=state.outputs.get("tcm_pattern",{}).get("primary_pattern","气血痹阻证")
        r=tools.call(broker,"formula_composition_search",pattern=pat)
        eid=state.add_evidence(EvidenceLevel.TOOL,r.tool,r.summary,r.data)
        if r.ok: state.outputs["formula"]={"formula_name":r.data["formula_name"],"treatment_principle":["补益肝肾","活血通络","祛风除湿"],"herbs":r.data["herbs"]}
        state.trace("FormulaAgent","formula_candidates",evidence_ids=[eid]); return state

class DoseAgent:
    def run(self,state,tools,broker):
        herbs=state.outputs.get("formula",{}).get("herbs",[])
        dist=tools.call(broker,"herb_dose_distribution",herbs=herbs); sp=tools.call(broker,"special_population_check",**state.facts.get("special_population",{})); inter=tools.call(broker,"interaction_check",herbs=herbs,medications=state.facts.get("medications",[])); pharm=tools.call(broker,"pharmacopeia_check",herbs=herbs)
        ids=[state.add_evidence(EvidenceLevel.EXPERT_CASE,dist.tool,dist.summary,dist.data),state.add_evidence(EvidenceLevel.TOOL,sp.tool,sp.summary,sp.data),state.add_evidence(EvidenceLevel.TOOL,inter.tool,inter.summary,inter.data),state.add_evidence(EvidenceLevel.PHARMACOPEIA,pharm.tool,pharm.summary,pharm.data)]
        missing=[h for h,v in dist.data.get("distributions",{}).items() if not v.get("median_g")]
        if missing or not sp.data.get("pass",False):
            state.safety_issues += [f"缺少剂量依据: {missing}" if missing else "", f"特殊人群必要信息缺失: {sp.data.get('missing')}"]
            state.release_status="treatment_advice_only"; state.trace("DoseAgent","dose_blocked",evidence_ids=ids,output_summary="剂量证据不足或特殊人群信息缺失"); return state
        state.outputs["prescription_draft"]={"formula_name":state.outputs["formula"]["formula_name"],"herbs":[{"herb_name":h,"dose_value":dist.data["distributions"][h]["median_g"],"dose_unit":"g","processing":"需医师/药典数据确认","administration":"常规煎服","clinical_role":"按治法配伍","dose_evidence_ids":[ids[0]],"risk_flags":[],"physician_status":"pending"} for h in herbs],"decoction":{"frequency":"每日1剂","times_per_day":2,"duration_days":7},"requires_physician_approval":True}
        state.release_status="draft_for_physician"; state.trace("DoseAgent","dose_generated",evidence_ids=ids); return state

class CriticAgent:
    def run(self,state,tools=None,broker=None):
        issues=[]
        if state.risk_mode=="urgent" and "prescription_draft" in state.outputs: issues.append("急症模式不得发布处方")
        if state.release_status=="draft_for_physician" and state.role!="physician": issues.append("患者端不得生成含剂量处方")
        if state.release_status=="draft_for_physician" and not state.outputs.get("prescription_draft",{}).get("requires_physician_approval"): issues.append("处方草案缺少医师审核标记")
        state.safety_issues += issues
        if issues: state.release_status="blocked"
        state.trace("CriticAgent","audit",output_summary=";".join(issues) or "未发现阻断问题")
        return state
