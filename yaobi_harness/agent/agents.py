from __future__ import annotations

from ..state import ClinicalRunState, EvidenceLevel, Task
from ..tools import CapabilityBroker, ToolRegistry, RISK_HERBS

MINIMUM_INFO = ["起病时间", "疼痛部位/放射", "神经症状", "大小便/会阴感觉", "发热外伤肿瘤史", "妊娠/年龄/肝肾功能", "当前用药/过敏", "舌脉"]

def record_tool(state: ClinicalRunState, result, level: EvidenceLevel | str) -> str:
    return state.add_evidence(level, result.tool, result.summary, result.data, ok=result.ok, error=result.error)

def require_ok(state: ClinicalRunState, result, context: str) -> bool:
    if result.ok: return True
    state.fail_closed(f"关键工具失败: {context}:{result.tool}:{result.error or result.summary}")
    return False

class PlannerAgent:
    def run(self, state: ClinicalRunState) -> ClinicalRunState:
        tasks=[Task("T1","TimelineAgent","标准化病历与时间线"), Task("T2","IntakeAgent","识别信息缺口和红旗")]
        if state.risk_mode == "urgent":
            tasks += [Task("U1","UrgentPlannerAgent","急症假设、追问与资源预算",["red_flag_evidence_search"]), Task("U2","UrgentCareAgent","急症鉴别、即时行动与转运",["emergency_resource_lookup","clinical_guideline_search"],["U1"]), Task("U3","CriticAgent","急症遗漏复核与修复",[],["U2"])]
        else:
            tasks += [Task("N1","BiomedicalAgent","西医鉴别",["clinical_guideline_search"]), Task("N2","TCMPatternAgent","辨证论治",["tcm_pattern_knowledge_search"]), Task("N3","ExpertCaseAgent","相似与反例病例",["similar_case_search","counterexample_case_search"]), Task("N4","FormulaAgent","候选治法方剂",["formula_composition_search"],["N1","N2","N3"]), Task("N5","DoseAgent","逐味剂量",["herb_dose_distribution","pharmacopeia_check","interaction_check","special_population_check"],["N4"]), Task("N6","CriticAgent","反证与安全审查",[],["N5"])]
        state.tasks=tasks; state.trace("Planner","plan",state.complaint,f"生成{len(tasks)}个任务"); return state

class IntakeAgent:
    def run(self, state: ClinicalRunState, tools: ToolRegistry, broker: CapabilityBroker) -> ClinicalRunState:
        red=tools.call(broker,"red_flag_evidence_search",text=state.complaint); eid=record_tool(state,red,EvidenceLevel.TOOL)
        if not require_ok(state, red, "intake_red_flag"): return state
        if red.data.get("hits"): state.risk_mode="urgent"
        state.missing_information=[x for x in MINIMUM_INFO if x not in state.facts]
        state.trace("IntakeAgent","screen_and_gap",output_summary=f"risk={state.risk_mode}; missing={len(state.missing_information)}",evidence_ids=[eid]); return state

class UrgentPlannerAgent:
    def run(self, state: ClinicalRunState) -> ClinicalRunState:
        hits=next((e.payload.get("hits", []) for e in state.evidence.values() if e.source=="red_flag_evidence_search"), [])
        hypotheses=sorted({h["signal"] for h in hits}) or ["unclassified_urgent_risk"]
        state.outputs["urgent_planner"]={"dangerous_hypotheses":hypotheses,"ask_now":["是否出现尿潴留/大小便失禁/会阴麻木？","是否胸痛、呼吸困难、大汗或晕厥？","是否发热、肿瘤史、外伤或进行性无力？"],"do_not_wait_for_answers":bool(hits),"max_questions":3}
        state.trace("UrgentPlannerAgent","urgent_plan_tasks",output_summary=",".join(hypotheses)); return state

class UrgentCareAgent:
    def run(self, state: ClinicalRunState, tools: ToolRegistry, broker: CapabilityBroker) -> ClinicalRunState:
        g=tools.call(broker,"clinical_guideline_search",topic="acute low back pain and non-spine emergency red flags"); ge=record_tool(state,g,EvidenceLevel.GUIDELINE)
        e=tools.call(broker,"emergency_resource_lookup",location=state.facts.get("location","中国大陆")); ee=record_tool(state,e,EvidenceLevel.TOOL)
        if not (require_ok(state,g,"urgent_guideline") and require_ok(state,e,"urgent_resource")): return state
        hypotheses=state.outputs.get("urgent_planner",{}).get("dangerous_hypotheses",[])
        phone=e.data.get("emergency_phone")
        state.outputs["urgent_action_plan"]={"risk_judgement":f"存在{hypotheses or ['急症']}风险信号，需要优先排除可致残或致命情况。","why_urgent":"红旗信号不能通过线上问诊安全排除，延误可能导致神经功能损害或危及生命。","immediate_action":f"现在不要等待完整线上问诊；请立即拨打{phone}或由家属陪同去急诊。","transport_advice":"尿潴留、会阴麻木、进行性无力、胸痛呼吸困难或晕厥时优先急救转运；不要自行驾车。","during_transport":["保持相对静止","准备既往病历、影像、用药和过敏信息","记录症状开始及进展时间"],"do_not":["不要推拿、正骨、牵引","不要自行加量止痛药或镇静药","不要因线上建议延迟急诊"],"tell_clinicians":["起病时间","神经症状进展","大小便和会阴感觉变化","胸痛/呼吸困难/发热/外伤/肿瘤/抗凝用药史"],"possible_exams":["生命体征和神经系统查体","腰椎MRI/CT或心肺急诊检查按医师判断","血常规、炎症指标等按疑点选择"],"key_questions":state.outputs.get("urgent_planner",{}).get("ask_now",[])[:3],"escalate_if":["无力或麻木进展","新发大小便异常","发热寒战","胸痛呼吸困难/意识改变"],"uncertainty":"这是风险分层与行动建议，不是确诊；线上不能排除严重疾病。"}
        state.release_status="urgent_action_plan"; state.trace("UrgentCareAgent","urgent_action",evidence_ids=[ge,ee],output_summary="动态急症行动计划"); return state

class BiomedicalAgent:
    def run(self,state,tools,broker):
        r=tools.call(broker,"clinical_guideline_search",topic="low back pain differential"); eid=record_tool(state,r,EvidenceLevel.GUIDELINE)
        if not require_ok(state,r,"biomedical_guideline"): return state
        state.outputs["biomedical"]={"differentials":["非特异性腰痛/腰肌劳损","腰椎间盘突出伴神经根病","腰椎管狭窄","脊柱感染/肿瘤/骨折需按红旗排除"],"exam_advice":["神经定位体检","红旗或持续神经根症状时线下影像"]}; state.trace("BiomedicalAgent","differential",evidence_ids=[eid]); return state

class TCMPatternAgent:
    def run(self,state,tools,broker):
        r=tools.call(broker,"tcm_pattern_knowledge_search",text=state.complaint); eid=record_tool(state,r,EvidenceLevel.TOOL)
        if not require_ok(state,r,"tcm_pattern"): return state
        state.outputs["tcm_pattern"]={"primary_pattern":r.data["patterns"][0],"candidate_patterns":r.data["patterns"],"counter_evidence_needed":["寒热表现","舌脉","疼痛固定或游走","乏力与夜痛"]}; state.trace("TCMPatternAgent","pattern",evidence_ids=[eid]); return state

class ExpertCaseAgent:
    def run(self,state,tools,broker):
        a=tools.call(broker,"similar_case_search",query=state.complaint); b=tools.call(broker,"counterexample_case_search",query=state.complaint)
        ids=[record_tool(state,a,EvidenceLevel.EXPERT_CASE),record_tool(state,b,EvidenceLevel.EXPERT_CASE)]
        if not (a.ok and b.ok): state.fail_closed("专家病例检索失败"); return state
        state.outputs["expert_cases"]={"similar":a.data.get("cases",[]),"counterexamples":b.data.get("cases",[]),"limitation":"已脱敏的单一专家经验库，不代表因果疗效证据"}; state.trace("ExpertCaseAgent","retrieve_cases",evidence_ids=ids); return state

class FormulaAgent:
    def run(self,state,tools,broker):
        pat=state.outputs.get("tcm_pattern",{}).get("primary_pattern","气血痹阻证")
        r=tools.call(broker,"formula_composition_search",pattern=pat); eid=record_tool(state,r,EvidenceLevel.TOOL)
        if not require_ok(state,r,"formula_search"): return state
        state.outputs["formula"]={"formula_name":r.data["formula_name"],"treatment_principle":["补益肝肾","活血通络","祛风除湿"],"herbs":r.data["herbs"]}; state.trace("FormulaAgent","formula_candidates",evidence_ids=[eid]); return state

class DoseAgent:
    def run(self,state,tools,broker):
        herbs=state.outputs.get("formula",{}).get("herbs",[]); pat=state.outputs.get("tcm_pattern",{}).get("primary_pattern")
        facts=state.facts.get("special_population",{}); meds=state.facts.get("medications",[]); allergies=state.facts.get("allergies",[])
        dist=tools.call(broker,"herb_dose_distribution",herbs=herbs,pattern=pat,age=facts.get("age")); sp=tools.call(broker,"special_population_check",**facts); inter=tools.call(broker,"interaction_check",herbs=herbs,medications=meds,allergies=allergies); pharm=tools.call(broker,"pharmacopeia_check",herbs=herbs)
        ids=[record_tool(state,dist,EvidenceLevel.EXPERT_CASE),record_tool(state,sp,EvidenceLevel.TOOL),record_tool(state,inter,EvidenceLevel.TOOL),record_tool(state,pharm,EvidenceLevel.TOOL)]
        if not all(x.ok for x in [dist,sp,inter,pharm]): state.fail_closed("处方安全关键工具失败"); return state
        missing=[h for h,v in dist.data.get("distributions",{}).items() if not v.get("median_g")]
        risk=[c["herb"] for c in pharm.data.get("checked",[]) if not c.get("ok")]
        if missing or not sp.data.get("pass") or not inter.data.get("pass") or risk:
            state.safety_issues += [x for x in [f"缺少剂量依据: {missing}" if missing else "", f"特殊人群必要信息缺失或风险: {sp.data}" if not sp.data.get("pass") else "", f"相互作用/过敏风险: {inter.data.get('risk_flags')}" if not inter.data.get("pass") else "", f"风险药需专项审查: {risk}" if risk else ""] if x]
            state.release_status="treatment_advice_only"; state.trace("DoseAgent","dose_blocked",evidence_ids=ids,output_summary="剂量/特殊人群/相互作用/风险药未通过"); return state
        state.outputs["prescription_draft"]={"formula_name":state.outputs["formula"]["formula_name"],"herbs":[{"herb_name":h,"dose_value":dist.data["distributions"][h]["median_g"],"dose_unit":"g","processing":"需药师/医师确认炮制","administration":"常规煎服","clinical_role":"按治法配伍","dose_evidence_ids":[ids[0]],"risk_flags":[],"physician_status":"pending"} for h in herbs],"decoction":{"frequency":"每日1剂","times_per_day":2,"duration_days":7},"overall_uncertainty":"medium","requires_physician_approval":True}
        state.release_status="draft_for_physician"; state.trace("DoseAgent","dose_generated",evidence_ids=ids); return state

class CriticAgent:
    def run(self,state,tools=None,broker=None):
        issues=[]
        if state.risk_mode=="urgent" and "prescription_draft" in state.outputs: issues.append("急症模式不得发布处方")
        if state.release_status=="draft_for_physician" and state.role!="physician": issues.append("患者端不得生成含剂量处方")
        if state.release_status=="draft_for_physician" and not state.outputs.get("prescription_draft",{}).get("requires_physician_approval"): issues.append("处方草案缺少医师审核标记")
        if any(not e.payload.get("tool_ok", True) for e in state.evidence.values()) and state.release_status not in {"failed_closed","blocked"}: issues.append("存在失败工具证据，不能发布")
        state.safety_issues += issues
        if issues: state.release_status="blocked" if state.release_status != "failed_closed" else "failed_closed"
        state.trace("CriticAgent","audit",output_summary=";".join(issues) or "未发现阻断问题"); return state
