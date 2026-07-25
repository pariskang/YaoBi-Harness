from __future__ import annotations

import json
from pathlib import Path
from .state import ClinicalRunState
from .tools import CapabilityBroker, ToolRegistry
from .skills.loader import SkillRegistry
from .agent.agents import PlannerAgent, IntakeAgent, UrgentPlannerAgent, UrgentCareAgent, BiomedicalAgent, TCMPatternAgent, ExpertCaseAgent, FormulaAgent, DoseAgent, CriticAgent

SKILL_FOR_AGENT = {"IntakeAgent":"yaobi.intake", "UrgentCareAgent":"yaobi.urgent_triage", "BiomedicalAgent":"yaobi.biomedical_differential", "TCMPatternAgent":"yaobi.tcm_pattern", "ExpertCaseAgent":"yaobi.expert_case_reasoning", "FormulaAgent":"yaobi.formula_design", "DoseAgent":"yaobi.dose_generation"}

class TimelineAgent:
    def run(self, state: ClinicalRunState) -> ClinicalRunState:
        state.outputs["timeline"]={"events":[{"source":"chief_complaint","text":state.complaint}],"limitations":"CLI single-turn input; no resume facts supplied"}
        state.trace("TimelineAgent","timeline_from_input",output_summary="single-turn timeline initialized")
        return state

class YaobiGraphRunner:
    """Task-driven state graph runner; node contracts mirror a future LangGraph implementation."""
    def __init__(self, tools: ToolRegistry | None = None, checkpoint_dir: str | Path | None = None, skill_manifest: str | Path | None = None):
        self.tools=tools or ToolRegistry(); self.checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None
        manifest=Path(skill_manifest) if skill_manifest else Path(__file__).parent / "skills" / "manifest.yaml"
        self.skill_registry=SkillRegistry.from_file(manifest) if manifest.exists() else None
    def _broker(self, state: ClinicalRunState, agent_name: str | None = None) -> CapabilityBroker:
        return CapabilityBroker(state.role,state.risk_mode,budget=state.budget,skill_registry=self.skill_registry,active_skill=SKILL_FOR_AGENT.get(agent_name or ""))
    def _checkpoint(self,state:ClinicalRunState,node:str)->None:
        if not self.checkpoint_dir: return
        self.checkpoint_dir.mkdir(parents=True,exist_ok=True)
        (self.checkpoint_dir/f"{state.run_id}.{node}.json").write_text(json.dumps(state.to_dict(),ensure_ascii=False,indent=2),encoding="utf-8")
    def _mark(self,state:ClinicalRunState,agent:str,status:str)->None:
        for t in state.tasks:
            if t.agent==agent and t.status=="pending": t.status=status
    def _deps_ok(self,state:ClinicalRunState, task)->bool:
        status={t.task_id:t.status for t in state.tasks}
        return all(status.get(dep)=="ok" for dep in task.depends_on)
    def run(self,state:ClinicalRunState,allow_prescription:bool=False)->ClinicalRunState:
        state=IntakeAgent().run(state,self.tools,self._broker(state,"IntakeAgent")); self._checkpoint(state,"IntakeAgent")
        if state.release_status=="failed_closed": return state
        state=PlannerAgent().run(state); self._checkpoint(state,"PlannerAgent")
        agents={"TimelineAgent":TimelineAgent(),"UrgentPlannerAgent":UrgentPlannerAgent(),"UrgentCareAgent":UrgentCareAgent(),"BiomedicalAgent":BiomedicalAgent(),"TCMPatternAgent":TCMPatternAgent(),"ExpertCaseAgent":ExpertCaseAgent(),"FormulaAgent":FormulaAgent(),"DoseAgent":DoseAgent(),"CriticAgent":CriticAgent()}
        for task in state.tasks:
            if task.agent=="IntakeAgent": task.status="ok"; continue
            if task.agent in {"FormulaAgent","DoseAgent"} and not (state.role=="physician" and allow_prescription): task.status="skipped"; continue
            if not self._deps_ok(state,task): task.status="skipped_dependency"; continue
            agent=agents.get(task.agent)
            if agent is None: task.status="failed"; state.fail_closed(f"未实现任务Agent: {task.agent}"); return state
            if task.agent in {"TimelineAgent","UrgentPlannerAgent","CriticAgent"}: state=agent.run(state)
            else: state=agent.run(state,self.tools,self._broker(state,task.agent))
            task.status="ok" if state.release_status!="failed_closed" else "failed"; self._checkpoint(state,task.agent)
            if state.release_status=="failed_closed": return state
            if state.risk_mode=="urgent" and task.agent=="CriticAgent": return state
        if state.release_status=="needs_more_information": state.release_status="treatment_advice_only" if len(state.missing_information)<5 else "needs_more_information"
        return state
