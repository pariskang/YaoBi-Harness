from __future__ import annotations

import json
from pathlib import Path
from .state import ClinicalRunState
from .tools import CapabilityBroker, ToolRegistry
from .agent.agents import PlannerAgent, IntakeAgent, UrgentPlannerAgent, UrgentCareAgent, BiomedicalAgent, TCMPatternAgent, ExpertCaseAgent, FormulaAgent, DoseAgent, CriticAgent

class YaobiGraphRunner:
    """Explicit state graph runner; node contracts mirror a future LangGraph implementation."""
    def __init__(self, tools: ToolRegistry | None = None, checkpoint_dir: str | Path | None = None):
        self.tools = tools or ToolRegistry(); self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def _broker(self, state: ClinicalRunState) -> CapabilityBroker:
        return CapabilityBroker(state.role, state.risk_mode, budget=state.budget)

    def _checkpoint(self, state: ClinicalRunState, node: str) -> None:
        if not self.checkpoint_dir: return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / f"{state.run_id}.{node}.json").write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _mark(self, state: ClinicalRunState, agent: str, status: str) -> None:
        for t in state.tasks:
            if t.agent == agent and t.status == "pending": t.status = status

    def run(self, state: ClinicalRunState, allow_prescription: bool = False) -> ClinicalRunState:
        for agent in (IntakeAgent(), PlannerAgent()):
            state = agent.run(state, self.tools, self._broker(state)) if isinstance(agent, IntakeAgent) else agent.run(state)
            self._mark(state, agent.__class__.__name__, "ok"); self._checkpoint(state, agent.__class__.__name__)
            if state.release_status == "failed_closed": return state
        if state.risk_mode == "urgent":
            for agent in (UrgentPlannerAgent(), UrgentCareAgent(), CriticAgent()):
                state = agent.run(state, self.tools, self._broker(state)) if not isinstance(agent, UrgentPlannerAgent) else agent.run(state)
                self._mark(state, agent.__class__.__name__, "ok"); self._checkpoint(state, agent.__class__.__name__)
                if state.release_status == "failed_closed": return state
            return state
        for agent in (BiomedicalAgent(), TCMPatternAgent(), ExpertCaseAgent()):
            state = agent.run(state, self.tools, self._broker(state)); self._mark(state, agent.__class__.__name__, "ok"); self._checkpoint(state, agent.__class__.__name__)
            if state.release_status == "failed_closed": return state
        if state.role == "physician" and allow_prescription:
            for agent in (FormulaAgent(), DoseAgent()):
                state = agent.run(state, self.tools, self._broker(state)); self._mark(state, agent.__class__.__name__, "ok"); self._checkpoint(state, agent.__class__.__name__)
                if state.release_status == "failed_closed": return state
        else:
            state.release_status = "treatment_advice_only" if len(state.missing_information) < 5 else "needs_more_information"
        state = CriticAgent().run(state, self.tools, self._broker(state)); self._mark(state, "CriticAgent", "ok"); self._checkpoint(state, "CriticAgent")
        return state
