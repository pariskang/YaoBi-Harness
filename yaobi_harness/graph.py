from __future__ import annotations

from .state import ClinicalRunState
from .tools import CapabilityBroker, ToolRegistry
from .agent.agents import PlannerAgent, IntakeAgent, UrgentCareAgent, BiomedicalAgent, TCMPatternAgent, ExpertCaseAgent, FormulaAgent, DoseAgent, CriticAgent

class YaobiGraphRunner:
    """Deterministic graph runner mirroring a LangGraph deployment: explicit state, branches, loops and safety gates."""
    def __init__(self, tools: ToolRegistry | None = None):
        self.tools = tools or ToolRegistry()

    def run(self, state: ClinicalRunState, allow_prescription: bool = False) -> ClinicalRunState:
        broker = CapabilityBroker(state.role, state.risk_mode)
        state = IntakeAgent().run(state, self.tools, broker)
        state = PlannerAgent().run(state)
        broker = CapabilityBroker(state.role, state.risk_mode)
        if state.risk_mode == "urgent":
            state = UrgentCareAgent().run(state, self.tools, broker)
            return CriticAgent().run(state, self.tools, broker)
        for agent in (BiomedicalAgent(), TCMPatternAgent(), ExpertCaseAgent()):
            state = agent.run(state, self.tools, broker)
        if state.role == "physician" and allow_prescription:
            state = FormulaAgent().run(state, self.tools, broker)
            state = DoseAgent().run(state, self.tools, broker)
        else:
            state.release_status = "treatment_advice_only" if len(state.missing_information) < 5 else "needs_more_information"
        return CriticAgent().run(state, self.tools, broker)
