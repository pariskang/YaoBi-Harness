# Yaobi-Harness V0.1 Architecture

The implementation encodes the V2.0 protocol as a deterministic, testable harness:

1. `IntakeAgent` screens red-flag evidence and computes information gaps.
2. `PlannerAgent` creates a task graph for either urgent or routine care.
3. Urgent care enters `UrgentCareAgent`; prescription and dose tools are denied by `CapabilityBroker`.
4. Routine care runs biomedical, TCM pattern, expert-case, formula and dose agents.
5. `DoseAgent` blocks every herb lacking case dose evidence or required special-population data.
6. `CriticAgent` performs independent final checks before release.

The offline runner can later be swapped with LangGraph nodes because all node transitions are state-in/state-out and every conclusion is linked to `Evidence` records.
