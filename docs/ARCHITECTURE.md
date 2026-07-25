# Yaobi-Harness V0.0 Safety Skeleton

This repository now intentionally describes the implementation as a safety skeleton rather than a complete V0.1 clinical agent.

## Implemented now

1. Raw Excel rows are reconstructed by A1 cell coordinates, converted to pseudonymized structured records before retrieval, direct identifiers and visit sequence numbers are dropped, dates are generalized, and free text receives a basic DLP pass.
2. Tool calls pass through `CapabilityBroker`, which enforces role/risk permissions and consumes the run budget.
3. Critical tool errors move the run to `failed_closed` instead of registering fake guideline/pharmacopeia evidence.
4. Urgent care is modeled as planner → urgent action → critic, with dynamic red-flag hypotheses and no prescription tools.
5. Dose generation requires stratified expert-case dose evidence, minimum sample size, outlier checks, authorized dose ranges, special-population clearance, confirmed medication/allergy history, interaction clearance and risk-herb clearance; without authorized ranges no gram-level draft is emitted.
6. Skill manifests are loaded by the graph runner and enforced by the Broker for each tool call.

## Not implemented yet

Real LangGraph execution, LLM-driven planning/question generation, checkpoint resume, licensed pharmacopoeia and interaction datasets, physician review UI, claim-level EvidenceBinder/CitationGuard and large adversarial evaluation remain future work.
