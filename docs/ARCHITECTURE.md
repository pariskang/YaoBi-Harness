# Yaobi-Harness V0.1 Architecture

## Layering

```
   ┌────────────────────────────────────────────────────────┐
   │ operator console (yaobi_harness.ui) — delivered | audit │
   └───────────────────────────┬────────────────────────────┘
                cognition (replaceable, advisory)
   ┌────────────────────────────────────────────────────────┐
   │ PlannerAgent(LLM)  semantic red-flag screen            │
   │ follow-up questions  adversarial critic                │
   └───────────────────────────┬────────────────────────────┘
                               │  proposals only
   ┌───────────────────────────▼────────────────────────────┐
   │ control plane (never bypassable)                       │
   │  plan validator · CapabilityBroker · SkillRegistry     │
   │  ToolHealth circuit breaker · Budget · Evidence ledger │
   │  CitationGuard · release-status machine · CriticAgent  │
   └───────────────────────────┬────────────────────────────┘
   ┌───────────────────────────▼────────────────────────────┐
   │ tools · de-identified expert-case store · safety tables│
   └───────────────────────────┬────────────────────────────┘
   ┌───────────────────────────▼────────────────────────────┐
   │ knowledge (licence-gated, operator-ingested)           │
   │  guidelines · drug labels · dose ranges · interactions │
   └────────────────────────────────────────────────────────┘
```

## Control-plane invariants

1. **The critic is terminal and unconditional.** `YaobiGraphRunner.run` invokes
   `CriticAgent` in a `finally` block; it is never a dependency-gated task, so
   it observes failed-closed runs and runs where every clinical task was
   skipped. (V0.0 gated it behind `DoseAgent`, so the patient path had no safety
   review at all.)
2. **Policy is checked before budget.** `CapabilityBroker.allow` performs
   health → risk mode → role → skill checks and only then asks the budget; the
   budget is charged by `ToolRegistry.call` after a call actually executes.
3. **Skills fail closed.** An agent with no declared `skill_id`, or a skill
   absent from the manifest, has no tool rights. An empty `allowed_tools` list
   means "no tools", not "all tools".
4. **Evidence grade is declared by the producing tool.** Placeholder sources set
   `is_stub=True` and land in the ledger as `stub_not_for_clinical_use`. No
   consumer can promote them.
5. **Dose safety is checked against the proposed dose.** `pharmacopeia_check`
   takes `doses={herb: grams}` and verifies range containment; `CriticAgent`
   re-verifies independently against the emitted draft.
6. **Screening escalates on ambiguity.** Suppression is clause-scoped and needs
   an explicit negation / third-party / hypothetical / history cue, with present
   -tense cues vetoing history cues.
7. **The console has no privileged path.** `/api/run` builds an ordinary
   `ClinicalRunState` and drives the ordinary runner; the answer it shows is
   produced by `render()` for the selected role, and the reasoning record is
   returned in a separate `audit` object labelled operator-only in the UI. See
   [CONSOLE.md](CONSOLE.md).
8. **Licences are enforced at write time.** `KnowledgeStore` rejects a
   non-commercial dataset in a commercial deployment, strips body text from
   read-only sources, and keeps credentialed sources closed without an
   attestation. The repository therefore ships connectors, never content. See
   [KNOWLEDGE.md](KNOWLEDGE.md).

## LLM containment

| Capability | Model may | Model may not |
| --- | --- | --- |
| Planning | propose a task graph | invent agents, exceed skill tool lists, add prescriptive agents in urgent mode, create cycles |
| Red flags | add signals | clear or downgrade a rule-based hit |
| Questions | rewrite/reorder | exceed `Budget.max_questions` |
| Critique | add issues (`block`/`warn`) | clear an existing safety issue or change release status directly |
| Doses | nothing | anything |

Any invalid proposal is discarded wholesale and the deterministic path is used;
the rejection is recorded in `state.warnings` and `outputs.plan.note`. Provider
errors, timeouts and budget exhaustion degrade the same way.

## Run loop

```
bootstrap(IntakeAgent) → plan → ┌ execute tasks ┐
                                │               │
                                └── critic ─────┘  repair_requests & budget.can_loop("repair")
                                        │
                                        └── finalize(release status, run_meta)
```

Repairs reset the requested task plus everything downstream of it, bounded by
`Budget.max_loops`. Each node checkpoints to `<run_id>.<node>.json` plus
`<run_id>.latest.json`; `ClinicalRunState.from_dict` and `resume_run` read them
back.

## Release states

`urgent_action_plan` · `needs_more_information` · `needs_examination` ·
`insufficient_evidence` · `treatment_advice_only` · `draft_for_physician` ·
`approved_by_physician` · `blocked` · `failed_closed`

All of them are reachable; `approved_by_physician` requires a signed per-herb
approval through `physician_review_submit`.

## Not implemented

Licensed guideline/pharmacopoeia/interaction datasets, LangGraph-native
interrupt/resume, physician review UI, multi-turn intake dialogue, and
large-scale adversarial evaluation with red-flag recall/specificity baselines.
