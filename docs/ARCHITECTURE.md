# Yaobi-Harness V0.1 Architecture

## Layering

```
   ┌────────────────────────────────────────────────────────┐
   │ operator console (ui) · dialogue (conversation)         │
   │  delivered | audit — chat only *renders* a governed run │
   └───────────────────────────┬────────────────────────────┘
                cognition (replaceable, advisory)
   ┌────────────────────────────────────────────────────────┐
   │ PlannerAgent(LLM)   ReAct tool loop per autonomous skill│
   │ InterviewLoop(ask_patient) · ConsultPanel(subagents)    │
   │ VisionClient · semantic red-flag screen · critic        │
   └───────────────────────────┬────────────────────────────┘
                               │  proposals only
   ┌───────────────────────────▼────────────────────────────┐
   │ control plane (never bypassable)                       │
   │  plan validator · CapabilityBroker · SkillRegistry     │
   │  AdequacyJudge · consult-mode filter · Budget slices   │
   │  ToolHealth circuit breaker · Evidence ledger          │
   │  CitationGuard · release-status machine · CriticAgent  │
   └───────────────────────────┬────────────────────────────┘
   ┌───────────────────────────▼────────────────────────────┐
   │ tools · de-identified expert-case store · safety tables│
   │ interview axes (十问歌 + 骨科专科) · image reader        │
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
8. **Chat is not a generation path.** Each dialogue turn is a fresh, fully
   audited run over the accumulated narrative and facts; the model may extract
   facts (through an allowlist that excludes `physician_review`) and rephrase
   the released answer, nothing more. The urgent script is never rephrased and
   replies are dose-scanned before they leave. See [CONVERSATION.md](CONVERSATION.md).
9. **Asking is an action, and its scope is not the model's to choose.** The model
   composes questions through the ``ask_patient`` tool, but which axes are
   *required* is rule-derived; a skipped required axis is added back from the
   probe bank. An interview may only end on an :class:`AdequacyJudge` verdict,
   and a ``blocked`` verdict — a red-flag axis still unanswered — can never be
   waived by the model, a round cap, or a verifier outage. See [INTERVIEW.md](INTERVIEW.md).
10. **No subagent is prescriptive.** ``consult_mode`` intersects with a skill's
    grant rather than unioning, so no persona or site ``SKILL.md`` can reach
    ``formula_composition_search``, ``herb_dose_distribution`` or
    ``physician_review_submit``. Consult depth is capped at one and each member
    runs on a carved budget slice that charges back to the parent. Panel synthesis
    takes the *maximum* urgency, never a majority vote. See [PANEL.md](PANEL.md).
11. **A model image read is never a report.** Vision results are graded
    ``model_reasoning`` (non-releasable), carry a mandatory
    ``requires_formal_read`` that the schema refuses to see set false for
    radiology, and are discarded wholesale when the PHI pre-check finds
    identifiers. Images are never persisted — only a SHA-256 survives. See
    [VISION.md](VISION.md).
12. **A concurrent panel produces a reproducible ledger.** Members run in a
    thread pool but each writes into its own ``MemberScope``; results merge in
    *convened order*, so evidence ids depend on the roster rather than on which
    response arrived first. ``Budget`` and ``ToolHealth`` are lock-guarded — both
    are read-modify-write, so a "hard" ceiling and a two-strike breaker were not
    actually either under concurrency. A member cannot write ``risk_mode`` or
    ``release_status`` at all. See [PANEL.md](PANEL.md).
13. **A replay reproduces or it says so.** Every tool call and model completion is
    content-addressed into a journal; a differing call at a given sequence
    position is a hard divergence, latched so that an agent's broad
    ``except Exception`` cannot turn a failed replay into a "fell back to rules"
    warning. Authorisation is re-derived live, so a journal supplies data and
    never permission. See [REPLAY.md](REPLAY.md).
14. **Licences are enforced at write time.** `KnowledgeStore` rejects a
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
| Tool use | choose tools and arguments from its skill's set, self-correct | see or reach a tool outside the skill; skip the broker |
| Outputs | any shape the skill's schema allows | violate the schema; emit a gram value |
| Dialogue | extract allowlisted facts, rephrase a released answer | set `physician_review`, add clinical content, rephrase the urgent script, emit a dose |
| Interview | word/order/deepen the enquiry, add axes, propose that it is complete | skip a required axis, embed advice or a dose in a question, decide that asking may stop |
| Consult panel | reason inside a speciality view, raise urgency, add concerns | reach a prescriptive tool, convene a nested consult, lower urgency or overspend its slice |
| Vision | describe what is visible, raise a red flag, suggest questions and examinations | diagnose, claim to replace a formal read, emit a dose, read an image carrying identifiers |
| Doses | nothing | anything |

Execution autonomy is documented in [AUTONOMY.md](AUTONOMY.md); a skill's
`autonomous: true` flag is what enables it, so the decision lives in the
reviewed policy file rather than in code.

Any invalid proposal is discarded wholesale and the deterministic path is used;
the rejection is recorded in `state.warnings` and `outputs.plan.note`. Provider
errors, timeouts and budget exhaustion degrade the same way.

## Run loop

```
bootstrap(IntakeAgent) → plan → ┌ execute tasks ┐
                                │  T3 InterviewAgent (every path, urgent too)
                                │  T4 VisionAgent   (only when images attached)
                                │  N8 ConsultPanelAgent (opt-in, or LLM-planned)
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

LangGraph-native interrupt/resume, a physician review UI, model-initiated
conversation turns (the agent answers and asks, but never opens a turn itself),
persistent sessions, structured recommendation extraction from Chinese
guidelines, and large-scale adversarial evaluation with red-flag
recall/specificity baselines.

Persona I/O contracts are declared but not enforced. The journal proves a replay
matches its recording but does not prove the recording was not edited — that needs
a signature at the storage layer. Concurrency covers the consult panel only; the
graph's own task execution is still sequential, so two independent branches of a
plan do not overlap.
