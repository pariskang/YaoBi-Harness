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
8. **Chat runs on top of a governed run, but the model writes it.** Each dialogue
   turn is a fresh, fully audited run over the accumulated narrative and facts —
   and that run produces *material* for the model rather than a script for it to
   read out. The agent opens the consultation, triages, composes the enquiry,
   decides when the enquiry is over, and writes every reply including the urgent
   one. Two things a message can never do: assert a physician's signature
   (`physician_review` is refused outright), and publish a dose — a gram count in
   a reply is **redacted in place**, leaving the model's sentence intact. Every
   other extracted fact is kept, governed keys through the typed allowlist and the
   rest in `facts["_extra"]`. See [CONVERSATION.md](CONVERSATION.md).
9. **Asking is an action, and the model's questions are never substituted.** The
   model composes questions through the ``ask_patient`` tool and they reach the
   patient as written: an unlabelled axis does not discard the question, a required
   axis it declined is not back-filled, and a dose inside a question is redacted
   rather than costing the enquiry. Stopping is the model's decision too — it ends
   the interview by returning no questions, with the :class:`AdequacyJudge`
   verdict supplied as advice. What the verdict still governs is the *dose
   pipeline*: a ``blocked`` verdict withholds dose-bearing output, and the
   reviewer may only clear a required axis by quoting the history that answered
   it. See [INTERVIEW.md](INTERVIEW.md).
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
| Triage | decide the level, having been given the screen's hits, the soft hits and its own semantic findings as material | do so silently — every disagreement with the keyword screen is recorded in both directions |
| Questions | compose them; they reach the patient as written | exceed the LLM budget; more than the per-round readability limit is deferred, not dropped |
| Critique | add issues (`block`/`warn`) | clear an existing safety issue or change release status directly |
| Tool use | choose tools and arguments from its skill's set, self-correct | see or reach a tool outside the skill; skip the broker |
| Outputs | any shape the skill's schema allows | violate the schema; emit a gram value |
| Dialogue | open the consultation, extract facts (governed keys typed, the rest kept in `_extra`), write every reply including the urgent one | set `physician_review`; publish a gram value — it is redacted in place, leaving the sentence intact |
| Interview | word, order and deepen the enquiry, add axes, decline a required one, **end the enquiry by asking nothing** | reach a dose draft while a required axis is open — the reviewer may clear one only by quoting the history that answered it |
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
