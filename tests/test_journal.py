"""Tests for the content-addressed call journal.

Two properties carry the weight:

* **A replay reproduces, or it says so.** Identical calls replay identically; a
  different call at a given position is a hard divergence, and a divergence that
  gets swallowed by an agent's fallback handling still fails the run closed.
* **A journal supplies data, never permission.** Authorisation is re-derived live
  on replay, so a journal recorded under one role cannot hand a different role a
  result it would not be allowed to fetch.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from yaobi_harness.graph import YaobiGraphRunner
from yaobi_harness.journal import (
    KIND_LLM, KIND_TOOL, MAX_JOURNAL_ENTRIES, Journal, JournalDivergence,
    JournalEntry, JournalError, JournaledLLM, canonical_json, open_journal,
    request_hash,
)
from yaobi_harness.llm.base import LLMResponse, ToolCall
from yaobi_harness.state import Budget, ClinicalRunState
from yaobi_harness.tools import CapabilityBroker, ToolRegistry


def case(complaint: str = "腰痛3月，久坐加重，右下肢麻木") -> ClinicalRunState:
    state = ClinicalRunState(complaint=complaint, role="physician")
    state.facts.update({"medications": ["布洛芬", "华法林"], "conditions": ["elderly"]})
    return state


def ledger(state: ClinicalRunState) -> list[tuple[str, str, str]]:
    return [(e.source, e.level, e.summary) for e in state.evidence.values()]


class Stub:
    name, model, available = "stub", "stub-1", True

    def __init__(self, text: str = "{}") -> None:
        self.text = text
        self.calls = 0

    def chat(self, messages, *, tools=None, temperature=0.0, max_tokens=1024, response_format_json=False):
        self.calls += 1
        return LLMResponse(text=self.text, prompt_tokens=3, completion_tokens=3)


class HashingTests(unittest.TestCase):
    def test_canonical_json_is_key_order_independent(self):
        self.assertEqual(canonical_json({"a": 1, "b": 2}), canonical_json({"b": 2, "a": 1}))

    def test_canonical_json_preserves_non_ascii(self):
        self.assertIn("腰痛", canonical_json({"complaint": "腰痛"}))

    def test_canonical_json_survives_an_unserialisable_value(self):
        """A request we cannot serialise must still hash, not crash the run."""
        self.assertTrue(canonical_json({"x": object()}))

    def test_hash_changes_with_payload_and_name_and_kind(self):
        base = request_hash(KIND_TOOL, "t", {"a": 1})
        self.assertNotEqual(base, request_hash(KIND_TOOL, "t", {"a": 2}))
        self.assertNotEqual(base, request_hash(KIND_TOOL, "u", {"a": 1}))
        self.assertNotEqual(base, request_hash(KIND_LLM, "t", {"a": 1}))

    def test_hash_is_stable_across_calls(self):
        payload = {"z": [1, 2], "a": {"n": "腰"}}
        self.assertEqual(request_hash(KIND_TOOL, "t", payload), request_hash(KIND_TOOL, "t", payload))


class JournalFileTests(unittest.TestCase):
    def test_record_then_load_round_trips(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.jsonl"
            journal = open_journal(path, mode="record")
            journal.record(KIND_TOOL, "a", {"x": 1}, {"ok": True})
            journal.record(KIND_TOOL, "b", {"x": 2}, {"ok": True})
            loaded = Journal.load(path)
            self.assertEqual([e.label for e in loaded.entries], ["a", "b"])
            self.assertEqual([e.seq for e in loaded.entries], [1, 2])

    def test_recording_truncates_rather_than_appending(self):
        """Appending to an old journal would produce a non-dense sequence."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.jsonl"
            first = open_journal(path, mode="record")
            first.record(KIND_TOOL, "a", {}, {})
            second = open_journal(path, mode="record")
            second.record(KIND_TOOL, "b", {}, {})
            self.assertEqual([e.label for e in Journal.load(path).entries], ["b"])

    def test_a_torn_tail_is_truncated_not_fatal(self):
        """A run killed mid-write leaves a partial last line; earlier lines are fine."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.jsonl"
            journal = open_journal(path, mode="record")
            journal.record(KIND_TOOL, "a", {}, {"ok": True})
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"seq": 2, "kind": "tool", "req_ha')
            loaded = Journal.load(path)
            self.assertEqual(len(loaded.entries), 1)

    def test_corruption_before_the_tail_is_fatal(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.jsonl"
            path.write_text('not json\n{"seq":1,"kind":"tool","req_hash":"x","result":{}}\n', encoding="utf-8")
            with self.assertRaises(JournalError):
                Journal.load(path)

    def test_a_non_dense_sequence_is_rejected(self):
        """A gap means entries were lost; replaying past it feeds the wrong result."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.jsonl"
            path.write_text(
                json.dumps({"seq": 1, "kind": "tool", "req_hash": "x", "result": {}}) + "\n"
                + json.dumps({"seq": 3, "kind": "tool", "req_hash": "y", "result": {}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(JournalError) as ctx:
                Journal.load(path)
            self.assertIn("dense", str(ctx.exception))

    def test_a_missing_file_is_an_error(self):
        with self.assertRaises(JournalError):
            Journal.load("/nonexistent/none.jsonl")

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaises(JournalError):
            Journal(None, mode="sideways")

    def test_meta_is_read_back_and_not_counted_as_an_entry(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.jsonl"
            journal = open_journal(path, mode="record")
            journal.write_meta({"llm_model": "m-1"})
            journal.record(KIND_TOOL, "a", {}, {})
            loaded = Journal.load(path)
            self.assertEqual(loaded.meta["llm_model"], "m-1")
            self.assertEqual(len(loaded.entries), 1)

    def test_malformed_entries_are_rejected(self):
        with self.assertRaises(JournalError):
            JournalEntry.from_dict({"kind": "tool"})

    def test_the_entry_cap_is_enforced(self):
        journal = Journal(None, mode="record")
        journal.entries = [JournalEntry(i + 1, KIND_TOOL, "h", {}) for i in range(MAX_JOURNAL_ENTRIES)]
        with self.assertRaises(JournalError):
            journal.record(KIND_TOOL, "over", {}, {})

    def test_off_mode_records_nothing(self):
        journal = Journal(None, mode="off")
        self.assertEqual(journal.record(KIND_TOOL, "a", {}, {}), -1)
        self.assertEqual(journal.entries, [])


class ReplaySemanticsTests(unittest.TestCase):
    def test_matching_calls_replay_in_order(self):
        journal = Journal(None, mode="record")
        journal.record(KIND_TOOL, "a", {"x": 1}, {"ok": True, "summary": "first"})
        journal.record(KIND_TOOL, "b", {"x": 2}, {"ok": True, "summary": "second"})

        replay = Journal(None, mode="replay", entries=journal.entries)
        hit, result = replay.next_result(KIND_TOOL, "a", {"x": 1})
        self.assertTrue(hit)
        self.assertEqual(result["summary"], "first")
        hit, result = replay.next_result(KIND_TOOL, "b", {"x": 2})
        self.assertTrue(hit)
        self.assertEqual(result["summary"], "second")

    def test_a_different_call_diverges_rather_than_falling_through(self):
        """Silently going live would make a failed replay look faithful."""
        journal = Journal(None, mode="record")
        journal.record(KIND_TOOL, "a", {"x": 1}, {"ok": True})
        replay = Journal(None, mode="replay", entries=journal.entries)
        with self.assertRaises(JournalDivergence):
            replay.next_result(KIND_TOOL, "a", {"x": 999})

    def test_a_different_kind_diverges(self):
        journal = Journal(None, mode="record")
        journal.record(KIND_TOOL, "a", {}, {})
        replay = Journal(None, mode="replay", entries=journal.entries)
        with self.assertRaises(JournalDivergence):
            replay.next_result(KIND_LLM, "a", {})

    def test_divergence_is_latched_even_though_it_raises(self):
        journal = Journal(None, mode="record")
        journal.record(KIND_TOOL, "a", {"x": 1}, {})
        replay = Journal(None, mode="replay", entries=journal.entries)
        with self.assertRaises(JournalDivergence):
            replay.next_result(KIND_TOOL, "a", {"x": 2})
        self.assertTrue(replay.diverged)
        self.assertEqual(replay.divergences[0]["seq"], 1)

    def test_exhaustion_is_a_miss_not_a_divergence(self):
        replay = Journal(None, mode="replay", entries=[])
        hit, _ = replay.next_result(KIND_TOOL, "a", {})
        self.assertFalse(hit)
        self.assertEqual(replay.live_after_exhaustion, 1)
        self.assertFalse(replay.diverged)

    def test_a_recording_journal_never_replays(self):
        journal = Journal(None, mode="record")
        journal.record(KIND_TOOL, "a", {}, {"ok": True})
        self.assertEqual(journal.next_result(KIND_TOOL, "a", {}), (False, None))

    def test_has_llm_entries_reflects_what_was_recorded(self):
        journal = Journal(None, mode="record")
        journal.record(KIND_TOOL, "a", {}, {})
        self.assertFalse(journal.has_llm_entries())
        journal.record(KIND_LLM, "m", {}, {})
        self.assertTrue(journal.has_llm_entries())


class JournaledLLMTests(unittest.TestCase):
    def test_completions_are_recorded_with_their_tool_calls(self):
        journal = Journal(None, mode="record")
        inner = Stub()
        inner.chat = lambda *a, **k: LLMResponse(  # type: ignore[method-assign]
            text="hi", tool_calls=[ToolCall("t", {"a": 1}, "c1")],
            prompt_tokens=7, completion_tokens=9,
        )
        JournaledLLM(inner, journal).chat([{"role": "user", "content": "x"}])
        self.assertEqual(len(journal.entries), 1)
        recorded = journal.entries[0].result
        self.assertEqual(recorded["text"], "hi")
        self.assertEqual(recorded["tool_calls"][0]["name"], "t")
        self.assertEqual(recorded["prompt_tokens"], 7)

    def test_replay_rebuilds_the_response_including_tool_calls(self):
        journal = Journal(None, mode="record")
        wrapper = JournaledLLM(Stub(), journal)
        wrapper.inner.chat = lambda *a, **k: LLMResponse(  # type: ignore[method-assign]
            text="body", tool_calls=[ToolCall("t", {"a": 1}, "c1")])
        wrapper.chat([{"role": "user", "content": "x"}])

        replay = Journal(None, mode="replay", entries=journal.entries, meta={"llm_model": "stub-1"})
        response = JournaledLLM(Stub(), replay).chat([{"role": "user", "content": "x"}])
        self.assertEqual(response.text, "body")
        self.assertEqual(response.tool_calls[0].name, "t")
        self.assertEqual(response.provider, "journal_replay")

    def test_availability_on_replay_reflects_the_recording(self):
        """A recording with no model must not offer the replay one.

        Otherwise agents take the autonomous branch the recording never took, and
        the journal is exhausted by a call that was never made.
        """
        tools_only = Journal(None, mode="replay", entries=[JournalEntry(1, KIND_TOOL, "h", {})])
        self.assertFalse(JournaledLLM(Stub(), tools_only).available)
        with_llm = Journal(None, mode="replay", entries=[JournalEntry(1, KIND_LLM, "h", {})])
        self.assertTrue(JournaledLLM(Stub(), with_llm).available)

    def test_the_recorded_model_name_is_adopted_on_replay(self):
        """The name is part of the request hash, so an offline replay needs it."""
        replay = Journal(None, mode="replay", entries=[], meta={"llm_model": "recorded-model"})
        self.assertEqual(JournaledLLM(Stub(), replay).model, "recorded-model")

    def test_offline_replay_raises_when_the_journal_runs_out(self):
        replay = Journal(None, mode="replay", entries=[JournalEntry(1, KIND_LLM, "h", {})])
        wrapper = JournaledLLM(Stub(), replay, offline=True)
        replay._cursor = 1  # exhausted
        with self.assertRaises(JournalError):
            wrapper.chat([{"role": "user", "content": "x"}])


class ToolJournalTests(unittest.TestCase):
    def _broker(self, journal=None, role="physician"):
        return CapabilityBroker(role, "routine", budget=Budget(), skill_registry=None, journal=journal)

    def test_a_tool_call_is_recorded_once_on_success(self):
        journal = Journal(None, mode="record")
        ToolRegistry().call(self._broker(journal), "red_flag_evidence_search", text="腰痛")
        self.assertEqual(len(journal.entries), 1)
        self.assertEqual(journal.entries[0].label, "red_flag_evidence_search")

    def test_only_the_final_result_is_recorded_after_retries(self):
        journal = Journal(None, mode="record")
        tools = ToolRegistry(failing_tools={"clinical_guideline_search"}, max_attempts=3)
        tools.call(self._broker(journal), "clinical_guideline_search", topic="x")
        self.assertEqual(len(journal.entries), 1, "retries must not each get a journal entry")
        self.assertFalse(journal.entries[0].result["ok"])

    def test_a_replayed_tool_result_is_returned_verbatim(self):
        journal = Journal(None, mode="record")
        live = ToolRegistry().call(self._broker(journal), "red_flag_evidence_search", text="腰痛")

        replay = Journal(None, mode="replay", entries=journal.entries)
        replayed = ToolRegistry().call(self._broker(replay), "red_flag_evidence_search", text="腰痛")
        self.assertEqual(replayed.summary, live.summary)
        self.assertEqual(replayed.data, live.data)
        self.assertEqual(replay.replayed, 1)

    def test_a_replayed_call_still_charges_the_budget(self):
        journal = Journal(None, mode="record")
        ToolRegistry().call(self._broker(journal), "red_flag_evidence_search", text="腰痛")
        replay = Journal(None, mode="replay", entries=journal.entries)
        broker = self._broker(replay)
        ToolRegistry().call(broker, "red_flag_evidence_search", text="腰痛")
        self.assertEqual(broker.budget.used_tool_calls, 1)

    def test_authorisation_is_re_derived_and_beats_the_journal(self):
        """A journal supplies data, never permission."""
        journal = Journal(None, mode="record")
        from yaobi_harness.skills.loader import SkillRegistry

        registry = SkillRegistry.discover(Path("yaobi_harness/skills/manifest.yaml"))
        physician = CapabilityBroker("physician", "routine", budget=Budget(),
                                    skill_registry=registry, active_skill="yaobi.formula_design",
                                    journal=journal)
        recorded = ToolRegistry().call(physician, "formula_composition_search", pattern="寒湿")
        self.assertTrue(recorded.ok)

        # Same journal, but replayed as a patient: the broker must deny before the
        # recorded result is ever reached.
        replay = Journal(None, mode="replay", entries=journal.entries)
        patient = CapabilityBroker("patient", "routine", budget=Budget(),
                                   skill_registry=registry, active_skill="yaobi.formula_design",
                                   journal=replay)
        denied = ToolRegistry().call(patient, "formula_composition_search", pattern="寒湿")
        self.assertFalse(denied.ok)
        self.assertIn("patient_role_forbids", denied.summary)
        self.assertEqual(replay.replayed, 0, "a denied call must not consume a journal entry")

    def test_a_malformed_journal_entry_becomes_a_failed_result(self):
        replay = Journal(None, mode="replay",
                         entries=[JournalEntry(1, KIND_TOOL,
                                               request_hash(KIND_TOOL, "red_flag_evidence_search",
                                                            {"text": "腰痛"}), "not-an-object")])
        result = ToolRegistry().call(self._broker(replay), "red_flag_evidence_search", text="腰痛")
        self.assertFalse(result.ok)
        self.assertIn("malformed", result.summary)


class EndToEndReplayTests(unittest.TestCase):
    def test_a_deterministic_run_replays_exactly(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.jsonl"
            record = open_journal(path, mode="record")
            first = YaobiGraphRunner(journal=record).run(case(), allow_prescription=True)

            replay = Journal.load(path, mode="replay")
            second = YaobiGraphRunner(journal=replay).run(case(), allow_prescription=True)

            self.assertEqual(first.release_status, second.release_status)
            self.assertEqual(ledger(first), ledger(second))
            self.assertEqual([(c.kind, c.text) for c in first.claims],
                             [(c.kind, c.text) for c in second.claims])
            self.assertEqual(replay.replayed, len(record.entries))
            self.assertEqual(replay.live_after_exhaustion, 0)
            self.assertFalse(replay.diverged)

    def test_a_model_driven_run_replays_with_no_model_configured(self):
        """The audit case: re-derive a past decision offline, no endpoints."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.jsonl"
            record = open_journal(path, mode="record")
            first = YaobiGraphRunner(llm=Stub(), journal=record).run(case(), allow_prescription=True)
            self.assertTrue(any(e.kind == KIND_LLM for e in record.entries))

            replay = Journal.load(path, mode="replay")
            second = YaobiGraphRunner(journal=replay).run(case(), allow_prescription=True)

            self.assertEqual(first.release_status, second.release_status)
            self.assertEqual(ledger(first), ledger(second))
            self.assertFalse(replay.diverged)
            self.assertEqual(replay.live_after_exhaustion, 0)

    def test_replaying_a_different_case_fails_closed(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.jsonl"
            record = open_journal(path, mode="record")
            YaobiGraphRunner(journal=record).run(case(), allow_prescription=True)

            replay = Journal.load(path, mode="replay")
            out = YaobiGraphRunner(journal=replay).run(case("膝关节肿痛2周"), allow_prescription=True)

            self.assertTrue(replay.diverged)
            self.assertEqual(out.release_status, "failed_closed")
            self.assertTrue(any("重放偏离" in issue for issue in out.safety_issues))

    def test_a_swallowed_divergence_still_fails_the_run_closed(self):
        """Agents catch broad exceptions to fall back; the latch must outlive that."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.jsonl"
            record = open_journal(path, mode="record")
            YaobiGraphRunner(llm=Stub(), journal=record).run(case(), allow_prescription=True)

            replay = Journal.load(path, mode="replay")
            # Corrupt one recorded LLM request so the replay issues a call the
            # journal does not expect; agents will swallow the raised divergence.
            for entry in replay.entries:
                if entry.kind == KIND_LLM:
                    entry.req_hash = "0" * 64
                    break
            out = YaobiGraphRunner(journal=replay).run(case(), allow_prescription=True)
            self.assertTrue(replay.diverged)
            self.assertEqual(out.release_status, "failed_closed")

    def test_run_meta_reports_the_journal(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.jsonl"
            out = YaobiGraphRunner(journal=open_journal(path, mode="record")).run(case())
            journal_meta = out.outputs["run_meta"]["journal"]
            self.assertEqual(journal_meta["mode"], "record")
            self.assertGreater(journal_meta["entries"], 0)

    def test_a_run_without_a_journal_reports_it_as_off(self):
        out = YaobiGraphRunner().run(case())
        self.assertEqual(out.outputs["run_meta"]["journal"], {"mode": "off"})

    def test_the_recording_notes_how_the_panel_was_configured(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "j.jsonl"
            record = open_journal(path, mode="record")
            YaobiGraphRunner(journal=record).run(case())
            hint = Journal.load(path).replay_hint()
            self.assertIsNotNone(hint["panel_concurrency"])


class CommandLineJournalTests(unittest.TestCase):
    """`chat --journal` accepted a path and recorded nothing.

    A flag that silently does nothing is the worst kind: the operator believes
    the dialogue is replayable and only finds out when they try to replay it.
    """

    def _run(self, *args: str) -> None:
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "yaobi_harness", *args],
            capture_output=True, text=True, timeout=180,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])

    def test_chat_journal_actually_records(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat.jsonl"
            self._run("chat", "--role", "patient", "--journal", str(path),
                      "--message", "腰痛3月，久坐加重，无大小便异常")
            self.assertTrue(path.is_file(), "chat --journal wrote no journal")
            self.assertGreater(len(Journal.load(path).entries), 0)

    def test_run_journal_records_and_replays(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            self._run("run", "--complaint", "腰痛3月，久坐加重", "--journal", str(path))
            self._run("run", "--complaint", "腰痛3月，久坐加重", "--replay", str(path))


if __name__ == "__main__":
    unittest.main()
