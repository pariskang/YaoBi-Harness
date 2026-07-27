"""Tests for the live progress stream.

The stream exists because a turn is a dozen sequential model calls and the
operator could previously see none of them — 「思考和调用工具skill等过程需要显示
出来」. Three properties matter, and all three are the kind that fail silently:

* the cursor delivers every event exactly once, so a poll neither repeats nor
  skips;
* the sink is per-thread and inherited *explicitly*, so a concurrent wave reports
  into the right stream and two turns never cross-talk;
* nothing here can fail a run, because observability that can take down the thing
  it observes is worse than no observability.
"""

from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from yaobi_harness import progress


class SinkTests(unittest.TestCase):
    def test_events_carry_a_monotonic_cursor(self):
        sink = progress.ProgressSink()
        for i in range(5):
            sink.emit("llm", f"agent{i}")
        self.assertEqual([e["seq"] for e in sink.since(0)], [1, 2, 3, 4, 5])
        self.assertEqual([e["seq"] for e in sink.since(3)], [4, 5])
        self.assertEqual(sink.since(5), [])

    def test_a_cursor_delivers_each_event_exactly_once(self):
        """The property the console's poll loop depends on."""
        sink = progress.ProgressSink()
        seen, cursor = [], 0
        for round_index in range(4):
            for i in range(3):
                sink.emit("tool", f"tool{round_index}{i}")
            batch = sink.since(cursor)
            cursor = batch[-1]["seq"]
            seen += batch
        self.assertEqual(len(seen), 12)
        self.assertEqual(len({e["seq"] for e in seen}), 12, "no duplicates")

    def test_the_log_is_bounded_and_says_what_it_dropped(self):
        """A silent hole in the log reads as 'nothing happened there'."""
        sink = progress.ProgressSink(max_events=10)
        for i in range(25):
            sink.emit("llm", f"call{i}")
        self.assertEqual(len(sink.since(0)), 10)
        self.assertEqual(sink.dropped, 15)

    def test_reasoning_is_carried_but_truncated(self):
        sink = progress.ProgressSink()
        sink.emit("llm_done", "IntakeAgent", "已返回", reasoning="想" * 5000)
        event = sink.since(0)[0]
        self.assertLessEqual(len(event["reasoning"]), progress.REASONING_PREVIEW)

    def test_an_event_with_no_reasoning_omits_the_field(self):
        sink = progress.ProgressSink()
        sink.emit("tool", "drug_label_lookup")
        self.assertNotIn("reasoning", sink.since(0)[0])


class BindingTests(unittest.TestCase):
    def test_emit_without_a_sink_is_a_no_op(self):
        """The CLI, the tests and any embedding never bind one."""
        progress.emit("llm", "nobody is listening")   # must not raise

    def test_binding_is_restored_on_exit(self):
        outer, inner = progress.ProgressSink(), progress.ProgressSink()
        with progress.bound(outer):
            with progress.bound(inner):
                progress.emit("llm", "inner")
            progress.emit("llm", "outer")
        self.assertEqual([e["label"] for e in outer.since(0)], ["outer"])
        self.assertEqual([e["label"] for e in inner.since(0)], ["inner"])
        self.assertIsNone(progress.current())

    def test_a_worker_thread_does_not_inherit_the_binding(self):
        """Which is exactly why the wave workers re-bind explicitly. If this ever
        starts passing by inheritance, the explicit re-bind is still correct and
        this test documents why it is there."""
        sink = progress.ProgressSink()
        with progress.bound(sink):
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(progress.emit, "llm", "from a worker").result()
        self.assertEqual(sink.since(0), [])

    def test_a_worker_that_rebinds_reports_to_the_right_sink(self):
        sink = progress.ProgressSink()

        def work(name):
            with progress.bound(sink, name):
                progress.emit("llm", progress.label())

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(work, ["A", "B", "C", "D"]))
        self.assertEqual({e["label"] for e in sink.since(0)}, {"A", "B", "C", "D"})

    def test_two_concurrent_turns_do_not_cross_talk(self):
        one, two = progress.ProgressSink(), progress.ProgressSink()
        ready = threading.Barrier(2)

        def turn(sink, tag):
            with progress.bound(sink):
                ready.wait()
                for i in range(20):
                    progress.emit("llm", f"{tag}{i}")

        threads = [threading.Thread(target=turn, args=(one, "A")),
                   threading.Thread(target=turn, args=(two, "B"))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertTrue(all(e["label"].startswith("A") for e in one.since(0)))
        self.assertTrue(all(e["label"].startswith("B") for e in two.since(0)))
        self.assertEqual(len(one.since(0)), 20)

    def test_activity_names_the_thread_without_changing_the_sink(self):
        sink = progress.ProgressSink()
        with progress.bound(sink):
            with progress.activity("BiomedicalAgent"):
                self.assertEqual(progress.label(), "BiomedicalAgent")
                progress.emit("tool", "clinical_guideline_search", agent=progress.label())
            self.assertEqual(progress.label(), "")
        self.assertEqual(sink.since(0)[0]["data"]["agent"], "BiomedicalAgent")


class StepTests(unittest.TestCase):
    def test_step_brackets_the_work_and_times_it(self):
        sink = progress.ProgressSink()
        with progress.bound(sink):
            with progress.step("agent", "TCMPatternAgent", "辨证论治") as report:
                report["detail"] = "N2 → ok"
        kinds = [(e["kind"], e["label"], e["detail"]) for e in sink.since(0)]
        self.assertEqual(kinds, [("agent", "TCMPatternAgent", "辨证论治"),
                                 ("agent_done", "TCMPatternAgent", "N2 → ok")])
        self.assertIn("elapsed_s", sink.since(0)[1]["data"])

    def test_a_raising_body_still_closes_its_step(self):
        """Otherwise a crashed agent leaves a line spinning in the log forever."""
        sink = progress.ProgressSink()
        with progress.bound(sink), self.assertRaises(RuntimeError):
            with progress.step("agent", "BoomAgent"):
                raise RuntimeError("boom")
        self.assertEqual([e["kind"] for e in sink.since(0)], ["agent", "agent_done"])


class NeverBreaksTheRunTests(unittest.TestCase):
    def test_a_sink_that_raises_does_not_propagate(self):
        class Hostile(progress.ProgressSink):
            def __init__(self):
                super().__init__()
                self._lock = None            # every emit will now raise

        sink = Hostile()
        with progress.bound(sink):
            progress.emit("llm", "still fine")   # must not raise
            with progress.step("agent", "IntakeAgent"):
                pass

    def test_unserialisable_extras_are_dropped_rather_than_carried(self):
        """``data`` ends up in a JSON response; a value that cannot be encoded
        would fail the whole poll, not just this line."""
        sink = progress.ProgressSink()
        sink.emit("tool", "x", ok=True, nothing=None)
        self.assertEqual(sink.since(0)[0]["data"], {"ok": True})


if __name__ == "__main__":
    unittest.main()
