"""Live progress events, so a multi-minute turn is something you can watch.

A conversation turn is a sequence of model calls and tool calls that the operator
previously experienced as a spinner. The complaint that produced this module was
exact: 「思考和调用工具skill等过程需要显示出来，流式输出」. What follows is that
stream — one event per step, published as it happens, read by the console over
the poll it already makes.

Three design notes, because the obvious implementations are all wrong here:

**Events are per-step, not per-token.** Streaming tokens would show the last of
the thirteen calls character by character and the first twelve not at all. The
useful unit is "which agent is working, on what, with which tool" — so that is
the unit. The model's own reasoning text rides along on the call that produced
it, which is the closest thing to a thought stream that a non-streaming endpoint
can honestly provide.

**The sink is bound per thread, and inherited explicitly.** Graph tasks may run
concurrently, and :mod:`threading` locals do not cross a thread boundary. A
worker therefore re-binds the sink it was handed, via :func:`bound`. Getting this
wrong fails silently — the events simply vanish — so the binding is a context
manager rather than a convention.

**Nothing here may fail a run.** Every emitter is wrapped: a progress sink that
raises would turn an observability feature into an outage. :func:`emit` swallows
its own errors by construction.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

#: Events retained per sink. A turn produces on the order of 40; a long
#: physician run with a consult panel produces a few hundred. Past this the
#: oldest are dropped, because the console only ever renders the tail.
MAX_EVENTS = 600

#: Reasoning text carried on an event. Enough to see what the model was doing,
#: bounded so a verbose reasoning model cannot inflate a poll response into
#: megabytes.
REASONING_PREVIEW = 600

_local = threading.local()


@dataclass
class ProgressEvent:
    """One observable step."""

    seq: int
    at: float
    kind: str
    label: str
    detail: str = ""
    #: Model reasoning, when the step produced any. Operator-facing only.
    reasoning: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {"seq": self.seq, "at": round(self.at, 2), "kind": self.kind,
               "label": self.label, "detail": self.detail}
        if self.reasoning:
            out["reasoning"] = self.reasoning
        if self.data:
            out["data"] = self.data
        return out


class ProgressSink:
    """A bounded, thread-safe event log with a monotonic cursor.

    The cursor is what makes polling cheap: the console asks for everything after
    the last sequence number it saw, so a turn that emits three hundred events
    still transfers each one exactly once.
    """

    def __init__(self, *, max_events: int = MAX_EVENTS) -> None:
        self._lock = threading.Lock()
        self._events: list[ProgressEvent] = []
        self._seq = 0
        self._started = time.monotonic()
        self.max_events = max_events
        #: Set when the oldest events were dropped, so a reader can say so rather
        #: than silently showing a log with a hole in it.
        self.dropped = 0

    def emit(self, kind: str, label: str, detail: str = "", *,
             reasoning: str = "", **data: Any) -> ProgressEvent | None:
        try:
            with self._lock:
                self._seq += 1
                event = ProgressEvent(
                    seq=self._seq, at=time.monotonic() - self._started,
                    kind=str(kind), label=str(label)[:160], detail=str(detail)[:400],
                    reasoning=str(reasoning or "")[:REASONING_PREVIEW],
                    data={k: v for k, v in data.items() if v is not None},
                )
                self._events.append(event)
                while len(self._events) > self.max_events:
                    self._events.pop(0)
                    self.dropped += 1
                return event
        except Exception:  # noqa: BLE001 - observability must never break a run
            return None

    def since(self, cursor: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self._events if e.seq > cursor]

    @property
    def cursor(self) -> int:
        with self._lock:
            return self._seq


# --------------------------------------------------------------------- binding
def current() -> ProgressSink | None:
    """The sink bound to this thread, if any."""
    return getattr(_local, "sink", None)


def label() -> str:
    """The activity label bound to this thread — usually the running agent."""
    return getattr(_local, "label", "") or ""


@contextmanager
def bound(sink: ProgressSink | None, activity: str = "") -> Iterator[None]:
    """Bind ``sink`` (and optionally an activity label) for this thread.

    Restores the previous binding on exit, so nesting works and a worker thread
    that borrows the parent's sink does not leak it into a pooled thread's next
    job.
    """
    previous_sink, previous_label = current(), label()
    _local.sink = sink
    if activity:
        _local.label = activity
    try:
        yield
    finally:
        _local.sink = previous_sink
        _local.label = previous_label


@contextmanager
def activity(name: str) -> Iterator[None]:
    """Name what this thread is doing, without changing which sink it writes to."""
    previous = label()
    _local.label = name
    try:
        yield
    finally:
        _local.label = previous


def emit(kind: str, text: str, detail: str = "", **data: Any) -> None:
    """Publish an event to the current thread's sink, if one is bound.

    A no-op when nothing is listening, which is the normal case for the CLI, the
    test suite and any embedding that never asked for progress.
    """
    sink = current()
    if sink is not None:
        sink.emit(kind, text, detail, **data)


@contextmanager
def step(kind: str, name: str, detail: str = "") -> Iterator[dict[str, Any]]:
    """Emit ``kind`` on entry and ``kind + "_done"`` on exit, with elapsed time.

    Yields a mutable dict; whatever the body puts in it is reported on the
    closing event. That is how an agent reports what it actually produced without
    the graph having to know the shape of every agent's output.
    """
    started = time.monotonic()
    emit(kind, name, detail)
    carry: dict[str, Any] = {}
    try:
        yield carry
    finally:
        emit(f"{kind}_done", name, str(carry.pop("detail", "") or ""),
             elapsed_s=round(time.monotonic() - started, 2), **carry)
