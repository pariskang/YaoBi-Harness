"""Content-addressed call journal for reproducible replay.

A checkpoint records what the state *was*; it cannot show what the system *saw*.
For a clinical audit that is the wrong half: the question is never "what did the
state look like" but "given exactly this evidence, why was that recommendation
released" — and answering it requires re-running against the same tool results
and the same model outputs, not against whatever the endpoints return today.

So every host call — tool executions and LLM completions — is appended to a
JSONL journal as ``(seq, kind, req_hash, result)``. Replaying feeds the recorded
results back in order, which gives three things:

* **Audit replay.** A regulator or reviewer can re-derive a past decision offline,
  with no network and no live endpoints, and get the same answer.
* **Divergence detection.** If the replayed run issues a *different* call at a
  given sequence position, the journal refuses rather than quietly substituting a
  mismatched result. That catches a silently swapped model, a tool that reads the
  clock, and a code change that altered call order — all of which would otherwise
  look like a successful replay.
* **Cheap regression cases.** A journal captured from a real run replays as a
  fixture, so a bug found in production becomes a test without hand-writing stubs.

Design borrowed from Grok Build's workflow journal (dense sequence validation,
request hashing, torn-tail truncation, replay-divergence errors), with the
clinical addition that a journal is **never** treated as authorisation: a
replayed run re-derives its release status through the ordinary state machine, so
a tampered journal cannot promote a draft to an approved prescription — the worst
it can do is make the replay diverge, loudly.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Refuse to load a journal larger than this. A journal is bounded by the run's
#: own budget, so anything past this is corruption or a wrong file.
MAX_JOURNAL_BYTES = 64 * 1024 * 1024

#: Entries per journal. Well above any real run's tool + LLM call ceiling.
MAX_JOURNAL_ENTRIES = 10_000

#: Journal record kinds.
KIND_TOOL = "tool"
KIND_LLM = "llm"


class JournalError(RuntimeError):
    """Base class for journal faults. Never swallowed silently."""


class JournalDivergence(JournalError):
    """The replayed run issued a different call than the one recorded.

    This is a *hard* failure. Substituting the recorded result for a call that
    does not match it would produce a replay that looks faithful and is not,
    which is worse than no replay at all.
    """

    def __init__(self, seq: int, kind: str, expected: str, actual: str, detail: str = "") -> None:
        super().__init__(
            f"replay divergence at seq {seq} ({kind}): recorded request {expected[:12]} "
            f"but the run issued {actual[:12]}. The journal, the code or the model changed. {detail}".strip()
        )
        self.seq = seq
        self.kind = kind
        self.expected = expected
        self.actual = actual


@dataclass
class JournalEntry:
    seq: int
    kind: str
    req_hash: str
    result: Any
    #: Human-readable label — the tool or model name. Not part of the hash, so
    #: renaming a display label never invalidates a journal.
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "kind": self.kind, "req_hash": self.req_hash,
                "result": self.result, "label": self.label}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JournalEntry":
        try:
            return cls(
                seq=int(payload["seq"]),
                kind=str(payload["kind"]),
                req_hash=str(payload["req_hash"]),
                result=payload.get("result"),
                label=str(payload.get("label", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise JournalError(f"malformed journal entry: {exc}") from exc


def canonical_json(value: Any) -> str:
    """Stable JSON for hashing: sorted keys, no whitespace, non-ASCII preserved.

    ``default=str`` keeps an unexpected type from raising during hashing — a
    request we cannot serialise still gets a stable hash rather than crashing the
    run it was only meant to record.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def request_hash(kind: str, name: str, payload: Any) -> str:
    """Content address for one host call."""
    return hashlib.sha256(canonical_json({"kind": kind, "name": name, "payload": payload}).encode("utf-8")).hexdigest()


class Journal:
    """Append-only call journal, usable for recording or replaying.

    One instance per run. Thread-safe, because the consult panel issues calls
    from several threads at once — under concurrency the *sequence* reflects
    completion order, so a concurrent run's journal replays faithfully only
    against a matching concurrency setting. :meth:`replay_hint` reports the
    setting the recording used so a replayer can match it.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        mode: str = "record",
        entries: list[JournalEntry] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if mode not in ("record", "replay", "off"):
            raise JournalError(f"unknown journal mode {mode!r}; expected record|replay|off")
        self.path = Path(path) if path else None
        self.mode = mode
        self.entries: list[JournalEntry] = list(entries or [])
        self.meta: dict[str, Any] = dict(meta or {})
        self._cursor = 0
        self._lock = threading.RLock()
        #: Calls served from the journal rather than executed. Reported so a
        #: replay cannot be mistaken for a live run.
        self.replayed = 0
        #: Calls the replay ran live because the journal was exhausted.
        self.live_after_exhaustion = 0
        #: Latched divergences. The exception alone is not enough: agents wrap
        #: model calls in broad ``except Exception`` so they can fall back to a
        #: deterministic path, which would quietly turn a failed replay into a
        #: "fell back to rules" warning. The latch makes the runner fail closed
        #: instead, so a replay that did not reproduce cannot look like one that did.
        self.divergences: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ record
    def record(self, kind: str, name: str, payload: Any, result: Any) -> int:
        """Append one executed call. Returns its sequence number."""
        with self._lock:
            if self.mode != "record":
                return -1
            if len(self.entries) >= MAX_JOURNAL_ENTRIES:
                raise JournalError(
                    f"journal full at {MAX_JOURNAL_ENTRIES} entries; a longer run cannot be replayed"
                )
            entry = JournalEntry(
                seq=len(self.entries) + 1, kind=kind,
                req_hash=request_hash(kind, name, payload), result=result, label=name,
            )
            self.entries.append(entry)
            self._append_to_disk(entry)
            return entry.seq

    def _append_to_disk(self, entry: JournalEntry) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ replay
    def next_result(self, kind: str, name: str, payload: Any) -> tuple[bool, Any]:
        """Return ``(hit, result)`` for the next recorded call.

        ``hit`` is False when this journal is not replaying, or when it has run
        out of entries — the caller then executes the call for real. A *mismatch*
        is not a miss: it raises :class:`JournalDivergence`, because silently
        falling through to a live call would hide the fact that the replay no
        longer reproduces the recording.
        """
        with self._lock:
            if self.mode != "replay":
                return False, None
            if self._cursor >= len(self.entries):
                self.live_after_exhaustion += 1
                return False, None
            entry = self.entries[self._cursor]
            actual = request_hash(kind, name, payload)
            if entry.kind != kind or entry.req_hash != actual:
                # Naming both sides is not enough when only the arguments changed:
                # "recorded X, issued X" reads like a bug in the journal rather
                # than a different call to the same tool.
                same_call = entry.kind == kind and entry.label == name
                detail = (
                    f"同一调用 {kind}:{name}，但参数不同（病例、事实或提示已变化）"
                    if same_call else f"recorded {entry.kind}:{entry.label}, issued {kind}:{name}"
                )
                self.divergences.append({
                    "seq": entry.seq, "recorded": f"{entry.kind}:{entry.label}",
                    "issued": f"{kind}:{name}",
                    "differs_by": "arguments" if same_call else "call",
                    "recorded_hash": entry.req_hash[:16], "issued_hash": actual[:16],
                })
                raise JournalDivergence(entry.seq, entry.kind, entry.req_hash, actual, detail=detail)
            self._cursor += 1
            self.replayed += 1
            return True, entry.result

    @property
    def diverged(self) -> bool:
        with self._lock:
            return bool(self.divergences)

    # -------------------------------------------------------------------- load
    @classmethod
    def load(cls, path: str | Path, *, mode: str = "replay") -> "Journal":
        """Read a journal from disk, validating it before it is trusted."""
        file_path = Path(path)
        if not file_path.is_file():
            raise JournalError(f"journal not found: {file_path}")
        size = file_path.stat().st_size
        if size > MAX_JOURNAL_BYTES:
            raise JournalError(
                f"journal is {size} bytes, over the {MAX_JOURNAL_BYTES}-byte cap; refusing to load"
            )

        entries: list[JournalEntry] = []
        meta: dict[str, Any] = {}
        text = file_path.read_text(encoding="utf-8")
        lines = text.split("\n")
        for index, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except ValueError as exc:
                # A torn tail is expected: a run killed mid-write leaves a partial
                # last line. Anything earlier is real corruption.
                if index == len([line for line in lines if line.strip()]):
                    break
                raise JournalError(f"journal parse error at line {index}: {exc}") from exc
            if isinstance(payload, dict) and payload.get("_meta"):
                meta = payload["_meta"]
                continue
            entries.append(JournalEntry.from_dict(payload))

        _validate_dense(entries)
        if len(entries) > MAX_JOURNAL_ENTRIES:
            raise JournalError(f"journal has {len(entries)} entries, over the {MAX_JOURNAL_ENTRIES} cap")
        journal = cls(file_path, mode=mode, entries=entries, meta=meta)
        return journal

    def write_meta(self, meta: dict[str, Any]) -> None:
        """Record how this run was configured, so a replay can match it.

        Called before the first entry, so this is a plain append: the meta line
        ends up first in the file and a reader sees the run's configuration before
        any call. ``load`` skips it rather than counting it as an entry.
        """
        with self._lock:
            self.meta.update(meta)
            if self.path is None or self.mode != "record" or self.entries:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"_meta": self.meta}, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ status
    def has_llm_entries(self) -> bool:
        """Whether this journal recorded any model completion.

        A replay uses this to decide whether the recording had a model at all, so
        agents take the same branch — autonomous or deterministic — that they took
        when the journal was written.
        """
        with self._lock:
            return any(entry.kind == KIND_LLM for entry in self.entries)

    def replay_hint(self) -> dict[str, Any]:
        """What a replayer needs to know to reproduce this recording."""
        return {
            "entries": len(self.entries),
            "kinds": {kind: sum(1 for e in self.entries if e.kind == kind) for kind in (KIND_TOOL, KIND_LLM)},
            "panel_concurrency": self.meta.get("panel_concurrency"),
            "note": (
                "并发会诊的调用顺序取决于完成顺序；重放时请用相同的 panel_concurrency，"
                "或用 YAOBI_PANEL_CONCURRENCY=1 录制以获得顺序确定的日志。"
            ),
        }

    def summary(self) -> dict[str, Any]:
        """A description of this journal, safe to hand to a screen.

        ``meta`` is redacted here rather than at each caller, because there are
        three of them — ``run_meta``, ``/api/run``'s journal block and
        ``/api/replay``'s — and the first two were covered while the third
        quietly kept publishing ``llm_model``. One seam, no exceptions.

        :attr:`meta` itself is untouched: the recorded model name is part of
        every request's content address, so :class:`JournaledLLM` needs the real
        one and a replay diverges on the name alone without it.
        """
        from .llm.factory import redact_model_identity

        with self._lock:
            return {
                "mode": self.mode,
                "path": str(self.path) if self.path else None,
                "entries": len(self.entries),
                "cursor": self._cursor,
                "replayed": self.replayed,
                "live_after_exhaustion": self.live_after_exhaustion,
                "exhausted": self.mode == "replay" and self._cursor >= len(self.entries),
                "meta": redact_model_identity(self.meta),
            }


def _validate_dense(entries: list[JournalEntry]) -> None:
    """A journal must be a dense 1..N sequence.

    A gap means entries were lost or reordered, and replaying past a gap would
    feed the wrong result to a call — the exact failure the journal exists to
    prevent.
    """
    for index, entry in enumerate(entries, start=1):
        if entry.seq != index:
            raise JournalError(
                f"journal is not dense at position {index}: expected seq {index}, found {entry.seq}"
            )


@dataclass
class JournaledLLM:
    """Wraps an LLM client so its completions are recorded or replayed.

    Only ``chat`` is intercepted, because that is the entire surface the harness
    uses. ``available`` and the identity fields pass through, so a replay reports
    the provider the recording used rather than whatever is configured now.
    """

    inner: Any
    journal: Journal
    #: Set when the wrapped client is absent during a replay — the journal then
    #: supplies every completion and no endpoint is needed at all.
    offline: bool = False

    @property
    def name(self) -> str:
        return str(getattr(self.inner, "name", "journal"))

    @property
    def model(self) -> str:
        """The model this run is using — during replay, the one it *was* using.

        The model name is part of a request's content address, so a replay has to
        report the recorded name or every completion diverges on the name alone.
        An offline replay has no live client to ask, which is exactly why the
        recording writes the name into the journal's meta.
        """
        if self.journal.mode == "replay":
            recorded = self.journal.meta.get("llm_model")
            if recorded:
                return str(recorded)
        return str(getattr(self.inner, "model", "unknown"))

    @property
    def available(self) -> bool:
        """Whether a model is reachable — during replay, whether one *was*.

        Reporting blanket availability on replay was wrong: a recording made with
        no model configured contains no LLM entries, so the replay would offer
        agents a model, they would take the autonomous path the recording never
        took, and the journal would be exhausted by a call that was never made.
        The journal's own contents are the honest answer.
        """
        if self.journal.mode == "replay":
            return self.journal.has_llm_entries()
        return bool(getattr(self.inner, "available", False))

    def chat(self, messages, *, tools=None, temperature=0.0, max_tokens=1024, response_format_json=False):
        from .llm.base import LLMResponse, ToolCall

        payload = {
            "messages": messages,
            "tools": [t.name for t in (tools or [])],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format_json": response_format_json,
        }
        hit, recorded = self.journal.next_result(KIND_LLM, self.model, payload)
        if hit:
            data = recorded or {}
            return LLMResponse(
                text=str(data.get("text", "")),
                tool_calls=[
                    ToolCall(str(c.get("name", "")), c.get("arguments") or {}, str(c.get("id", "")))
                    for c in (data.get("tool_calls") or [])
                ],
                prompt_tokens=int(data.get("prompt_tokens") or 0),
                completion_tokens=int(data.get("completion_tokens") or 0),
                model=str(data.get("model") or self.model),
                provider=str(data.get("provider") or "journal_replay"),
            )
        if self.offline:
            raise JournalError(
                "重放日志已耗尽且未配置模型：本次重放需要一次未被记录的模型调用。"
                "请用完整日志重放，或配置模型以允许其余调用走实时链路。"
            )

        response = self.inner.chat(
            messages, tools=tools, temperature=temperature,
            max_tokens=max_tokens, response_format_json=response_format_json,
        )
        self.journal.record(KIND_LLM, self.model, payload, {
            "text": response.text,
            "tool_calls": [{"name": c.name, "arguments": c.arguments, "id": c.id} for c in response.tool_calls],
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "model": response.model,
            "provider": response.provider,
        })
        return response


def open_journal(path: str | Path | None, mode: str = "record") -> Journal | None:
    """Build a journal, or ``None`` when journalling is off.

    Recording to an existing path appends, which would produce a non-dense
    sequence on the next load, so a fresh recording truncates first.
    """
    if not path:
        return None
    file_path = Path(path)
    if mode == "record":
        if file_path.exists():
            file_path.unlink()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return Journal(file_path, mode="record")
    return Journal.load(file_path, mode=mode)


def journal_from_env() -> Journal | None:
    """``YAOBI_JOURNAL`` records to a path; ``YAOBI_JOURNAL_REPLAY`` replays one."""
    replay = os.environ.get("YAOBI_JOURNAL_REPLAY")
    if replay:
        return open_journal(replay, mode="replay")
    record = os.environ.get("YAOBI_JOURNAL")
    return open_journal(record, mode="record") if record else None
