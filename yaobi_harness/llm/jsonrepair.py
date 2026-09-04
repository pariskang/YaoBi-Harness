"""Repair the JSON a language model actually produces.

Every consumer in this harness validates *content* — schemas check fields and
types, the plan validator checks the agent catalogue, the capability broker checks
permission. So a payload that fails to parse for a syntactic reason buys no
safety when it is rejected; it just drops the whole model-driven path to its
deterministic fallback, silently. This module exists to make that stop happening.

**Why a scanner and not regexes.** The obvious repairs — strip trailing commas,
swap single quotes for double — are one-liners with a regex and wrong. A brace
inside a string literal, an apostrophe inside a Chinese sentence, a ``//`` inside
a URL: each turns a regex repair into corruption that parses, which is far worse
than a parse failure. So :func:`repair` walks the text one character at a time,
tracking whether it is inside a string and whether the previous character was an
escape, and only ever rewrites what it knows is structure.

**What is repaired**, all of it observed in real model output:

* markdown code fences, with or without a language tag, and prose either side
* trailing commas before ``}`` or ``]``, and doubled commas
* single-quoted strings and bare (unquoted) object keys
* Python literals — ``True`` / ``False`` / ``None`` — and ``NaN`` / ``Infinity``
* ``//`` and ``/* */`` comments
* typographic and full-width punctuation a Chinese IME produces: “ ” ‘ ’ ，：
* raw newlines and tabs inside string literals
* **truncation** — an output cut off by a token limit, leaving an unterminated
  string and unclosed brackets. This is the single most valuable repair, because
  a truncated answer usually contains everything that matters.

**What is never repaired**: anything that would require guessing what the model
meant. A missing value, a key with no colon, two objects concatenated with no
delimiter — these are ambiguities, not slips, and inventing a reading for them
would put fabricated content into a clinical record. They fail.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["repair", "loads", "REPAIRS"]

#: Repair names, in the order :func:`repair` may apply them. Exposed so a caller
#: can report *what* was repaired rather than just that something was.
REPAIRS = (
    "fence",            # stripped a markdown code fence
    "prose",            # narrowed to the widest bracket-delimited span
    "smart_quotes",     # normalised typographic/full-width punctuation
    "comments",         # removed // or /* */
    "single_quotes",    # re-quoted a single-quoted string
    "bare_keys",        # quoted an unquoted object key
    "literals",         # True/False/None/NaN/Infinity -> JSON
    "control_chars",    # escaped a raw newline or tab inside a string
    "trailing_comma",   # removed a comma before a closer
    "unterminated",     # closed a string cut off by truncation
    "unclosed",         # closed brackets left open by truncation
)

#: Typographic and full-width characters that appear where JSON structure is
#: expected. Only replaced *outside* string literals, so a quotation mark inside
#: a Chinese sentence survives untouched.
_SMART = {
    "“": '"', "”": '"',      # “ ”
    "‘": "'", "’": "'",      # ‘ ’
    "，": ",", "：": ":",      # ，：
    "（": "(", "）": ")",      # （）  (inside values only; harmless)
    "［": "[", "］": "]",      # ［］
    "｛": "{", "｝": "}",      # ｛｝
}

_LITERALS = (("True", "true"), ("False", "false"), ("None", "null"),
             ("NaN", "null"), ("Infinity", "null"), ("-Infinity", "null"))

_WHITESPACE = " \t\r\n　"


def loads(text: str) -> Any:
    """Parse ``text`` as JSON, repairing it if necessary.

    Returns the parsed value, or ``None`` if it could not be parsed even after
    repair. ``None`` is also what a literal ``"null"`` parses to; callers in this
    harness treat both the same way (fall back), so the ambiguity costs nothing
    and keeping the signature simple is worth more.
    """
    result, _ = loads_with_repairs(text)
    return result


def loads_with_repairs(text: str) -> tuple[Any, list[str]]:
    """Like :func:`loads`, but also returns the repairs that were applied.

    The list is empty when the text was already valid JSON, which lets a caller
    distinguish "the model got it right" from "we fixed it" — worth logging,
    because a model that always needs repair is a prompt problem.
    """
    if not text or not text.strip():
        return None, []
    try:
        return json.loads(text), []
    except (ValueError, TypeError):
        pass
    repaired, applied = repair(text)
    if repaired is None:
        return None, applied
    try:
        return json.loads(repaired), applied
    except (ValueError, TypeError):
        pass
    return _retry_without_incomplete_tail(text, applied)


#: How many trailing fragments to try dropping. A truncation leaves one; the cap
#: is a bound on cost for pathological input, not a policy.
_MAX_TAIL_DROPS = 24


def _retry_without_incomplete_tail(text: str, applied: list[str]) -> tuple[Any, list[str]]:
    """Retry after discarding a trailing fragment truncation left behind.

    ``[{"a": 1}, {"b": 2}, {"c"`` ends with a key and no value — genuinely
    ambiguous, so it cannot be closed. But it plainly *contains* two complete
    objects, and returning them is a reading rather than an invention: this only
    ever drops from the end, and never adds anything.

    Retrying at successive commas from the back is deliberately cheap and dumb. A
    smarter incremental parser would be more precise and would also be a second
    JSON implementation to keep correct.
    """
    positions = [index for index, char in enumerate(text) if char == ","]
    for cut in reversed(positions[-_MAX_TAIL_DROPS:]):
        candidate, _ = repair(text[:cut])
        if candidate is None:
            continue
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        return value, [*applied, "dropped_incomplete_tail"]
    return None, applied


def repair(text: str) -> tuple[str | None, list[str]]:
    """Return ``(repaired_text, repairs_applied)``.

    ``repaired_text`` is ``None`` when no plausible JSON span was found at all.
    It is *not* guaranteed to parse: this function repairs what it recognises and
    leaves the verdict to :func:`json.loads`, because a repairer that also decided
    validity would have to re-implement a JSON parser to do it.
    """
    applied: list[str] = []
    candidate = _strip_fence(text, applied)
    candidate = _narrow_to_span(candidate, applied)
    if candidate is None:
        return None, applied
    return _scan(candidate, applied), applied


# --------------------------------------------------------------------- stage 1
def _strip_fence(text: str, applied: list[str]) -> str:
    """Remove a markdown code fence, keeping only the fenced body.

    Handles the common malformation of an opening fence with no closing one,
    which is what a truncated response looks like.
    """
    stripped = text.strip()
    if "```" not in stripped:
        return stripped
    applied.append("fence")
    parts = stripped.split("```")
    # An opening fence splits into ["prose", "json\n{...}", "trailer"]. Take the
    # longest inner part: with an unclosed fence there is only one, and with a
    # closed one the body is longer than the language tag.
    body = max(parts[1:], key=len) if len(parts) > 1 else stripped
    if "\n" in body:
        first, rest = body.split("\n", 1)
        # Only treat the first line as a language tag if it looks like one.
        if first.strip().isalnum() and len(first.strip()) <= 12:
            body = rest
    return body.strip()


def _narrow_to_span(text: str, applied: list[str]) -> str | None:
    """Narrow to the outermost object or array, discarding prose either side.

    The end is found by a **balanced, string-aware scan**, not by ``rfind`` of the
    closing bracket. ``rfind`` is right for prose after a complete object and
    catastrophically wrong for a truncated one: in
    ``{"tasks": [{"id": "P1"}, {"id": "P2", "agent": "Biomed`` the last ``}``
    closes the *first* task, so trimming there silently deleted the second task
    before the truncation repair could recover it. Scanning depth also handles two
    objects concatenated with no delimiter — the first complete one wins, which is
    a reading rather than a guess.
    """
    stripped = text.strip()
    starts = [i for i in (stripped.find("{"), stripped.find("[")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    end = _balanced_end(stripped, start)
    if end is None:
        # Depth never returned to zero: the document is truncated. Keep all of it
        # so ``_scan`` can close what is open.
        if start > 0:
            applied.append("prose")
        return stripped[start:]
    if start > 0 or end < len(stripped) - 1:
        applied.append("prose")
    return stripped[start : end + 1]


def _balanced_end(text: str, start: int) -> int | None:
    """Index of the bracket closing the value at ``start``, or ``None`` if unclosed.

    String-aware, because a bracket inside a string literal must not change depth.
    """
    depth = 0
    in_string = False
    quote = '"'
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in "\"'" or char in ("“", "‘"):
            in_string = True
            quote = {"“": "”", "‘": "’"}.get(char, char)
            continue
        if char in "{[" or char in ("｛", "［"):
            depth += 1
        elif char in "}]" or char in ("｝", "］"):
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                return None
    return None


# --------------------------------------------------------------------- stage 2
def _scan(text: str, applied: list[str]) -> str:
    """Rewrite structure while walking the text, then close what truncation left.

    One pass, character by character. ``in_string`` and ``escape`` are the whole
    reason this is not a pile of regexes: every rewrite below is gated on being
    outside a string literal, so content is never touched.
    """
    out: list[str] = []
    stack: list[str] = []          # open brackets, for the truncation repair
    in_string = False
    quote = '"'                    # which quote opened the current string
    #: Whether the *opening* quote was typographic. Only then may its typographic
    #: partner close the string — inside a normally-quoted string a “ or ” is
    #: content ("他说“好”") and must survive.
    smart_opened = False
    escape = False
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        # ---------------------------------------------------------- in a string
        if in_string:
            if escape:
                out.append(char)
                escape = False
                index += 1
                continue
            if char == "\\":
                out.append(char)
                escape = True
                index += 1
                continue
            if char == quote or (smart_opened and char in _SMART and _SMART[char] == quote):
                out.append('"')
                in_string = smart_opened = False
                index += 1
                continue
            if char == '"' and quote == "'":
                # A double quote inside a single-quoted string must be escaped
                # once the string is re-quoted with double quotes.
                out.append('\\"')
                index += 1
                continue
            if char in "\n\r\t":
                # A raw control character is invalid inside a JSON string. Models
                # emit them when a value contains a line break.
                _mark(applied, "control_chars")
                out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[char])
                index += 1
                continue
            out.append(char)
            index += 1
            continue

        # ------------------------------------------------------ outside strings
        was_smart = char in _SMART
        if was_smart:
            _mark(applied, "smart_quotes")
            char = _SMART[char]

        if char == "/" and index + 1 < length and text[index + 1] in "/*":
            _mark(applied, "comments")
            index = _skip_comment(text, index)
            continue

        if char == '"' or char == "'":
            if char == "'":
                _mark(applied, "single_quotes")
            # Strings are tracked by ``in_string``, deliberately *not* pushed onto
            # ``stack``: the stack exists only to close brackets truncation left
            # open, and an entry per string would make every well-formed document
            # look unclosed.
            in_string, quote, smart_opened = True, char, was_smart
            out.append('"')
            index += 1
            continue

        if char in "{[":
            stack.append(char)
            out.append(char)
            index += 1
            continue

        if char in "}]":
            _drop_trailing_comma(out, applied)
            if stack and stack[-1] in "{[":
                stack.pop()
            out.append(char)
            index += 1
            continue

        if char == ",":
            # Collapse a doubled comma; a bare one is emitted normally.
            if _last_significant(out) == ",":
                _mark(applied, "trailing_comma")
                index += 1
                continue
            out.append(char)
            index += 1
            continue

        if char.isalpha() or char == "_":
            word, next_index = _read_word(text, index)
            replacement = _literal_or_key(word, text, next_index, applied)
            out.append(replacement)
            index = next_index
            continue

        out.append(char)
        index += 1

    if in_string:
        _mark(applied, "unterminated")
        out.append('"')

    # Truncation left brackets open. Closing them recovers everything the model
    # did manage to say, which is usually the part that matters.
    while stack:
        opener = stack.pop()
        _drop_trailing_comma(out, applied)
        _mark(applied, "unclosed")
        out.append("}" if opener == "{" else "]")

    _drop_trailing_comma(out, applied)
    return "".join(out)


# --------------------------------------------------------------------- helpers
def _mark(applied: list[str], name: str) -> None:
    if name not in applied:
        applied.append(name)


def _skip_comment(text: str, index: int) -> int:
    if text[index + 1] == "/":
        end = text.find("\n", index)
        return length_or(end, len(text))
    end = text.find("*/", index)
    return len(text) if end < 0 else end + 2


def length_or(value: int, fallback: int) -> int:
    return fallback if value < 0 else value


def _read_word(text: str, index: int) -> tuple[str, int]:
    end = index
    while end < len(text) and (text[end].isalnum() or text[end] in "_-."):
        end += 1
    return text[index:end], end


def _literal_or_key(word: str, text: str, next_index: int, applied: list[str]) -> str:
    """Map a bare word to a JSON literal, or quote it as a key.

    A word followed by ``:`` is an unquoted key. Anything else that is not a
    known literal is quoted as a string value — a model writing ``{"sex": male}``
    meant the string, and quoting it is a reading, not an invention.
    """
    for python_literal, json_literal in _LITERALS:
        if word == python_literal:
            _mark(applied, "literals")
            return json_literal
    if word in ("true", "false", "null"):
        return word

    rest = text[next_index:].lstrip(_WHITESPACE)
    if rest.startswith(":"):
        _mark(applied, "bare_keys")
    else:
        _mark(applied, "literals")
    return json.dumps(word, ensure_ascii=False)


def _last_significant(out: list[str]) -> str:
    for char in reversed(out):
        if char not in _WHITESPACE:
            return char
    return ""


def _drop_trailing_comma(out: list[str], applied: list[str]) -> None:
    """Remove a comma that now sits immediately before a closer."""
    index = len(out) - 1
    while index >= 0 and out[index] in _WHITESPACE:
        index -= 1
    if index >= 0 and out[index] == ",":
        _mark(applied, "trailing_comma")
        del out[index]
