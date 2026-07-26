"""Skill discovery from ``SKILL.md`` files.

Why a second skill format at all? Because the two things a skill carries pull in
opposite directions. The *policy* half — tool allowlists, output contracts,
roles — belongs in one reviewable file where a compliance reader can see every
grant at once; that is ``manifest.yaml``. The *procedure* half is long
professional prose: a full orthopaedic history-taking protocol runs to hundreds
of lines, and folding that into YAML scalars makes it unreadable and unreviewable
exactly where clinical review matters most.

So a skill may also be a directory holding a ``SKILL.md``: YAML frontmatter for
the policy fields, markdown body for the procedure the model follows. The layout
follows the convention Grok Build and Claude Code both use, so the same file is
portable between harnesses.

Precedence, highest first:

1. ``$YAOBI_SKILL_PATH`` entries and explicit ``--skill-dir`` arguments
2. ``./.yaobi/skills/`` in the working directory (site-local override)
3. the packaged library shipped in ``yaobi_harness/skills/library/``
4. ``manifest.yaml``

A higher tier *replaces* a lower one for the same ``skill_id``. That is what lets
a hospital pin its own history-taking protocol without patching the package, and
what lets ``skill build-expert`` overwrite the placeholder expert skill.

Discovery never widens capability on its own: whatever a discovered skill claims
in ``allowed_tools`` is still checked by :class:`~yaobi_harness.tools.CapabilityBroker`
against the role and risk mode of the run, and a tool name that does not exist is
rejected at load time rather than silently ignored.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:  # pragma: no cover - import guard
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required to load Yaobi skills") from exc

SKILL_FILENAME = "SKILL.md"

#: Frontmatter keys, in the kebab-case spelling the file format uses. They are
#: normalised to the snake_case attribute names of :class:`SkillSpec`.
_KEY_ALIASES = {
    "skill-id": "skill_id",
    "id": "skill_id",
    "name": "skill_id",
    "when-to-use": "when_to_use",
    "allowed-tools": "allowed_tools",
    "forbidden-tools": "forbidden_tools",
    "allowed-roles": "allowed_roles",
    "hard-requirements": "hard_requirements",
    "output-schema": "output_schema",
    "output-key": "output_key",
    "max-tool-calls": "max_tool_calls",
    "consult-mode": "consult_mode",
    "vision": "vision",
}

#: Fields that accept "a list, or a string of comma/space separated names".
_LIST_FIELDS = ("allowed_tools", "forbidden_tools", "allowed_roles", "hard_requirements", "inputs", "outputs")


class SkillFileError(ValueError):
    """Raised for an unparseable or malformed ``SKILL.md``."""


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a ``SKILL.md`` into ``(frontmatter, body)``.

    A file with no frontmatter is not an error — it is a body-only skill whose
    identity comes from its directory name. A file that *opens* a frontmatter
    fence and never closes it is an error, because silently treating the whole
    file as prose would drop every policy field in it.
    """
    stripped = text.lstrip("﻿")
    if not stripped.startswith("---"):
        return {}, stripped.strip()

    lines = stripped.splitlines()
    closing = next((i for i, line in enumerate(lines[1:], start=1) if line.strip() in ("---", "...")), None)
    if closing is None:
        raise SkillFileError("frontmatter fence opened with '---' but never closed")

    raw = "\n".join(lines[1:closing])
    try:
        front = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise SkillFileError(f"frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(front, dict):
        raise SkillFileError("frontmatter must be a mapping")
    return front, "\n".join(lines[closing + 1 :]).strip()


def _as_list(value: Any) -> list[Any]:
    """Accept a YAML list or a comma/space separated string."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [part for part in (p.strip() for p in value.replace(",", " ").replace("，", " ").split()) if part]
    return [value]


def parse_skill_markdown(text: str, *, default_skill_id: str = "", source: str = "") -> dict[str, Any]:
    """Turn a ``SKILL.md`` into the entry shape :class:`SkillRegistry` consumes."""
    front, body = split_frontmatter(text)

    entry: dict[str, Any] = {}
    for key, value in front.items():
        normalised = _KEY_ALIASES.get(str(key).strip().lower(), str(key).strip().lower().replace("-", "_"))
        entry[normalised] = value

    entry.setdefault("skill_id", default_skill_id)
    if not entry["skill_id"]:
        raise SkillFileError("skill has no skill_id and no containing directory name to fall back on")

    for field_name in _LIST_FIELDS:
        if field_name in entry:
            entry[field_name] = _as_list(entry[field_name])

    # The body *is* the procedure. An explicit `instructions:` key in the
    # frontmatter wins only if there is no body, so the readable form is the
    # one that normally applies.
    if body:
        entry["instructions"] = body
    entry.setdefault("instructions", "")
    if source:
        entry["source"] = source
    return entry


def load_skill_file(path: str | Path) -> dict[str, Any]:
    """Read and parse one ``SKILL.md``, naming the file in any error."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillFileError(f"cannot read skill file {file_path}: {exc}") from exc
    try:
        return parse_skill_markdown(
            text, default_skill_id=file_path.parent.name, source=str(file_path)
        )
    except SkillFileError as exc:
        raise SkillFileError(f"{file_path}: {exc}") from exc


def find_skill_files(root: str | Path) -> list[Path]:
    """Every ``SKILL.md`` under ``root``, sorted for a stable load order.

    ``root`` may itself be a ``SKILL.md``, a skill directory, or a directory of
    skill directories, so a single ``--skill-dir`` flag covers all three.
    """
    base = Path(root).expanduser()
    if base.is_file():
        return [base] if base.name == SKILL_FILENAME else []
    if not base.is_dir():
        return []
    direct = base / SKILL_FILENAME
    if direct.is_file():
        return [direct]
    return sorted(base.rglob(SKILL_FILENAME))


def library_root() -> Path:
    """The packaged skill library that ships with the harness."""
    return Path(__file__).parent / "library"


def default_skill_roots(working_directory: str | Path | None = None) -> list[Path]:
    """Skill roots in *ascending* precedence, so later entries override earlier.

    ``manifest.yaml`` is loaded separately and sits below all of these.
    """
    roots = [library_root()]
    cwd = Path(working_directory) if working_directory else Path.cwd()
    site = cwd / ".yaobi" / "skills"
    if site.is_dir():
        roots.append(site)
    for entry in (os.environ.get("YAOBI_SKILL_PATH") or "").split(os.pathsep):
        if entry.strip():
            roots.append(Path(entry.strip()).expanduser())
    return roots


def collect_entries(roots: list[str | Path]) -> list[dict[str, Any]]:
    """Parse every skill under ``roots``, later roots overriding earlier ones."""
    by_id: dict[str, dict[str, Any]] = {}
    for root in roots:
        for path in find_skill_files(root):
            entry = load_skill_file(path)
            by_id[entry["skill_id"]] = entry
    return list(by_id.values())
