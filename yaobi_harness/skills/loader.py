"""Skill manifest loading and enforcement.

The manifest is a *policy file*: it decides which tools an agent may touch and
what its output must contain. It is therefore parsed with a real YAML parser
and validated strictly — an unparseable or malformed manifest raises rather
than silently degrading to an empty (and thus permissive-looking) policy set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - import guard
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required to load Yaobi skill manifests (policy files must not "
        "be parsed by a hand-rolled parser). Install with: pip install pyyaml"
    ) from exc

_ALLOWED_KEYS = {
    "skill_id", "version", "description", "allowed_roles", "allowed_tools",
    "forbidden_tools", "hard_requirements", "output_schema", "output_key", "max_tool_calls",
}


class SkillManifestError(ValueError):
    """Raised for a malformed manifest; never swallowed."""


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    version: str
    description: str = ""
    allowed_roles: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    hard_requirements: tuple[str, ...] = ()
    output_schema: str = ""
    output_key: str = ""
    max_tool_calls: int | None = None


@dataclass
class SkillRegistry:
    specs: dict[str, SkillSpec] = field(default_factory=dict)

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_file(cls, path: str | Path) -> "SkillRegistry":
        raw = Path(path).read_text(encoding="utf-8")
        try:
            document = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise SkillManifestError(f"skill manifest {path} is not valid YAML: {exc}") from exc
        return cls.from_document(document, source=str(path))

    @classmethod
    def from_document(cls, document: Any, source: str = "<memory>") -> "SkillRegistry":
        if not isinstance(document, dict) or "skills" not in document:
            raise SkillManifestError(f"skill manifest {source} must be a mapping with a top-level 'skills' key")
        entries = document.get("skills")
        if not isinstance(entries, list):
            raise SkillManifestError(f"skill manifest {source}: 'skills' must be a list")

        specs: dict[str, SkillSpec] = {}
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise SkillManifestError(f"skill manifest {source}: entry #{index} is not a mapping")
            unknown = set(entry) - _ALLOWED_KEYS
            if unknown:
                raise SkillManifestError(f"skill manifest {source}: unknown keys {sorted(unknown)} in entry #{index}")
            skill_id = entry.get("skill_id")
            if not skill_id or not isinstance(skill_id, str):
                raise SkillManifestError(f"skill manifest {source}: entry #{index} lacks a string skill_id")
            if skill_id in specs:
                raise SkillManifestError(f"skill manifest {source}: duplicate skill_id {skill_id!r}")
            specs[skill_id] = SkillSpec(
                skill_id=skill_id,
                version=str(entry.get("version", "")),
                description=str(entry.get("description", "")),
                allowed_roles=_as_tuple(entry.get("allowed_roles"), source, skill_id, "allowed_roles"),
                allowed_tools=_as_tuple(entry.get("allowed_tools"), source, skill_id, "allowed_tools"),
                forbidden_tools=_as_tuple(entry.get("forbidden_tools"), source, skill_id, "forbidden_tools"),
                hard_requirements=_as_tuple(entry.get("hard_requirements"), source, skill_id, "hard_requirements"),
                output_schema=str(entry.get("output_schema", "")),
                output_key=str(entry.get("output_key", "")),
                max_tool_calls=entry.get("max_tool_calls"),
            )
        if not specs:
            raise SkillManifestError(f"skill manifest {source} declares no skills")
        return cls(specs)

    # -------------------------------------------------------------- enforcement
    def enforce(self, skill_id: str, role: str, requested_tools: list[str]) -> tuple[bool, list[str]]:
        """Authorise tools for a skill. Unknown skills are denied, not ignored."""
        spec = self.specs.get(skill_id)
        if spec is None:
            return False, [f"skill {skill_id} not found"]
        problems: list[str] = []
        if spec.allowed_roles and role not in spec.allowed_roles:
            problems.append(f"role {role} not allowed")
        forbidden = set(spec.forbidden_tools).intersection(requested_tools)
        if forbidden:
            problems.append(f"forbidden tools requested: {sorted(forbidden)}")
        outside = set(requested_tools) - set(spec.allowed_tools)
        if outside:
            # An empty allowlist means "no tools", not "all tools".
            problems.append(f"tools outside skill allowlist: {sorted(outside)}")
        return not problems, problems

    def validate_output(self, skill_id: str, output: Any) -> tuple[bool, list[str]]:
        """Check a skill's declared ``hard_requirements`` against its output."""
        spec = self.specs.get(skill_id)
        if spec is None:
            return False, [f"skill {skill_id} not found"]
        if not spec.hard_requirements:
            return True, []
        if not isinstance(output, dict):
            return False, [f"{skill_id}: output is not a mapping"]
        # Presence, not truthiness: an empty result list is a legitimate answer,
        # a missing key is a broken contract.
        missing = [key for key in spec.hard_requirements if key not in output or output[key] is None]
        return (not missing), ([f"{skill_id}: missing required output fields {missing}"] if missing else [])

    def output_key_for(self, skill_id: str) -> str:
        spec = self.specs.get(skill_id)
        return spec.output_key if spec else ""


def _as_tuple(value: Any, source: str, skill_id: str, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise SkillManifestError(f"skill manifest {source}: {skill_id}.{key} must be a list, got a string")
    if not isinstance(value, (list, tuple)):
        raise SkillManifestError(f"skill manifest {source}: {skill_id}.{key} must be a list")
    return tuple(str(v) for v in value)
