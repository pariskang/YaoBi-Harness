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
    "instructions", "autonomous",
    # Fields a discovered SKILL.md may add. They are policy-neutral metadata or
    # *narrowing* controls; none of them can widen the tool grant above.
    "when_to_use", "source", "model", "persona", "consult_mode", "inputs", "outputs",
    "vision",
}

#: Consult modes, from most to least restrictive. A subagent runs under one of
#: these on top of its skill allowlist; the mode can only ever remove tools.
CONSULT_MODES = ("evidence_only", "screening", "advisory")


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
    #: Procedure text loaded into the agent's system prompt. This is what makes
    #: a skill something the model *follows*, not just an allowlist it is bound by.
    instructions: str = ""
    #: Whether this skill may be executed as a model-driven tool-calling loop.
    autonomous: bool = False
    #: Trigger phrasing, kept apart from ``description`` so a planner model can
    #: read "when would I reach for this" without the full procedure.
    when_to_use: str = ""
    #: Where this spec came from — ``manifest.yaml`` or a ``SKILL.md`` path.
    #: Surfaced in the console so an operator can tell a shipped policy from a
    #: site override at a glance.
    source: str = "manifest"
    #: Per-skill model override. The vision skill needs a multimodal model even
    #: when the run's default model is text-only.
    model: str = ""
    #: Behavioural overlay for a consult subagent (see :mod:`yaobi_harness.agent.panel`).
    persona: str = ""
    #: Coarse capability filter applied *on top of* ``allowed_tools``.
    consult_mode: str = ""
    #: Declared I/O contract, so a caller knows what to supply and expect back.
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    #: True when this skill consumes images and therefore needs a vision client.
    vision: bool = False

    def effective_tools(self, consult_mode: str | None = None) -> tuple[str, ...]:
        """Tools this skill really grants, after the consult mode narrows them.

        Narrowing only. A mode that named a tool outside ``allowed_tools`` would
        be a widening grant, so the intersection is taken rather than the union.
        """
        granted = tuple(t for t in self.allowed_tools if t not in self.forbidden_tools)
        mode = consult_mode or self.consult_mode
        if not mode:
            return granted
        keep = _CONSULT_MODE_TOOLS.get(mode)
        if keep is None:
            return granted
        return tuple(t for t in granted if t in keep)


#: Which tools each consult mode leaves reachable. ``advisory`` is the widest a
#: subagent can be, and it still excludes every prescriptive tool: no subagent
#: may design a formula, pull a dose distribution or submit a signature, no
#: matter what its skill claims.
_PRESCRIPTIVE = {"formula_composition_search", "herb_dose_distribution", "physician_review_submit"}
_EVIDENCE_TOOLS = {
    "clinical_guideline_search", "tcm_pattern_knowledge_search", "similar_case_search",
    "counterexample_case_search", "patient_timeline_search", "expert_practice_profile",
    "drug_label_lookup", "drug_normalize", "interview_axis_lookup",
}
_SCREENING_TOOLS = _EVIDENCE_TOOLS | {
    "red_flag_evidence_search", "drug_interaction_check", "interaction_check",
    "special_population_check", "emergency_resource_lookup", "medical_image_read",
}
_CONSULT_MODE_TOOLS: dict[str, set[str]] = {
    "evidence_only": set(_EVIDENCE_TOOLS),
    "screening": set(_SCREENING_TOOLS),
    "advisory": set(_SCREENING_TOOLS) | {"ask_patient"},
}


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
    def from_entries(cls, entries: list[dict[str, Any]], source: str = "<skills>") -> "SkillRegistry":
        """Build a registry from already-parsed entries (e.g. ``SKILL.md`` files)."""
        return cls.from_document({"skills": entries}, source=source)

    @classmethod
    def discover(
        cls,
        manifest: str | Path | None = None,
        *,
        extra_roots: list[str | Path] | None = None,
        working_directory: str | Path | None = None,
    ) -> "SkillRegistry":
        """Load ``manifest.yaml`` then overlay every discovered ``SKILL.md``.

        Precedence is the reason this exists: a site can drop its own
        ``SKILL.md`` into ``./.yaobi/skills/`` and replace a shipped procedure
        without editing the package. Roots are applied in ascending precedence,
        so the last one wins.
        """
        from .discovery import collect_entries, default_skill_roots

        registry = cls()
        if manifest and Path(manifest).exists():
            registry = cls.from_file(manifest)

        roots: list[str | Path] = list(default_skill_roots(working_directory))
        roots += list(extra_roots or [])
        entries = collect_entries(roots)
        if entries:
            registry = registry.merge(cls.from_entries(entries))
        return registry

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
                instructions=str(entry.get("instructions", "") or ""),
                autonomous=bool(entry.get("autonomous", False)),
                when_to_use=str(entry.get("when_to_use", "") or ""),
                source=str(entry.get("source", "") or source),
                model=str(entry.get("model", "") or ""),
                persona=str(entry.get("persona", "") or ""),
                consult_mode=_consult_mode(entry.get("consult_mode"), source, skill_id),
                inputs=_as_tuple(entry.get("inputs"), source, skill_id, "inputs"),
                outputs=_as_tuple(entry.get("outputs"), source, skill_id, "outputs"),
                vision=bool(entry.get("vision", False)),
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

    def catalog(self, role: str | None = None) -> list[dict[str, Any]]:
        """What each skill does, for a planner model to choose between.

        Only name, purpose and tool surface are exposed — never the full
        instruction text, which can be long and is loaded per agent instead.
        """
        return [
            {
                "skill_id": spec.skill_id,
                "description": spec.description,
                "when_to_use": spec.when_to_use,
                "tools": list(spec.allowed_tools),
                "autonomous": spec.autonomous,
                "roles": list(spec.allowed_roles) or ["patient", "physician", "researcher"],
                "source": "manifest" if spec.source == "manifest" else "SKILL.md",
            }
            for spec in self.specs.values()
            if not role or not spec.allowed_roles or role in spec.allowed_roles
        ]

    def merge(self, other: "SkillRegistry") -> "SkillRegistry":
        """Overlay another registry (e.g. a generated expert skill) onto this one."""
        merged = dict(self.specs)
        merged.update(other.specs)
        return SkillRegistry(merged)

    def output_key_for(self, skill_id: str) -> str:
        spec = self.specs.get(skill_id)
        return spec.output_key if spec else ""


def _consult_mode(value: Any, source: str, skill_id: str) -> str:
    """Validate a declared consult mode. An unknown mode is a policy bug.

    Defaulting an unrecognised mode to "no filter" would turn a typo into a
    silently wider grant, which is the one failure mode a policy file must not
    have.
    """
    if value in (None, ""):
        return ""
    mode = str(value).strip()
    if mode not in CONSULT_MODES:
        raise SkillManifestError(
            f"skill {source}: {skill_id}.consult_mode {mode!r} is not one of {list(CONSULT_MODES)}"
        )
    return mode


def _as_tuple(value: Any, source: str, skill_id: str, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise SkillManifestError(f"skill manifest {source}: {skill_id}.{key} must be a list, got a string")
    if not isinstance(value, (list, tuple)):
        raise SkillManifestError(f"skill manifest {source}: {skill_id}.{key} must be a list")
    return tuple(str(v) for v in value)
