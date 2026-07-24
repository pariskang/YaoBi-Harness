from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    version: str
    allowed_roles: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    hard_requirements: tuple[str, ...] = ()
    output_schema: str = ""

class SkillRegistry:
    """Small manifest loader/enforcer for the YAML subset used by Yaobi skills."""
    def __init__(self, specs: dict[str, SkillSpec]): self.specs = specs
    @classmethod
    def from_file(cls, path: str | Path) -> "SkillRegistry":
        text=Path(path).read_text(encoding="utf-8")
        blocks=re.split(r"\n\s*-\s+skill_id:\s*", "\n"+text)
        specs={}
        for b in blocks[1:]:
            first,*rest=b.splitlines(); data={"skill_id": first.strip()}
            current=None
            for line in rest:
                m=re.match(r"\s*([a-zA-Z_]+):\s*(.*)", line)
                if m:
                    current=m.group(1); val=m.group(2).strip()
                    if val.startswith("[") and val.endswith("]"):
                        data[current]=tuple(x.strip() for x in val.strip("[]").split(",") if x.strip())
                    else: data[current]=val
            spec=SkillSpec(skill_id=data["skill_id"], version=str(data.get("version","")), allowed_roles=tuple(data.get("allowed_roles",()) or ()), allowed_tools=tuple(data.get("allowed_tools",()) or ()), forbidden_tools=tuple(data.get("forbidden_tools",()) or ()), hard_requirements=tuple(data.get("hard_requirements",()) or ()), output_schema=str(data.get("output_schema", "")))
            specs[spec.skill_id]=spec
        return cls(specs)
    def enforce(self, skill_id: str, role: str, requested_tools: list[str]) -> tuple[bool, list[str]]:
        spec=self.specs[skill_id]; problems=[]
        if spec.allowed_roles and role not in spec.allowed_roles: problems.append(f"role {role} not allowed")
        forbidden=set(spec.forbidden_tools).intersection(requested_tools)
        if forbidden: problems.append(f"forbidden tools requested: {sorted(forbidden)}")
        outside=set(requested_tools)-set(spec.allowed_tools)
        if spec.allowed_tools and outside: problems.append(f"tools outside skill allowlist: {sorted(outside)}")
        return not problems, problems
