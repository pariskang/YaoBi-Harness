"""Turning a single expert's case corpus into structure an agent can reason over."""

from .profile import ExpertPracticeProfile, PatternProfile, build_profile, extract_pattern
from .skillgen import SKILL_ID, audit_for_phi, build_instructions, build_skill_entry, write_skill_file

__all__ = [
    "ExpertPracticeProfile", "PatternProfile", "build_profile", "extract_pattern",
    "SKILL_ID", "audit_for_phi", "build_instructions", "build_skill_entry", "write_skill_file",
]
