"""Turn a mined expert profile into a loadable skill.

A skill in this harness is two things at once: a **capability grant** the broker
enforces, and a **procedure description** the model reads. Generating one from
the expert corpus is therefore how tacit expertise becomes something an agent
can both cite and be constrained by.

The generated file contains aggregate statistics only. Free text from any
individual record never reaches it, and values below the profile's
``min_support`` are already withheld upstream — so the skill can be reviewed,
version-controlled and shared inside the institution without carrying PHI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .profile import ExpertPracticeProfile

#: The generated skill replaces the built-in one by id, so regenerating it
#: upgrades the agent that already uses it rather than adding a parallel path.
SKILL_ID = "yaobi.expert_case_reasoning"
SKILL_VERSION = "1.0.0"


def build_instructions(profile: ExpertPracticeProfile, max_patterns: int = 8) -> str:
    """Render the profile as the instruction block an agent reads."""
    lines = [
        "你在使用一位专家的回顾性经验库。以下统计来自该专家已脱敏的历史病例，是**经验倾向**，不是疗效证据。",
        "",
        f"语料规模：{profile.total_cases} 例；"
        f"{profile.followup.get('distinct_patients', 0)} 名假名患者，"
        f"其中 {profile.followup.get('patients_with_followup', 0)} 名有复诊记录。",
        "",
        "使用规则：",
        "1. 引用经验时必须说明例数，例数 < 3 的证型要明确标注为不可靠。",
        "2. 经验与指南冲突时以指南为准，并把冲突显式写出来。",
        "3. 不要从这些统计直接推导克数；剂量由独立的剂量安全链路处理。",
        "4. 反例（加重/无效/复发）与正例同等重要，必须一并呈现。",
        "",
        "## 各证型经验概览",
    ]
    patterns = sorted(profile.patterns.values(), key=lambda p: -p.n_cases)[:max_patterns]
    for pattern in patterns:
        lines.append("")
        lines.append(f"### {pattern.summary_line()}")
        if pattern.adjunct_herbs:
            lines.append("- 随证加减常见药：" + "、".join(h.herb for h in pattern.adjunct_herbs[:10]))
        for label, rows in (
            ("常用治法", pattern.treatment_methods),
            ("合并西药", pattern.western_drugs),
            ("常做检查", pattern.investigations),
            ("常见合并症/术史", pattern.comorbidities),
        ):
            if rows:
                lines.append(f"- {label}：" + "、".join(f"{r['value']}({r['count']})" for r in rows[:6]))
        if pattern.age_summary.get("median") is not None:
            lines.append(f"- 年龄中位数：{pattern.age_summary['median']} 岁")
        if pattern.counterexample_cases:
            lines.append(f"- 语料中含加重/无效/复发描述的病例：{pattern.counterexample_cases} 例")
    lines += ["", "## 局限", profile.limitations]
    return "\n".join(lines)


def build_skill_entry(
    profile: ExpertPracticeProfile,
    *,
    skill_id: str = SKILL_ID,
    version: str = SKILL_VERSION,
    source_label: str = "授权专家病例库",
) -> dict[str, Any]:
    """Build the manifest entry for the generated expert skill."""
    return {
        "skill_id": skill_id,
        "version": version,
        "description": (
            f"{source_label}经验：{profile.total_cases} 例、"
            f"{len(profile.patterns)} 个证型的核心用药、治法与随访倾向"
        ),
        "allowed_tools": [
            "expert_practice_profile",
            "similar_case_search",
            "counterexample_case_search",
            "patient_timeline_search",
        ],
        "forbidden_tools": ["herb_dose_distribution", "physician_review_submit"],
        "output_schema": "ExpertCaseEvidence",
        "output_key": "expert_cases",
        "hard_requirements": ["similar", "counterexamples", "limitation"],
        "autonomous": True,
        "instructions": build_instructions(profile),
    }


def write_skill_file(
    profile: ExpertPracticeProfile,
    path: str | Path,
    *,
    skill_id: str = SKILL_ID,
    source_label: str = "授权专家病例库",
) -> Path:
    """Write a standalone skill manifest fragment derived from the corpus."""
    entry = build_skill_entry(profile, skill_id=skill_id, source_label=source_label)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# 由 `python -m yaobi_harness skill build-expert` 从授权专家病例库生成。\n"
        "# 内容为聚合统计，不含任何个体自由文本；请由本机构医师复核后启用。\n"
        "# 合并方式：`--merge` 写回主 manifest，或用 --skill-manifest 指向合并后的文件。\n"
    )
    out.write_text(header + yaml.safe_dump({"skills": [entry]}, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    return out


def merge_into_manifest(skill_entry: dict[str, Any], manifest_path: str | Path) -> Path:
    """Insert or replace the generated skill inside an existing manifest."""
    path = Path(manifest_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    skills = [s for s in (document.get("skills") or []) if s.get("skill_id") != skill_entry["skill_id"]]
    skills.append(skill_entry)
    document["skills"] = skills
    path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def audit_for_phi(skill_entry: dict[str, Any]) -> list[str]:
    """Cheap safety net: flag anything that looks like leaked free text.

    The generator only emits counted short tokens, so a long run of prose in the
    instructions means an upstream change broke that guarantee.
    """
    import re

    problems = []
    text = skill_entry.get("instructions", "")
    for line in text.splitlines():
        stripped = line.strip("-# ").strip()
        if len(stripped) > 160:
            problems.append(f"instruction line too long, possible free text: {stripped[:60]}…")
    for pattern, label in (
        (r"1[3-9]\d{9}", "phone number"),
        (r"\b\d{17}[\dXx]\b", "national id"),
        (r"\b\d{15}\b", "legacy national id"),
        (r"[\w.+-]+@[\w-]+\.[\w.]+", "e-mail"),
    ):
        if re.search(pattern, text):
            problems.append(f"possible {label} in generated skill")
    return problems
