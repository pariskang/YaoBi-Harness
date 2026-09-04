"""Mine an expert practice profile from de-identified case records.

Retrieval alone does not transfer expertise: dumping five similar cases into an
output tells a reader nothing about what the expert *habitually does*. This
module turns the case corpus into aggregate structure — per-pattern core herb
sets, dose habits, co-prescribed western drugs, investigations ordered, and
follow-up trajectories — which a reasoning agent can actually cite.

Privacy: the profile is aggregate only. Free-text sentences never enter it;
values are short tokens, and anything appearing fewer than ``min_support``
times is dropped so a single patient's record cannot be reconstructed from it.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

#: Herbs present in at least this share of a pattern's cases form its core set.
CORE_HERB_THRESHOLD = 0.6
#: Aggregate values seen fewer times than this are withheld.
DEFAULT_MIN_SUPPORT = 2
#: Patterns with fewer cases than this are reported but flagged as unreliable.
MIN_PATTERN_CASES = 3

PATTERN_RE = re.compile(r"证型[:：]\s*([^/、，,;；\s]+)")
TOKEN_SPLIT_RE = re.compile(r"[、,，;；/|\n\r\t]+")
MAX_TOKEN_LEN = 14


def extract_pattern(diagnosis: str) -> str:
    """Pull the TCM pattern out of a diagnosis string.

    Records look like ``腰痹/证型：气血痹阻证``; fall back to the whole string so
    a differently-formatted corpus still groups sensibly.
    """
    text = str(diagnosis or "").strip()
    match = PATTERN_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.split("/")[-1].strip() or "未标注证型"


def _tokens(value: Any) -> list[str]:
    """Short, countable tokens from a free-text field — never whole sentences."""
    out = []
    for raw in TOKEN_SPLIT_RE.split(str(value or "")):
        token = raw.strip().strip("。.:：()（）")
        if token and len(token) <= MAX_TOKEN_LEN and not token.isdigit() and token != "无":
            out.append(token)
    return out


def _age(value: Any) -> int | None:
    digits = re.sub(r"[^0-9]", "", str(value or ""))
    return int(digits) if digits else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    return round(values[lo] if lo == hi else values[lo] * (hi - idx) + values[hi] * (idx - lo), 2)


@dataclass
class HerbUsage:
    herb: str
    count: int
    share: float
    median_g: float | None = None
    p25_g: float | None = None
    p75_g: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatternProfile:
    """What the expert habitually does for one TCM pattern."""

    pattern: str
    n_cases: int
    core_herbs: list[HerbUsage] = field(default_factory=list)
    adjunct_herbs: list[HerbUsage] = field(default_factory=list)
    treatment_methods: list[dict[str, Any]] = field(default_factory=list)
    western_drugs: list[dict[str, Any]] = field(default_factory=list)
    investigations: list[dict[str, Any]] = field(default_factory=list)
    comorbidities: list[dict[str, Any]] = field(default_factory=list)
    age_summary: dict[str, Any] = field(default_factory=dict)
    sex_summary: dict[str, int] = field(default_factory=dict)
    counterexample_cases: int = 0
    reliable: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["core_herbs"] = [h.to_dict() if isinstance(h, HerbUsage) else h for h in self.core_herbs]
        data["adjunct_herbs"] = [h.to_dict() if isinstance(h, HerbUsage) else h for h in self.adjunct_herbs]
        return data

    def summary_line(self, include_doses: bool = False) -> str:
        """One-line summary.

        Doses are excluded by default: reasoning agents are forbidden from
        emitting gram values, so putting medians in their prompt would both
        contradict that rule and invite an output the guard has to reject.
        Dose statistics reach only the dose pipeline, via
        ``herb_dose_distribution``.
        """
        core = "、".join(
            f"{h.herb}({h.median_g}g)" if include_doses and h.median_g else h.herb
            for h in self.core_herbs[:10]
        )
        return (
            f"{self.pattern}｜{self.n_cases}例"
            f"{'' if self.reliable else '（样本量不足，仅供参考）'}"
            f"｜核心药: {core or '未形成稳定核心'}"
        )


@dataclass
class ExpertPracticeProfile:
    """Corpus-level view of a single expert's habits."""

    total_cases: int = 0
    patterns: dict[str, PatternProfile] = field(default_factory=dict)
    followup: dict[str, Any] = field(default_factory=dict)
    corpus_top_herbs: list[dict[str, Any]] = field(default_factory=list)
    min_support: int = DEFAULT_MIN_SUPPORT
    limitations: str = (
        "单一专家的回顾性经验汇总，非因果疗效证据；未经随机对照验证，"
        "不能替代指南，且必须由医师结合具体患者判断。"
    )

    def pattern_for(self, pattern: str) -> PatternProfile | None:
        if not pattern:
            return None
        if pattern in self.patterns:
            return self.patterns[pattern]
        # Tolerate "腰痹/证型：气血痹阻证" vs "气血痹阻证" and partial matches.
        key = extract_pattern(pattern)
        if key in self.patterns:
            return self.patterns[key]
        return next((p for name, p in self.patterns.items() if key and (key in name or name in key)), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "patterns": {name: p.to_dict() for name, p in self.patterns.items()},
            "followup": self.followup,
            "corpus_top_herbs": self.corpus_top_herbs,
            "min_support": self.min_support,
            "limitations": self.limitations,
        }

    def brief(self, pattern: str | None = None, max_patterns: int = 6, include_doses: bool = False) -> dict[str, Any]:
        """Compact view suitable for an LLM context window.

        ``include_doses`` stays off for anything a reasoning agent will read;
        see :meth:`PatternProfile.summary_line`.
        """
        chosen = [self.pattern_for(pattern)] if pattern else []
        chosen = [p for p in chosen if p] or sorted(
            self.patterns.values(), key=lambda p: -p.n_cases
        )[:max_patterns]
        return {
            "total_cases": self.total_cases,
            "followup": self.followup,
            "limitations": self.limitations,
            "patterns": [
                {
                    "pattern": p.pattern,
                    "n_cases": p.n_cases,
                    "reliable": p.reliable,
                    "core_herbs": [
                        h.to_dict() if include_doses
                        else {"herb": h.herb, "count": h.count, "share": h.share}
                        for h in p.core_herbs[:12]
                    ],
                    "adjunct_herbs": [h.herb for h in p.adjunct_herbs[:10]],
                    "treatment_methods": p.treatment_methods[:5],
                    "western_drugs": p.western_drugs[:5],
                    "investigations": p.investigations[:5],
                    "counterexample_cases": p.counterexample_cases,
                }
                for p in chosen
            ],
        }


COUNTEREXAMPLE_MARKERS = ("加重", "无效", "未缓解", "复发", "疗效不佳", "反复")


def _counted(values: Iterable[str], min_support: int, limit: int = 12) -> list[dict[str, Any]]:
    counter = Counter(values)
    return [
        {"value": value, "count": count}
        for value, count in counter.most_common(limit)
        if count >= min_support
    ]


def build_profile(records: list[dict[str, Any]], min_support: int = DEFAULT_MIN_SUPPORT) -> ExpertPracticeProfile:
    """Aggregate de-identified records into an :class:`ExpertPracticeProfile`.

    ``records`` must already be de-identified (i.e. produced by
    :class:`~yaobi_harness.tools.ExpertCaseStore`); this function never sees or
    stores direct identifiers.
    """
    profile = ExpertPracticeProfile(total_cases=len(records), min_support=min_support)
    by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_pattern[extract_pattern(record.get("中医诊断", ""))].append(record)

    corpus_herbs: Counter = Counter()

    for pattern, group in by_pattern.items():
        n = len(group)
        herb_counts: Counter = Counter()
        herb_doses: dict[str, list[float]] = defaultdict(list)
        for record in group:
            seen = set()
            for item in record.get("herbs", []) or []:
                name = item.get("herb_name")
                if not name:
                    continue
                if name not in seen:
                    herb_counts[name] += 1
                    seen.add(name)
                if item.get("dose_g"):
                    herb_doses[name].append(float(item["dose_g"]))
            corpus_herbs.update(seen)

        usages = []
        for herb, count in herb_counts.most_common():
            if count < min_support:
                continue
            doses = herb_doses.get(herb, [])
            usages.append(HerbUsage(
                herb=herb, count=count, share=round(count / n, 3),
                median_g=round(statistics.median(doses), 2) if doses else None,
                p25_g=_percentile(doses, 0.25), p75_g=_percentile(doses, 0.75),
            ))

        ages = [a for a in (_age(r.get("年龄")) for r in group) if a is not None]
        blob = lambda key: [t for r in group for t in _tokens(r.get(key, ""))]  # noqa: E731

        profile.patterns[pattern] = PatternProfile(
            pattern=pattern,
            n_cases=n,
            core_herbs=[u for u in usages if u.share >= CORE_HERB_THRESHOLD],
            adjunct_herbs=[u for u in usages if u.share < CORE_HERB_THRESHOLD],
            treatment_methods=_counted(blob("治疗方法") + blob("治疗"), min_support),
            western_drugs=_counted(blob("西药"), min_support),
            investigations=_counted(blob("辅助检查"), min_support),
            comorbidities=_counted(blob("既往史") + blob("手术史"), min_support),
            age_summary={
                "median": round(statistics.median(ages), 1) if ages else None,
                "p25": _percentile([float(a) for a in ages], 0.25),
                "p75": _percentile([float(a) for a in ages], 0.75),
                "n": len(ages),
            },
            sex_summary=dict(Counter(str(r.get("性别", "")).strip() for r in group if r.get("性别"))),
            counterexample_cases=sum(
                1 for r in group
                if any(m in " ".join(str(v) for v in r.values()) for m in COUNTEREXAMPLE_MARKERS)
            ),
            reliable=n >= MIN_PATTERN_CASES,
        )

    profile.corpus_top_herbs = [
        {"herb": herb, "count": count}
        for herb, count in corpus_herbs.most_common(20)
        if count >= min_support
    ]
    profile.followup = _followup_summary(records)
    return profile


def _followup_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """How much longitudinal signal the corpus actually carries."""
    visits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pid = record.get("research_patient_id")
        if pid:
            visits[pid].append(record)
    repeat = {pid: rows for pid, rows in visits.items() if len(rows) > 1}
    counts = [len(rows) for rows in visits.values()]
    changed = 0
    for rows in repeat.values():
        patterns = [extract_pattern(r.get("中医诊断", "")) for r in rows]
        if len(set(patterns)) > 1:
            changed += 1
    return {
        "distinct_patients": len(visits),
        "patients_with_followup": len(repeat),
        "median_visits": round(statistics.median(counts), 1) if counts else 0,
        "max_visits": max(counts) if counts else 0,
        "patients_whose_pattern_changed": changed,
        "note": "复诊轨迹依赖稳定的假名 ID；未设置 YAOBI_DEID_KEY 时该统计不可用",
    }
