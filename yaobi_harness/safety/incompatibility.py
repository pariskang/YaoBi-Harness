"""Classical Chinese-medicine combination contraindications.

Covers 十八反 (eighteen antagonisms), 十九畏 (nineteen incompatibilities) and
pregnancy-contraindicated herbs. These are absolute, well-established rules: a
formula that violates them must never leave the system as a dosed draft, no
matter how much expert-case support the individual doses have.

Matching is alias-aware because source records use processed names
(``制川乌``、``法半夏``、``全瓜蒌``) rather than canonical ones.
"""

from __future__ import annotations

from typing import Any, Iterable

#: canonical herb -> spellings that appear in real prescriptions
HERB_ALIASES: dict[str, tuple[str, ...]] = {
    "甘草": ("甘草", "生甘草", "炙甘草", "甘草片", "粉甘草"),
    "甘遂": ("甘遂", "醋甘遂"),
    "大戟": ("大戟", "京大戟", "红大戟", "醋大戟"),
    "海藻": ("海藻",),
    "芫花": ("芫花", "醋芫花"),
    "乌头": ("川乌", "制川乌", "草乌", "制草乌", "附子", "附片", "黑顺片", "白附片", "淡附片", "生川乌", "生草乌"),
    "半夏": ("半夏", "法半夏", "姜半夏", "清半夏", "生半夏"),
    "瓜蒌": ("瓜蒌", "全瓜蒌", "瓜蒌子", "瓜蒌皮", "瓜蒌仁", "天花粉"),
    "贝母": ("贝母", "川贝母", "浙贝母", "平贝母", "土贝母"),
    "白蔹": ("白蔹",),
    "白及": ("白及", "白芨"),
    "藜芦": ("藜芦",),
    "人参": ("人参", "红参", "生晒参", "党参"),
    "沙参": ("沙参", "北沙参", "南沙参"),
    "丹参": ("丹参",),
    "玄参": ("玄参",),
    "苦参": ("苦参",),
    "细辛": ("细辛",),
    "芍药": ("芍药", "白芍", "赤芍", "麸白芍", "炒白芍"),
    "硫黄": ("硫黄", "硫磺"),
    "朴硝": ("朴硝", "芒硝", "玄明粉", "牙硝"),
    "水银": ("水银",),
    "砒霜": ("砒霜", "砒石", "红砒", "白砒"),
    "狼毒": ("狼毒",),
    "密陀僧": ("密陀僧",),
    "巴豆": ("巴豆", "巴豆霜"),
    "牵牛": ("牵牛", "牵牛子", "黑丑", "白丑", "二丑"),
    "丁香": ("丁香", "母丁香", "公丁香"),
    "郁金": ("郁金", "川郁金", "广郁金"),
    "犀角": ("犀角", "水牛角"),
    "三棱": ("三棱", "醋三棱"),
    "肉桂": ("肉桂", "官桂", "桂皮"),
    "赤石脂": ("赤石脂",),
    "五灵脂": ("五灵脂", "醋五灵脂"),
}

#: 十八反 — pairs that must never appear together.
EIGHTEEN_ANTAGONISMS: list[tuple[str, str]] = [
    ("甘草", "甘遂"), ("甘草", "大戟"), ("甘草", "海藻"), ("甘草", "芫花"),
    ("乌头", "半夏"), ("乌头", "瓜蒌"), ("乌头", "贝母"), ("乌头", "白蔹"), ("乌头", "白及"),
    ("藜芦", "人参"), ("藜芦", "沙参"), ("藜芦", "丹参"), ("藜芦", "玄参"),
    ("藜芦", "苦参"), ("藜芦", "细辛"), ("藜芦", "芍药"),
]

#: 十九畏 — pairs that must not be combined without specialist justification.
NINETEEN_INCOMPATIBILITIES: list[tuple[str, str]] = [
    ("硫黄", "朴硝"), ("水银", "砒霜"), ("狼毒", "密陀僧"), ("巴豆", "牵牛"),
    ("丁香", "郁金"), ("乌头", "犀角"), ("朴硝", "三棱"), ("肉桂", "赤石脂"), ("人参", "五灵脂"),
]

#: Herbs that must never be drafted for a pregnant patient.
PREGNANCY_CONTRAINDICATED: dict[str, str] = {
    "乌头": "毒性/堕胎风险", "巴豆": "峻下逐水", "甘遂": "峻下逐水", "大戟": "峻下逐水",
    "芫花": "峻下逐水", "牵牛": "峻下逐水", "三棱": "破血消癥", "水银": "重金属毒性",
    "砒霜": "剧毒", "麝香": "活血开窍堕胎", "水蛭": "破血逐瘀", "虻虫": "破血逐瘀",
    "莪术": "破血消癥", "桃仁": "活血祛瘀慎用", "红花": "活血祛瘀慎用", "大黄": "攻下慎用",
}
PREGNANCY_ALIASES: dict[str, tuple[str, ...]] = {
    **HERB_ALIASES,
    "麝香": ("麝香", "人工麝香"),
    "水蛭": ("水蛭",),
    "虻虫": ("虻虫",),
    "莪术": ("莪术", "醋莪术"),
    "桃仁": ("桃仁", "燀山桃仁", "炒桃仁"),
    "红花": ("红花", "藏红花", "西红花"),
    "大黄": ("大黄", "生大黄", "熟大黄", "酒大黄"),
}

#: Potent herbs that always need a named specialist review before dosing.
RISK_HERBS: frozenset[str] = frozenset(
    {"附片", "附子", "制川乌", "川乌", "草乌", "制草乌", "细辛", "麻黄", "全蝎", "蜈蚣", "生半夏", "马钱子", "雷公藤"}
)


def _canonical(name: str, alias_table: dict[str, tuple[str, ...]] | None = None) -> set[str]:
    """Return the canonical group names that ``name`` belongs to."""
    table = alias_table or HERB_ALIASES
    hit = set()
    clean = (name or "").strip()
    for canon, aliases in table.items():
        if clean in aliases or clean == canon:
            hit.add(canon)
    return hit


def canonical_groups(herbs: Iterable[str], alias_table: dict[str, tuple[str, ...]] | None = None) -> dict[str, list[str]]:
    """Map canonical group -> the raw herb names in ``herbs`` that belong to it."""
    groups: dict[str, list[str]] = {}
    for herb in herbs:
        for canon in _canonical(herb, alias_table):
            groups.setdefault(canon, []).append(herb)
    return groups


def check_combination(herbs: Iterable[str]) -> list[dict[str, Any]]:
    """Return every 十八反/十九畏 violation present in ``herbs``."""
    names = [h for h in herbs if h]
    groups = canonical_groups(names)
    violations: list[dict[str, Any]] = []
    for rule_name, pairs, severity in (
        ("十八反", EIGHTEEN_ANTAGONISMS, "absolute"),
        ("十九畏", NINETEEN_INCOMPATIBILITIES, "absolute"),
    ):
        for left, right in pairs:
            if left in groups and right in groups:
                violations.append(
                    {
                        "rule": rule_name,
                        "pair": [left, right],
                        "matched_herbs": sorted(set(groups[left]) | set(groups[right])),
                        "severity": severity,
                        "detail": f"{rule_name}: {left}与{right}不可同用",
                    }
                )
    return violations


def check_pregnancy(herbs: Iterable[str]) -> list[dict[str, Any]]:
    """Return pregnancy-contraindicated herbs present in ``herbs``."""
    groups = canonical_groups(herbs, PREGNANCY_ALIASES)
    return [
        {
            "rule": "妊娠禁忌",
            "herb_group": canon,
            "matched_herbs": sorted(set(raw)),
            "severity": "absolute",
            "detail": f"妊娠禁忌/慎用: {canon}({PREGNANCY_CONTRAINDICATED[canon]})",
        }
        for canon, raw in sorted(groups.items())
        if canon in PREGNANCY_CONTRAINDICATED
    ]


def risk_herbs_in(herbs: Iterable[str]) -> list[str]:
    return sorted({h for h in herbs if h in RISK_HERBS})
