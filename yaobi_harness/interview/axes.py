"""History-taking axes: 十问歌 plus orthopaedic specialty enquiry.

An *axis* is one line of clinical enquiry — "寒热", "神经根定位", "跌倒与骨折风险".
Each carries a tier, the facts answering it closes, the roles that can actually
answer it, and a bank of professionally-worded probes.

The division of labour with the model is the whole point of this module:

============================  ==============================================
Decided by rule (here)        Decided by the model
============================  ==============================================
which axes are *required*     what to ask, in what words, in what order
which axes are *relevant*     how deeply to probe one axis
that a RED_FLAG axis is       which probe to reuse verbatim and which to
never skippable               rewrite for this particular patient
============================  ==============================================

So the model drives the interview but cannot decide that cauda-equina screening
is unnecessary — the same containment the rest of the harness applies to red
flags, applied to questioning. A model that goes quiet still leaves the required
axes open, and an open required axis blocks release.

The probes are a *bank*, not a script. They exist so a model with no useful idea
still asks something clinically sound, and so the deterministic fallback path
(no model configured at all) asks real questions rather than nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

#: Tier drives escalation, not just presentation.
#:
#: ``RED_FLAG``  — an unanswered axis blocks any prescriptive release.
#: ``CORE``      — needed for any orthopaedic assessment at all.
#: ``SPECIALTY`` — needed for the specific presentation (radicular, claudicant,
#:                 osteoporotic, post-traumatic …).
#: ``TCM``       — 四诊 material; needed before a pattern-based formula.
#: ``CONTEXT``   — function, occupation, psychosocial. Shapes the plan, never gates it.
TIERS = ("RED_FLAG", "CORE", "SPECIALTY", "TCM", "CONTEXT")

Predicate = Callable[[dict[str, Any], str], bool]


@dataclass(frozen=True)
class Axis:
    axis_id: str
    label: str
    tier: str
    #: Where the axis comes from, shown in the console so a clinician can see
    #: that the 十问歌 coverage is real rather than decorative.
    tradition: str
    #: Fact keys that, once present, close this axis.
    closes: tuple[str, ...] = ()
    #: Professionally-worded probes. First entry is the plainest phrasing.
    probes: tuple[str, ...] = ()
    #: What the answer is used for — fed to the model so it can explain itself.
    rationale: str = ""
    answerable_by: tuple[str, ...] = ("patient", "physician")
    #: Applicability test; ``None`` means always relevant.
    applies_when: Predicate | None = None

    def relevant(self, facts: dict[str, Any], complaint: str) -> bool:
        if self.applies_when is None:
            return True
        try:
            return bool(self.applies_when(facts, complaint or ""))
        except Exception:  # noqa: BLE001 - a broken predicate must not hide an axis
            return True

    def satisfied(self, facts: dict[str, Any]) -> bool:
        """Closed only when *every* declared fact is present.

        ``all`` rather than ``any``: an axis that lists two facts needs both, or
        "已问过大便、没问小便" would count as screened.
        """
        if not self.closes:
            return False
        known = dict(facts)
        known.update(facts.get("special_population") or {})
        return all(known.get(key) is not None for key in self.closes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id, "label": self.label, "tier": self.tier,
            "tradition": self.tradition, "closes": list(self.closes),
            "probes": list(self.probes), "rationale": self.rationale,
            "answerable_by": list(self.answerable_by),
        }


# --------------------------------------------------------------- predicates
def _text(facts: dict[str, Any], complaint: str) -> str:
    """Everything the patient has said plus the free-text facts, lowercased."""
    parts = [complaint]
    for key in ("pain_location", "radiation", "neuro_symptoms", "onset", "four_diagnoses", "location"):
        value = facts.get(key)
        if isinstance(value, str):
            parts.append(value)
    return " ".join(parts).lower()


def _age(facts: dict[str, Any]) -> int | None:
    known = dict(facts)
    known.update(facts.get("special_population") or {})
    age = known.get("age")
    return age if isinstance(age, int) else None


def _mentions(*terms: str) -> Predicate:
    return lambda facts, complaint: any(t in _text(facts, complaint) for t in terms)


def _spine_or_limb(facts: dict[str, Any], complaint: str) -> bool:
    return any(
        t in _text(facts, complaint)
        for t in ("腰", "颈", "背", "腿", "臀", "肩", "膝", "髋", "踝", "肘", "腕", "手", "足", "关节", "骨", "痛", "麻")
    )


def _older_or_fragile(facts: dict[str, Any], complaint: str) -> bool:
    age = _age(facts)
    if age is not None and age >= 50:
        return True
    if any(c in (facts.get("conditions") or []) for c in ("elderly", "osteoporosis", "postmenopausal")):
        return True
    return any(t in _text(facts, complaint) for t in ("骨质疏松", "骨密度", "压缩", "驼背", "变矮", "身高", "绝经", "脆性骨折"))


def _female_reproductive(facts: dict[str, Any], complaint: str) -> bool:
    """The 妇女尤必问经期 line of the 十问歌, kept clinically bounded.

    Asked when the patient is recorded female and of plausible reproductive age,
    or when pregnancy status is still unknown and a dose decision may follow —
    pregnancy contraindications are the reason this matters here.
    """
    known = dict(facts)
    known.update(facts.get("special_population") or {})
    sex = str(facts.get("sex") or "")
    female = any(t in sex for t in ("女", "female", "F", "f"))
    age = _age(facts)
    if female and (age is None or 10 <= age <= 60):
        return True
    return female and known.get("pregnancy") is None


# ------------------------------------------------------------------- the axes
#
# 十问歌（张景岳《景岳全书·十问篇》，陈修园《医学实在易》传本）：
#   一问寒热二问汗，三问头身四问便，五问饮食六胸腹，
#   七聋八渴俱当辨，九问旧病十问因，再兼服药参机变，
#   妇女尤必问经期，迟速闭崩皆可见。
#
# Each of those lines is an axis below, tagged ``十问歌``. They are not folklore
# decoration: 寒热/汗/二便/饮食 are how a TCM pattern is actually differentiated,
# and 旧病/病因/服药 are ordinary history-taking under older names.
AXES: tuple[Axis, ...] = (
    # ---------------------------------------------------------- RED_FLAG tier
    Axis(
        "cauda_equina", "大小便与鞍区感觉", "RED_FLAG", "红旗",
        closes=("bowel_bladder",),
        probes=(
            "最近有没有小便解不出来、要用力才能解，或者尿了不知道、漏尿？",
            "大便有没有失禁或明显控制不住？",
            "坐着或骑车时接触座垫的那一圈（会阴、肛周）有没有发麻、没知觉？",
            "这些变化是这几天新出现的，还是一直都有？",
        ),
        rationale="马尾神经综合征是需要数小时内手术减压的急症，漏诊代价最高。",
    ),
    Axis(
        "progressive_neuro", "神经功能是否进行性恶化", "RED_FLAG", "红旗",
        closes=("neuro_symptoms",),
        probes=(
            "腿或脚有没有越来越没力气？比如抬不起脚尖、走路拖步、上楼费劲？",
            "麻木的范围是在扩大还是在缩小？",
            "这种无力是几小时内出现的，还是几周慢慢来的？",
        ),
        rationale="进行性运动功能缺损意味着神经正在受损，观察等待会造成不可逆后果。",
    ),
    Axis(
        "infection_tumor_fracture", "感染、肿瘤与外伤线索", "RED_FLAG", "红旗",
        closes=("fever_trauma_tumor",),
        probes=(
            "近期有没有发烧、寒战、盗汗？",
            "最近半年体重有没有不明原因下降？以前有没有查出过肿瘤？",
            "有没有摔倒、扭伤、车祸，或者搬重物之后突然剧痛？",
            "有没有静脉吸毒、长期用激素、免疫抑制剂，或近期做过穿刺、手术？",
        ),
        rationale="脊柱感染、转移瘤与病理性骨折的首诊线索几乎全在病史里，影像常滞后。",
    ),
    Axis(
        "night_rest_pain", "夜间痛与休息不缓解", "RED_FLAG", "红旗",
        closes=("night_pain",),
        probes=(
            "晚上会不会痛到醒过来？醒了之后换个姿势能缓解吗？",
            "完全躺着不动的时候还痛吗？",
        ),
        rationale="休息与夜间不缓解的疼痛把机械性病因排在后面，把肿瘤、感染、炎症排在前面。",
    ),
    Axis(
        "vascular_limb", "肢体血供与肿胀", "RED_FLAG", "红旗",
        closes=("limb_vascular",),
        probes=(
            "小腿或整条腿有没有突然肿起来、发热发红，或者一侧比另一侧明显粗？",
            "有没有胸闷、气短、咳血？",
            "脚趾有没有发凉、发紫、颜色不对？",
        ),
        rationale="深静脉血栓/肺栓塞与骨筋膜室综合征都以肢体表现起病，而首诊常按骨科疼痛处理。",
        applies_when=_mentions("腿", "小腿", "肿", "足", "踝", "膝", "石膏", "术后", "卧床", "骨折"),
    ),
    # -------------------------------------------------------------- CORE tier
    Axis(
        "pain_character", "疼痛性质与起病（OPQRST）", "CORE", "骨科专科",
        closes=("onset", "pain_location"),
        probes=(
            "这次疼是什么时候开始的？突然发生还是慢慢加重？",
            "最痛的点具体在哪里？能用一根手指指出来吗？",
            "是什么样的痛——酸胀、刀割、电击、烧灼、还是说不清的钝痛？",
            "什么动作或姿势会加重？什么姿势能缓解？",
            "一天里什么时候最重，什么时候最轻？",
        ),
        rationale="性质与时间节律是机械性、炎症性、神经性、内脏源性疼痛的第一层分流。",
    ),
    Axis(
        "radiation_dermatome", "放射范围与节段定位", "CORE", "骨科专科",
        closes=("radiation",),
        probes=(
            "痛会不会往下窜？窜到臀部、大腿后侧、小腿，还是一直到脚？",
            "麻的地方是脚背、脚底，还是脚外侧？（这三处对应不同节段）",
            "咳嗽、打喷嚏、用力大便时会不会加重？",
        ),
        rationale="放射的止点与麻木分布决定怀疑哪个节段（L4 小腿前内/L5 足背/S1 足底外侧）。",
    ),
    Axis(
        "medication_history", "当前用药与过敏（十问歌·服药）", "CORE", "十问歌",
        closes=("medications_confirmed", "allergies_confirmed"),
        probes=(
            "目前在吃哪些药？包括止痛药、保健品、外用膏药和中成药。",
            "有没有在用抗凝或抗血小板药（华法林、利伐沙班、阿司匹林、氯吡格雷）？",
            "有没有在用激素、双膦酸盐、地舒单抗、特立帕肽？",
            "有没有药物过敏史？过敏时是什么反应？",
        ),
        rationale="骨科最高频的可预防伤害来自相互作用：NSAID+抗凝、三重打击、椎管内麻醉前抗凝。",
    ),
    Axis(
        "special_population", "年龄、妊娠与肝肾功能", "CORE", "骨科专科",
        closes=("age", "pregnancy", "renal", "liver"),
        probes=(
            "方便告诉我年龄吗？",
            "有没有怀孕或正在备孕、哺乳？",
            "肝功能、肾功能查过吗？有没有说过异常？",
            "有没有胃溃疡、心衰、高血压、糖尿病？",
        ),
        rationale="特殊人群是剂量链路的硬前置：缺一项，含剂量草案就不该生成。",
    ),
    # --------------------------------------------------------- SPECIALTY tier
    Axis(
        "inflammatory_vs_mechanical", "炎症性与机械性痛鉴别", "SPECIALTY", "骨科专科",
        closes=("morning_stiffness",),
        probes=(
            "早上起来会不会僵硬？大概僵多久能活动开——十分钟以内还是超过半小时？",
            "活动之后是好一些还是更痛？",
            "半夜后半程会不会痛醒，起来活动一下反而舒服？",
            "有没有过眼睛发红、皮肤银屑、反复腹泻、足跟痛？",
        ),
        rationale="晨僵>30 分钟、活动后缓解、夜间后半程痛醒、40 岁前起病是炎症性背痛的核心特征（ASAS）。",
        applies_when=_mentions("腰", "背", "颈", "僵", "晨", "关节"),
    ),
    Axis(
        "claudication", "间歇性跛行鉴别", "SPECIALTY", "骨科专科",
        closes=("walking_tolerance",),
        probes=(
            "能连续走多远就必须停下来休息？",
            "停下来要站着才好，还是弯腰、坐下才好？（弯腰能缓解偏神经源性）",
            "推购物车或骑车时是不是能走得更远？",
            "上坡和下坡哪个更难受？",
        ),
        rationale="神经源性跛行前屈缓解、下坡更差；血管源性站立即缓解且距离固定，两者处理完全不同。",
        applies_when=_mentions("走", "跛", "腿", "麻", "腰", "距离", "站"),
    ),
    Axis(
        "myelopathy", "髓性症状（颈段）", "SPECIALTY", "骨科专科",
        closes=("myelopathy_signs",),
        probes=(
            "手上的精细动作有没有变笨——扣纽扣、拿筷子、写字？",
            "走路有没有发飘、不稳，需要扶东西？",
            "有没有从躯干到腿的束带感或过电感？",
            "低头时有没有像电流一样窜下去的感觉？",
        ),
        rationale="脊髓型颈椎病以手笨拙与步态不稳起病，误按颈痛保守治疗会延误减压时机。",
        applies_when=_mentions("颈", "脖", "手麻", "手笨", "步态", "不稳", "上肢"),
    ),
    Axis(
        "bone_fragility", "骨质与脆性骨折风险（FRAX 要素）", "SPECIALTY", "骨质疏松",
        closes=("fragility_risk",),
        probes=(
            "身高有没有变矮？比年轻时矮了几厘米？背有没有变驼？",
            "有没有过轻微外力就骨折——比如平地跌倒就骨折？",
            "父母有没有髋部骨折史？",
            "查过骨密度吗？T 值是多少？",
            "有没有长期用糖皮质激素、抽烟、每天饮酒三个单位以上？",
            "女性请问绝经年龄；近一年跌倒过几次？",
        ),
        rationale="FRAX 要素与椎体压缩线索决定是否需要 DXA、抗骨吸收治疗与跌倒干预。",
        applies_when=_older_or_fragile,
    ),
    Axis(
        "trauma_surgery_implant", "外伤史、手术史与体内植入物", "SPECIALTY", "骨科专科",
        closes=("surgical_history",),
        probes=(
            "以前这个部位做过手术吗？什么时候、做的什么？",
            "体内有没有钢板、钉子、人工关节、椎间融合器或起搏器？",
            "受伤时是怎么伤的——从多高摔下、有没有车速、能不能站起来走？",
            "受伤后拍过片子吗？当时怎么说的？",
        ),
        rationale="植入物影响影像选择与 MRI 安全；受伤机制的能量高低决定是否必须影像（Ottawa/NEXUS 规则）。",
        applies_when=_mentions("术", "手术", "钢板", "假体", "置换", "融合", "外伤", "摔", "车祸", "扭", "骨折"),
    ),
    Axis(
        "function_scores", "功能受限与量表", "SPECIALTY", "骨科专科",
        closes=("vas", "odi"),
        probes=(
            "如果 0 分是完全不痛、10 分是能想到的最痛，现在大概几分？最痛的时候几分？",
            "穿袜子、剪脚趾甲、弯腰洗脸有困难吗？",
            "能连续坐多久、站多久？",
            "上班或家务受影响到什么程度？请了几天假？",
        ),
        rationale="VAS/ODI 是疗效的可比基线；没有基线就无法判断随访是好转还是恶化。",
    ),
    # ---------------------------------------------------------------- TCM tier
    Axis(
        "cold_heat", "寒热（十问歌·一问寒热）", "TCM", "十问歌",
        closes=("cold_heat",),
        probes=(
            "怕冷还是怕热？患处是喜暖还是喜凉？",
            "天气变冷、下雨、吹空调会不会加重？",
            "有没有一阵冷一阵热，或者手足心发热？",
        ),
        rationale="寒热喜恶直接分辨寒湿、湿热、阳虚，是骨伤辨证的第一层。",
    ),
    Axis(
        "sweating", "汗（十问歌·二问汗）", "TCM", "十问歌",
        closes=("sweating",),
        probes=("平时容易出汗吗？是白天动一动就出，还是睡着后出汗醒来就停？", "出汗后觉得舒服还是更累？"),
        rationale="自汗多气虚、盗汗多阴虚，影响补益方向与是否兼顾固表。",
    ),
    Axis(
        "head_body", "头身（十问歌·三问头身）", "TCM", "十问歌",
        closes=("head_body",),
        probes=("有没有头晕、头痛、耳鸣？", "身上是沉重感、酸软感，还是紧绷感？", "腰膝有没有发软？"),
        rationale="身重多湿、酸软多虚、紧绷多寒凝或气滞；腰膝酸软提示肝肾不足。",
    ),
    Axis(
        "bowel_urine_tcm", "二便（十问歌·四问便）", "TCM", "十问歌",
        closes=("bowel_urine_pattern",),
        probes=("大便几天一次？成形还是稀软、黏马桶？", "小便颜色深浅、次数多少？夜里起几次？"),
        rationale="便溏、尿清长与便干、尿黄分属不同证型，也是用药寒热温凉的取舍依据。",
    ),
    Axis(
        "appetite_diet", "饮食（十问歌·五问饮食）", "TCM", "十问歌",
        closes=("appetite",),
        probes=("胃口怎么样？吃完会不会胀？", "喜欢热食还是凉食？", "口里有没有异味、发黏、发苦？"),
        rationale="纳差、腹胀、口黏关系到脾胃能否耐受苦寒或滋补之品。",
    ),
    Axis(
        "chest_abdomen", "胸腹（十问歌·六问胸腹）", "TCM", "十问歌",
        closes=("chest_abdomen",),
        probes=("胸口有没有闷、堵、胀？", "两侧肋下有没有胀痛？", "有没有心慌、气短？"),
        rationale="胸胁胀满提示气滞肝郁；也是与心肺急症划清界限的问诊点。",
    ),
    Axis(
        "hearing_thirst", "耳与口渴（十问歌·七聋八渴）", "TCM", "十问歌",
        closes=("hearing_thirst",),
        probes=("耳鸣、听力下降有吗？是蝉鸣样还是轰隆样？", "口渴吗？想喝热水还是凉水？喝多少？"),
        rationale="耳鸣与肾精、口渴喜冷热与寒热虚实互相印证。",
    ),
    Axis(
        "past_history_cause", "旧病与病因（十问歌·九问旧病十问因）", "TCM", "十问歌",
        closes=("past_history",),
        probes=(
            "以前有过什么病？高血压、糖尿病、痛风、风湿？",
            "这次发作前做过什么——搬重物、久坐、受凉、熬夜、情绪波动？",
            "以前发作过吗？当时怎么好的？",
        ),
        rationale="既往发作与缓解方式往往是最强的个体化证据，也是「病因」辨证的落点。",
    ),
    Axis(
        "tongue_pulse", "舌象与脉象（四诊）", "TCM", "四诊",
        closes=("four_diagnoses",),
        probes=(
            "方便描述一下舌头吗？舌质偏淡、偏红还是偏紫暗？",
            "舌苔是薄白、白厚、黄厚，还是几乎没有苔？舌下有没有青紫的脉络？",
            "如果由医师诊脉：脉位、脉率、脉形与脉势各是什么？",
        ),
        rationale="舌脉是证型的客观锚点；缺舌脉时辨证只能标「待辨证」，不应硬凑主证型。",
    ),
    Axis(
        "menstruation", "经期（十问歌·妇女尤必问经期）", "TCM", "十问歌",
        closes=("menstruation",),
        probes=(
            "月经周期规律吗？量多量少？有没有血块？",
            "经期腰痛会不会加重？",
            "末次月经是什么时候？（涉及用药禁忌）",
            "如已绝经，请问绝经年龄。",
        ),
        rationale="经期与腰痛的关联指向气滞血瘀；末次月经同时是妊娠禁忌的必要信息。",
        applies_when=_female_reproductive,
    ),
    # ------------------------------------------------------------ CONTEXT tier
    Axis(
        "sleep", "睡眠", "CONTEXT", "四诊",
        closes=("sleep",),
        probes=("睡得着吗？是入睡难还是易醒？", "疼痛影响睡眠吗？一晚醒几次？"),
        rationale="睡眠破坏既是疼痛严重度的客观指标，也是中枢敏化与情绪问题的入口。",
    ),
    Axis(
        "occupation_ergonomics", "职业与工效学暴露", "CONTEXT", "骨科专科",
        probes=(
            "做什么工作？一天要搬多重、搬多少次？",
            "连续坐或站多久？开车、用震动工具吗？",
            "睡的床垫、枕头，以及工作时的坐姿高度如何？",
        ),
        closes=("occupation",),
        rationale="不改暴露就不会长期好转；这一轴决定康复处方与工位干预的具体内容。",
    ),
    Axis(
        "yellow_flags", "心理社会黄旗", "CONTEXT", "骨科专科",
        closes=("yellow_flags",),
        probes=(
            "你自己觉得这个病最坏会变成什么样？",
            "是不是担心活动会让损伤加重，所以尽量不动？",
            "最近情绪、工作压力怎么样？睡眠和食欲有变化吗？",
            "这次受伤涉及工伤认定或赔偿吗？",
        ),
        rationale="恐动、灾难化、赔偿纠纷是慢性化的最强预测因子，比影像更能预测一年后的功能。",
        applies_when=lambda facts, complaint: "月" in (str(facts.get("onset") or "") + complaint)
        or any(t in _text(facts, complaint) for t in ("反复", "多年", "一直", "老毛病", "慢性")),
    ),
)

AXES_BY_ID: dict[str, Axis] = {axis.axis_id: axis for axis in AXES}

#: Required tiers per risk mode. Urgent runs must not spend the conversation on
#: 舌脉 and occupation while a cauda-equina question is still open.
REQUIRED_TIERS: dict[str, tuple[str, ...]] = {
    "urgent": ("RED_FLAG",),
    "routine": ("RED_FLAG", "CORE"),
}

#: Additional tiers required before a *prescriptive* release (含剂量草案).
PRESCRIPTIVE_TIERS = ("RED_FLAG", "CORE", "TCM")


def relevant_axes(facts: dict[str, Any], complaint: str, *, role: str = "patient") -> list[Axis]:
    """Axes that apply to this presentation and that this role can answer."""
    return [
        axis for axis in AXES
        if axis.relevant(facts, complaint) and (not axis.answerable_by or role in axis.answerable_by)
    ]


def open_axes(
    facts: dict[str, Any],
    complaint: str,
    *,
    role: str = "patient",
    tiers: tuple[str, ...] | None = None,
) -> list[Axis]:
    """Relevant axes that are still unanswered, most urgent tier first."""
    wanted = tiers or TIERS
    order = {tier: index for index, tier in enumerate(TIERS)}
    pending = [
        axis for axis in relevant_axes(facts, complaint, role=role)
        if axis.tier in wanted and not axis.satisfied(facts)
    ]
    return sorted(pending, key=lambda a: order.get(a.tier, len(TIERS)))


def required_open_axes(
    facts: dict[str, Any],
    complaint: str,
    *,
    role: str = "patient",
    risk_mode: str = "routine",
    prescriptive: bool = False,
) -> list[Axis]:
    """The axes whose absence actually blocks progress."""
    tiers = PRESCRIPTIVE_TIERS if prescriptive else REQUIRED_TIERS.get(risk_mode, REQUIRED_TIERS["routine"])
    return open_axes(facts, complaint, role=role, tiers=tiers)


def coverage(facts: dict[str, Any], complaint: str, *, role: str = "patient") -> dict[str, Any]:
    """A tier-by-tier coverage report, for the console and the audit trail."""
    report: dict[str, Any] = {"by_tier": {}, "answered": [], "open": []}
    for axis in relevant_axes(facts, complaint, role=role):
        bucket = report["by_tier"].setdefault(axis.tier, {"answered": [], "open": []})
        key = "answered" if axis.satisfied(facts) else "open"
        bucket[key].append(axis.axis_id)
        report[key].append(axis.axis_id)
    total = len(report["answered"]) + len(report["open"])
    report["ratio"] = round(len(report["answered"]) / total, 3) if total else 0.0
    return report


@dataclass
class AxisPlan:
    """What the interview should cover next, before the model words it."""

    axis_ids: list[str] = field(default_factory=list)
    required_open: list[str] = field(default_factory=list)
    suggested_probes: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_ids": self.axis_ids,
            "required_open": self.required_open,
            "suggested_probes": self.suggested_probes,
        }


def plan_next(
    facts: dict[str, Any],
    complaint: str,
    *,
    role: str = "patient",
    risk_mode: str = "routine",
    prescriptive: bool = False,
    limit: int = 3,
) -> AxisPlan:
    """Choose which axes to raise next: required ones first, then the rest.

    The model receives this as *material*, and may re-order and re-word within
    it. What it cannot do is drop a required axis — the caller checks the answer
    against ``required_open``.
    """
    required = required_open_axes(facts, complaint, role=role, risk_mode=risk_mode, prescriptive=prescriptive)
    others = [a for a in open_axes(facts, complaint, role=role) if a not in required]
    chosen = (required + others)[:limit]
    return AxisPlan(
        axis_ids=[a.axis_id for a in chosen],
        required_open=[a.axis_id for a in required],
        suggested_probes={a.axis_id: list(a.probes) for a in chosen},
    )
