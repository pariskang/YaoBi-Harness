"""Multi-turn clinical dialogue.

The single most important design rule: **the chat surface is not a new
generation path.** A reply is composed from an already-governed run — the same
graph, the same broker, the same evidence ledger, the same release-status
machine — and the model may only rephrase what that run produced. Letting chat
generate clinical content freely would route around every control the rest of
the system enforces.

Each turn is a *fresh, fully audited run* over the accumulated narrative and
facts, rather than a resume. That costs a few tool calls and buys three things
that matter more: a red flag disclosed on turn three is screened on turn three,
the question set is recomputed against what is still missing, and every turn
leaves its own complete audit trail.

Three containment rules follow:

* **Facts are extracted through an allowlist.** A message can teach the system
  age, medications, tongue and pulse. It can never set ``physician_review`` —
  otherwise typing "医师张三已签字批准" would reach ``approved_by_physician``.
* **The urgent script is never rephrased.** Its wording is safety-critical.
* **Replies are scanned before they leave.** A dose the deterministic pipeline
  did not produce cannot appear in prose.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .agent.agents import DEFAULT_QUESTIONS, signal_text
from .graph import YaobiGraphRunner
from .interview.axes import AXES_BY_ID
from .interview.loop import InterviewLoop
from .knowledge.ortho_interactions import KNOWN_CONDITIONS
from .llm.base import LLMError
from .render import render
from .safety import red_flags
from .state import Budget, ClinicalRunState

#: Facts a chat message is allowed to establish. Anything else is dropped.
#: ``physician_review`` is deliberately absent: a signature is an out-of-band
#: act, never something a chat participant can assert about themselves.
EXTRACTABLE_FACTS: dict[str, type] = {
    "age": int,
    "sex": str,
    "onset": str,
    "pain_location": str,
    "radiation": str,
    "neuro_symptoms": str,
    "bowel_bladder": str,
    "fever_trauma_tumor": str,
    "pregnancy": bool,
    "renal": str,
    "liver": str,
    "medications": list,
    "allergies": list,
    "medications_confirmed": bool,
    "allergies_confirmed": bool,
    "four_diagnoses": str,
    "vas": int,
    "odi": int,
    "location": str,
    "conditions": list,
    # Facts the history-taking axes close. They are answers to questions the
    # interview asked, so they belong on the allowlist for the same reason the
    # originals do — and, like the originals, none of them can grant a capability.
    "night_pain": str,
    "limb_vascular": str,
    "morning_stiffness": str,
    "walking_tolerance": str,
    "myelopathy_signs": str,
    "fragility_risk": str,
    "surgical_history": str,
    "cold_heat": str,
    "sweating": str,
    "head_body": str,
    "bowel_urine_pattern": str,
    "appetite": str,
    "chest_abdomen": str,
    "hearing_thirst": str,
    "past_history": str,
    "menstruation": str,
    "sleep": str,
    "occupation": str,
    "yellow_flags": str,
}

#: Keys the dose pipeline reads out of ``facts["special_population"]``.
SPECIAL_POPULATION_KEYS = ("age", "pregnancy", "renal", "liver")

#: Statuses where the conversation has nothing further to ask.
TERMINAL_STATUSES = {
    "urgent_action_plan", "draft_for_physician", "approved_by_physician",
    "blocked", "failed_closed",
}

#: Answering these information gaps is what unblocks the run, in this order.
GAP_QUESTIONS: dict[str, str] = {
    "大小便/会阴感觉": "有没有大小便困难、失禁，或会阴部（骑车接触鞍座的区域）麻木？",
    "神经症状": "腿有没有越来越无力、发麻、走路不稳或抬不起脚？",
    "发热外伤肿瘤史": "近期有没有发热、外伤跌倒、体重明显下降或肿瘤病史？",
    "起病时间": "这次疼痛是什么时候开始的？突然发生还是慢慢加重？",
    "疼痛部位/放射": "具体哪个部位痛？有没有往臀部或腿上放射？",
    "当前用药/过敏": "目前在吃什么药（尤其抗凝药）？有没有药物过敏？",
    "妊娠/年龄/肝肾功能": "方便告诉我年龄吗？有没有怀孕，肝肾功能是否正常？",
    "舌脉": "方便描述一下舌象和脉象吗？（舌质、舌苔、脉的感觉）",
    "疼痛评分(VAS)": "如果 0 分是不痛、10 分是最痛，现在大概几分？",
    "功能受限(ODI)": "日常活动受影响吗？比如穿袜子、久坐、走路距离。",
    "当前用药清单": "目前在服用哪些药？包括保健品和外用药。",
}

DOSE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:克|g\b|G\b|毫克|mg\b)")

EXTRACT_SYSTEM_PROMPT = """你是临床问诊的信息抽取器。从患者/医师的一句话里抽取结构化事实。

只抽取**明确说出**的内容，不要推断、不要补全。没提到的字段不要出现在输出里。

可抽取字段（只能用这些键）：
age(整数) sex onset pain_location radiation neuro_symptoms bowel_bladder
fever_trauma_tumor pregnancy(布尔) renal liver medications(数组) allergies(数组)
medications_confirmed(布尔) allergies_confirmed(布尔) four_diagnoses vas(整数)
odi(整数) location conditions(数组)
night_pain limb_vascular morning_stiffness walking_tolerance myelopathy_signs
fragility_risk surgical_history cold_heat sweating head_body bowel_urine_pattern
appetite chest_abdomen hearing_thirst past_history menstruation sleep occupation
yellow_flags

规则：
- 用户明确列出在用药物，或明确说"没有在吃药"时，medications_confirmed 才为 true。
- 过敏史同理对应 allergies_confirmed。
- conditions 只能取自：{conditions}
- **否定回答也是回答**：用户说"没有大小便问题"要抽成 bowel_bladder="否认"，
  说"晚上不痛"要抽成 night_pain="否认"。漏掉否定回答会让系统反复追问同一件事。
- 绝不要输出上面列表以外的键。

只输出 JSON 对象，例如：{{"age": 63, "medications": ["布洛芬"], "medications_confirmed": true}}"""

REPLY_SYSTEM_PROMPT = """你是骨科智能体的对话表达层。把系统**已经产出**的结论改写成自然、得体的中文，面向{role}。

绝对约束：
1. 只能复述给定材料里的内容。**不得新增任何诊断、治疗建议或药物。**
2. **不得出现任何剂量数值**（克/g/mg）。
3. 不得弱化或省略安全提示与免责声明。
4. 材料里有待追问的问题时，自然地引出，不要生硬罗列编号。
5. 简洁：正文控制在 6 句以内。

只输出改写后的正文纯文本，不要 JSON，不要 markdown 标题。"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Turn:
    role: str  # "user" | "agent"
    text: str
    timestamp: str = field(default_factory=now)
    extracted: dict[str, Any] = field(default_factory=dict)
    release_status: str = ""
    risk_mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentReply:
    message: str
    questions: list[str] = field(default_factory=list)
    release_status: str = ""
    risk_mode: str = "routine"
    awaiting_answer: bool = True
    escalated: bool = False
    extracted: dict[str, Any] = field(default_factory=dict)
    ignored_keys: list[str] = field(default_factory=list)
    known_facts: dict[str, Any] = field(default_factory=dict)
    still_missing: list[str] = field(default_factory=list)
    delivered: dict[str, Any] = field(default_factory=dict)
    composer: str = "template"
    #: The interview's own record for this turn: which axes were raised, who
    #: composed the wording, and how the adequacy judge ruled.
    interview: dict[str, Any] = field(default_factory=dict)
    #: Questions with their axis and tier, for a surface that wants to group them.
    structured_questions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coerce_facts(raw: Any, known_conditions: set[str] | None = None) -> tuple[dict[str, Any], list[str]]:
    """Filter a proposed fact dict down to the allowlist.

    Returns ``(accepted, ignored_keys)``. Type mismatches are dropped rather
    than coerced: a wrong type here would silently corrupt the triage or dose
    inputs downstream, and a missing fact is far safer than a wrong one.
    """
    accepted: dict[str, Any] = {}
    ignored: list[str] = []
    if not isinstance(raw, dict):
        return accepted, ignored
    conditions = known_conditions if known_conditions is not None else KNOWN_CONDITIONS

    for key, value in raw.items():
        expected = EXTRACTABLE_FACTS.get(key)
        if expected is None or value is None:
            ignored.append(key)
            continue
        if expected is int and isinstance(value, bool):
            ignored.append(key)          # booleans are ints in Python; not here
            continue
        if not isinstance(value, expected):
            ignored.append(key)
            continue
        if key == "conditions":
            value = [c for c in value if isinstance(c, str) and c in conditions]
            if not value:
                continue
        elif key in ("medications", "allergies"):
            value = [str(v).strip() for v in value if str(v).strip()]
        accepted[key] = value
    return accepted, ignored


#: "在吃布洛芬和华法林" — the phrasing a patient actually uses. This feeds the
#: interaction rule pack, so failing to catch it loses the highest-value finding
#: the system can make.
MEDICATION_LEAD_RE = re.compile(
    r"(?:在|正在|目前|平时|一直)?\s*(?:吃|服用|服|用|口服)(?:着|的)?\s*[:：]?\s*"
    r"([^。；;！!?？\n]{1,60})"
)
MEDICATION_SPLIT_RE = re.compile(r"[、,，和跟与＋+]|以及|还有|加上")
#: The captured span must stop before the sentence moves on to another topic,
#: otherwise "在吃布洛芬和华法林，没有过敏" lists 没有过敏 as a drug.
MEDICATION_STOP_RE = re.compile(r"没有?|未曾?|无|不曾|过敏|另外|其他|其它|平时|以前|睡眠|血压")
MEDICATION_NOISE = ("药", "中药", "西药", "止痛药", "这些", "那些", "什么", "别的")


def _extract_medications(text: str) -> list[str]:
    """Pull a medication list out of natural phrasing, conservatively."""
    match = MEDICATION_LEAD_RE.search(text or "")
    if not match:
        return []
    segment = match.group(1)
    stop = MEDICATION_STOP_RE.search(segment)
    if stop:
        segment = segment[: stop.start()]
    drugs = []
    for chunk in MEDICATION_SPLIT_RE.split(segment):
        name = chunk.strip().strip("的了吗呢。，,、").lstrip("点些一两三")
        if not name or name in MEDICATION_NOISE or len(name) > 24:
            continue
        drugs.append(name)
    return drugs


#: **Symptom** terms: their un-suppressed presence means the symptom is reported.
#:
#: Whether an occurrence is suppressed is decided by the *existing* clause-scoped
#: logic in :mod:`yaobi_harness.safety.red_flags` — the same code that decides
#: whether a red flag fires. Re-implementing negation here produced exactly the
#: failure that module exists to prevent.
RED_FLAG_SYMPTOMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bowel_bladder", ("失禁", "尿不出", "解不出", "漏尿", "尿失禁", "大便失禁", "控制不住",
                       "会阴麻", "会阴发麻", "鞍区麻", "会阴部麻")),
    ("neuro_symptoms", ("越来越无力", "越来越没力", "抬不起", "走路不稳", "拖步", "麻木加重", "无力加重")),
    ("fever_trauma_tumor", ("发烧", "发热", "寒战", "盗汗", "体重下降", "消瘦", "肿瘤", "癌",
                            "摔倒", "跌倒", "车祸", "外伤")),
    ("night_pain", ("痛醒", "疼醒", "夜间加重", "静息痛")),
    ("limb_vascular", ("发紫", "花斑", "咳血", "张力性", "明显肿胀")),
)

#: **Topic** terms: they merely name the axis. "大小便" says nothing on its own —
#: "大小便正常" is a denial and "大小便失禁" is a report. Treating a topic word as a
#: symptom is what turned "大小便正常" into a cauda-equina emergency.
RED_FLAG_TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bowel_bladder", ("大小便", "小便", "大便", "排尿", "排便", "会阴", "鞍区")),
    ("neuro_symptoms", ("无力", "没力", "麻木", "发麻", "腿麻", "脚麻")),
    ("fever_trauma_tumor", ("发烧", "发热", "体温", "盗汗", "体重", "肿瘤", "外伤")),
    ("night_pain", ("夜里", "夜间", "晚上", "半夜")),
    ("limb_vascular", ("肿", "皮温", "胸闷", "气短")),
)

#: Cues that turn a topic mention into a denial. ``red_flags.NEGATION_CUES``
#: covers 没有/无/否认; these cover the way patients actually answer "正常"、"没事"。
NORMALITY_CUES = ("正常", "没事", "挺好", "都好", "还好", "没问题", "无异常", "没异常", "不痛", "没变化")

#: Colloquial denials that ``red_flags.NEGATION_CUES`` deliberately omits.
#:
#: That list is tuned for clinical notes, where a nurse writes "无发热" rather than
#: "不会发烧". Extending the shared list would change red-flag screening for every
#: run, so the conversational forms are handled here, at the extraction layer only.
COLLOQUIAL_DENIALS = ("不会", "不太", "没怎么", "从来不", "从没", "不曾", "不咋")

#: Axes where a *historical* mention is a positive answer rather than a suppressed
#: one. "以前查出过肿瘤" is precisely what this axis asks about, even though the
#: screening layer correctly declines to treat an old diagnosis as an emergency.
HISTORY_IS_ANSWER = {"fever_trauma_tumor", "surgical_history", "past_history"}

#: Standalone phrases that answer whatever was just asked, naming no topic at all.
BARE_DENIALS = ("都正常", "都好", "没事", "一切正常", "没什么", "都没有", "没有这些", "以上都无", "都不是")

#: Literal denials the topic-plus-cue machinery cannot reach, because the negation
#: is fused into the phrase: "腿不麻" contains neither the topic "腿麻" nor a
#: recognised cue before it. These are frequent enough in speech that leaving them
#: out kept re-asking a question the patient had just answered.
AXIS_DENIAL_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("neuro_symptoms", ("腿不麻", "脚不麻", "手不麻", "不麻", "没麻", "没无力", "力气正常", "没觉得无力")),
    ("bowel_bladder", ("能解出来", "尿得出", "解得出", "会阴不麻", "屁股不麻")),
    ("night_pain", ("晚上不痛", "夜里不痛", "夜间不痛", "睡得着", "不影响睡眠")),
    ("limb_vascular", ("腿不肿", "没肿", "不肿", "两条腿一样粗")),
    ("fever_trauma_tumor", ("不发烧", "没发烧", "体温正常", "没受伤", "没摔", "没外伤")),
)

#: Onset phrasing. Kept broad because it is low-risk: a wrong onset changes
#: wording, not safety, and having *no* onset blocks a CORE axis indefinitely.
ONSET_RE = re.compile(r"((?:\d+\s*(?:天|周|个?月|年)|昨天|今天|前天|这两天|最近|突然|逐渐)[^。；;，,]{0,10})")
PAIN_SITE_RE = re.compile(r"((?:腰|颈|背|肩|膝|髋|踝|肘|腕|足|手|臀|大腿|小腿|脊柱|骶)[^。；;，,]{0,6}(?:痛|疼|酸|麻|不适))")
RADIATION_RE = re.compile(r"((?:放射|窜|串|传|连)到?[^。；;，,]{0,12}|往[^。；;，,]{0,10}(?:窜|串|放射))")


def _colloquially_denied(clause: str, term: str) -> bool:
    """True when a colloquial denial governs ``term`` within this clause.

    Scoped the same way as :func:`red_flags._is_negated`: the cue must sit before
    the term and close to it, so "不会痛醒" denies and "不会走路了，晚上痛醒" does not.
    """
    index = clause.find(term)
    if index < 0:
        return False
    left = clause[:index]
    position = max((left.rfind(cue) for cue in COLLOQUIAL_DENIALS if cue in left), default=-1)
    return position >= 0 and len(left[position:]) <= 8


def _extract_red_flag_answers(text: str, asked_axes: tuple[str, ...] = ()) -> dict[str, Any]:
    """Record explicit answers — positive or negative — to the red-flag axes.

    Classification is delegated to :func:`red_flags.is_current_patient_symptom`,
    so a denial, a family history and a hypothetical all read the same way here as
    they do in the screening itself. An affirmation is stored as ``"报告"`` and a
    denial as ``"否认"``; both close the axis, which is the point — the interview
    must be able to tell "answered no" from "not asked".
    """
    facts: dict[str, Any] = {}
    clauses = red_flags.split_clauses(text)

    # Reports first: a symptom term that survives clause classification wins over
    # any denial elsewhere in the message.
    for key, terms in RED_FLAG_SYMPTOMS:
        for clause in clauses:
            hit = next((t for t in terms if t in clause), None)
            if hit is None:
                continue
            if _colloquially_denied(clause, hit):
                facts.setdefault(key, "否认")
                continue
            suppressed, reason = red_flags._classify_clause(clause, hit)
            if not suppressed:
                facts[key] = "报告"
                break
            if reason.startswith("historical") and key in HISTORY_IS_ANSWER:
                facts[key] = "既往报告"
                break
            if reason == "negated_in_clause":
                # An explicit denial is an answer. Third-party and hypothetical
                # mentions are not — "我爸有肿瘤" says nothing about this patient.
                facts.setdefault(key, "否认")

    # Then denials expressed against the topic rather than the symptom.
    for key, terms in RED_FLAG_TOPICS:
        if facts.get(key) == "报告":
            continue
        for clause in clauses:
            hit = next((t for t in terms if t in clause), None)
            if hit is None:
                continue
            if (
                red_flags._is_negated(clause, hit)
                or _colloquially_denied(clause, hit)
                or any(cue in clause for cue in NORMALITY_CUES)
            ):
                facts.setdefault(key, "否认")
                break

    # Fused denials, checked last so an explicit report always wins.
    for key, phrases in AXIS_DENIAL_PHRASES:
        if facts.get(key) == "报告":
            continue
        if any(phrase in text for phrase in phrases):
            facts.setdefault(key, "否认")

    if not facts and any(phrase in text for phrase in BARE_DENIALS):
        # A bare "都正常" answers whatever was just asked, and nothing else.
        for axis_id in asked_axes:
            axis = AXES_BY_ID.get(axis_id)
            if axis is not None and axis.tier == "RED_FLAG":
                for key in axis.closes:
                    facts.setdefault(key, "否认")
    return facts


def rule_extract(message: str, asked_axes: tuple[str, ...] = ()) -> dict[str, Any]:
    """Deterministic fallback extractor.

    Deliberately conservative on anything that feeds a dose decision, and
    deliberately broader on narrative fields like onset and pain site — getting
    those wrong changes wording, while never having them stalls the interview.

    ``asked_axes`` is what the previous round asked about, which is what makes a
    bare "都正常" interpretable.
    """
    facts: dict[str, Any] = {}
    text = message or ""
    facts.update(_extract_red_flag_answers(text, asked_axes))

    onset = ONSET_RE.search(text)
    if onset:
        facts["onset"] = onset.group(1).strip()
    site = PAIN_SITE_RE.search(text)
    if site:
        facts["pain_location"] = site.group(1).strip()
    radiation = RADIATION_RE.search(text)
    if radiation:
        facts["radiation"] = radiation.group(1).strip()

    age = re.search(r"(\d{1,3})\s*(?:岁|周岁)", text)
    if age and 0 < int(age.group(1)) < 130:
        facts["age"] = int(age.group(1))

    vas = re.search(r"(?:VAS|疼痛评分|疼痛)\D{0,6}(\d{1,2})\s*分", text, re.I)
    if vas and int(vas.group(1)) <= 10:
        facts["vas"] = int(vas.group(1))

    if re.search(r"(没有?|未曾?|无|不)\s*(在)?\s*(吃|用|服)(任何)?药", text):
        facts["medications"], facts["medications_confirmed"] = [], True
    else:
        drugs = _extract_medications(text)
        if drugs:
            facts["medications"], facts["medications_confirmed"] = drugs, True
    if re.search(r"(没有?|未曾?|无)\s*(任何)?(药物|食物)?过敏", text):
        facts["allergies"], facts["allergies_confirmed"] = [], True

    # Check the denial first, and allow the bare 没 — "没怀孕" is the common form
    # and was previously read as a *confirmation* of pregnancy.
    if re.search(r"(没有?|未曾?|无|不是?|非)\s*(在)?\s*(怀孕|妊娠)", text) or "已绝经" in text:
        facts["pregnancy"] = False
    elif re.search(r"(怀孕|妊娠)", text):
        facts["pregnancy"] = True

    if re.search(r"(肝肾功能|肝功和?肾功|肾功能)[^。；;]{0,6}(正常|无异常|没问题)", text):
        facts["renal"] = facts["liver"] = "normal"

    tongue = re.search(r"(舌[^。；;，,]{1,24})", text)
    if tongue:
        facts["four_diagnoses"] = tongue.group(1)

    # Specialty axes patients answer in a recognisable form. Without these the
    # deterministic path cannot close a SPECIALTY axis at all, so a model-less run
    # re-asks the same three questions until the judge calls it stalled.
    walking = re.search(r"(?:走|步行)[^。；;，,]{0,4}(\d+)\s*(?:米|m|公里|km|步)", text)
    if walking:
        facts["walking_tolerance"] = walking.group(0)
    elif re.search(r"(?:走|步行)[^。；;，,]{0,6}(?:就得?停|要休息|停下来|走不了)", text):
        facts["walking_tolerance"] = "行走受限，具体距离未量化"

    stiff = re.search(r"(?:晨僵|早上|晨起)[^。；;，,]{0,8}(?:僵|硬)[^。；;，,]{0,10}", text)
    if stiff:
        facts["morning_stiffness"] = stiff.group(0)
    elif re.search(r"(?:早上|晨起)[^。；;，,]{0,6}(?:不僵|没有?僵)", text):
        facts["morning_stiffness"] = "否认晨僵"

    odi = re.search(r"(?:ODI|功能障碍指数)\D{0,4}(\d{1,3})", text, re.I)
    if odi and int(odi.group(1)) <= 100:
        facts["odi"] = int(odi.group(1))
    elif re.search(r"(?:穿袜|剪脚趾甲|弯腰洗脸|系鞋带)[^。；;，,]{0,8}(?:困难|费劲|做不了|不方便)", text):
        facts["odi"] = 40  # a coarse marker of real functional limitation

    job = re.search(r"(?:做|干|从事|职业是)[^。；;，,]{0,10}(?:工作|工人|搬运|司机|教师|护士|程序员|销售|农活|厨师)[^。；;，,]{0,6}", text)
    if job:
        facts["occupation"] = job.group(0)

    sleep = re.search(r"(?:睡|入睡|失眠)[^。；;，,]{0,12}", text)
    if sleep and any(cue in text for cue in ("睡不", "失眠", "睡得", "易醒", "入睡")):
        facts["sleep"] = sleep.group(0)
    return facts


class ConversationSession:
    """One clinical conversation.

    The session owns the accumulated narrative and facts; each turn runs the
    ordinary graph over them from scratch, so nothing about the conversation
    bypasses the control plane.
    """

    def __init__(
        self,
        role: str = "patient",
        *,
        runner: YaobiGraphRunner | None = None,
        allow_prescription: bool = False,
        session_id: str | None = None,
        budget_factory: Any | None = None,
    ) -> None:
        self.session_id = session_id or f"conv_{uuid.uuid4().hex[:10]}"
        self.runner = runner or YaobiGraphRunner()
        self.role = role
        self.allow_prescription = allow_prescription
        self.budget_factory = budget_factory or Budget
        self.narrative: list[str] = []
        self.facts: dict[str, Any] = {}
        self.turns: list[Turn] = []
        self.asked: list[str] = []
        self.images: list[dict[str, Any]] = []
        self.state: ClinicalRunState | None = None
        self.llm = getattr(self.runner, "llm", None)
        # One interview loop for the whole conversation. This is what makes
        # "不断追问" converge: round history spans turns, so the judge can see that
        # two consecutive rounds produced the same gaps and stop asking, rather
        # than each turn starting over and re-asking forever.
        self.interview = InterviewLoop(
            self.llm,
            skill_spec=getattr(self.runner, "skill_registry", None)
            and self.runner.skill_registry.specs.get("yaobi.interview"),
        )
        interview_agent = getattr(self.runner, "agents", {}).get("InterviewAgent")
        if interview_agent is not None:
            interview_agent.loop = self.interview

    # ------------------------------------------------------------------ public
    def send(self, message: str) -> AgentReply:
        """Take one user message, run the graph, and return the agent's reply."""
        text = (message or "").strip()
        if not text:
            raise ValueError("消息不能为空")

        previous_risk = self.state.risk_mode if self.state else "routine"
        accepted, ignored = self._extract(text)
        self._merge(accepted)
        self.narrative.append(text)
        self.turns.append(Turn("user", text, extracted=accepted))

        self.state = self._run()
        escalated = self.state.risk_mode == "urgent" and previous_risk != "urgent"
        if escalated:
            self.state.warn(
                "对话过程中检出红旗信号，已升级为急症模式: "
                + signal_text(sorted({h.get("signal", "") for h in self._hits()}))
            )

        reply = self._compose(escalated=escalated, extracted=accepted, ignored=ignored)
        self.asked += [q for q in reply.questions if q not in self.asked]
        self.turns.append(Turn("agent", reply.message, release_status=reply.release_status,
                               risk_mode=reply.risk_mode))
        return reply

    @property
    def complaint(self) -> str:
        return "。".join(self.narrative)

    def transcript(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.turns]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "role": self.role,
            "allow_prescription": self.allow_prescription,
            "narrative": list(self.narrative),
            "facts": dict(self.facts),
            "asked": list(self.asked),
            "turns": self.transcript(),
            "state": self.state.to_dict() if self.state else None,
            "interview": self.interview.summary(self.facts, self.complaint, role=self.role),
            # Attachments carry a path or data URI; the image bytes themselves are
            # never written into a transcript.
            "images": [{"kind": i["kind"], "deidentified": i["deidentified"]} for i in self.images],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], runner: YaobiGraphRunner | None = None) -> "ConversationSession":
        session = cls(
            role=payload.get("role", "patient"),
            runner=runner,
            allow_prescription=bool(payload.get("allow_prescription")),
            session_id=payload.get("session_id"),
        )
        session.narrative = list(payload.get("narrative", []))
        session.facts = dict(payload.get("facts", {}))
        session.asked = list(payload.get("asked", []))
        session.turns = [Turn(**t) for t in payload.get("turns", [])]
        if payload.get("state"):
            session.state = ClinicalRunState.from_dict(payload["state"])
        return session

    # --------------------------------------------------------------- internals
    def _merge(self, facts: dict[str, Any]) -> None:
        """Apply accepted facts, routing special-population keys where doses read them."""
        population = dict(self.facts.get("special_population", {}))
        for key, value in facts.items():
            if key in SPECIAL_POPULATION_KEYS:
                population[key] = value
            if key in ("medications", "allergies"):
                merged = list(self.facts.get(key, [])) + list(value)
                self.facts[key] = list(dict.fromkeys(merged))
                continue
            if key == "conditions":
                self.facts["conditions"] = sorted(set(self.facts.get("conditions", [])) | set(value))
                continue
            self.facts[key] = value
        if population:
            self.facts["special_population"] = population

    def attach_image(self, ref: str, *, kind: str = "other", deidentified: bool = False) -> dict[str, Any]:
        """Attach an image to the conversation; it is read on the next turn.

        The attestation is stored with the attachment rather than assumed, so the
        run records who asserted de-identification. Bytes are not copied anywhere:
        ``ref`` is a path or a ``data:`` URI that the vision tool reads once.
        """
        from .vision.client import IMAGE_KINDS

        if kind not in IMAGE_KINDS:
            raise ValueError(f"未知图片类型 {kind!r}；支持 {list(IMAGE_KINDS)}")
        entry = {"kind": kind, "ref": ref, "deidentified": bool(deidentified)}
        self.images.append(entry)
        return entry

    def _run(self) -> ClinicalRunState:
        state = ClinicalRunState(complaint=self.complaint, role=self.role)
        state.facts.update(self.facts)
        state.images = [dict(i) for i in self.images]
        state.budget = self.budget_factory()
        return self.runner.run(state, allow_prescription=self.allow_prescription)

    def _hits(self) -> list[dict[str, Any]]:
        if not self.state:
            return []
        return self.state.outputs.get("intake", {}).get("screening", {}).get("hits", [])

    def _extract(self, text: str) -> tuple[dict[str, Any], list[str]]:
        """LLM extraction behind the allowlist filter, falling back to rules."""
        proposed: Any = None
        budget = self.state.budget if self.state else self.budget_factory()
        if self.llm is not None and getattr(self.llm, "available", False) and budget.reserve_llm():
            try:
                response = self.llm.chat(
                    [
                        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT.format(
                            conditions=", ".join(sorted(KNOWN_CONDITIONS)))},
                        {"role": "user", "content": text},
                    ],
                    temperature=0.0, max_tokens=400, response_format_json=True,
                )
                proposed = response.json(None)
            except (LLMError, Exception):  # noqa: BLE001 - extraction must never break a turn
                proposed = None

        asked_axes = tuple(
            q.get("axis_id", "") for q in (
                (self.state.outputs.get("interview") or {}).get("questions", []) if self.state else []
            )
        )
        accepted, ignored = coerce_facts(proposed)
        # Merge rather than replace: the model is better at narrative fields, the
        # rules are better at explicit denials, and losing a denial is what makes
        # the interview re-ask a question the patient already answered.
        rules, _ = coerce_facts(rule_extract(text, asked_axes))
        for key, value in rules.items():
            accepted.setdefault(key, value)
        return accepted, sorted(set(ignored))

    def _next_questions(self, limit: int = 3) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """The interview's questions for this turn, plus its own record.

        The questions come from :class:`~yaobi_harness.interview.loop.InterviewLoop`,
        which ran as a graph node during ``_run``. Reading its output rather than
        re-deriving questions here is what keeps a single source of enquiry: what
        the audit trail records is exactly what the patient was asked.
        """
        state = self.state
        record = dict((state.outputs.get("interview") or {})) if state else {}
        questions = [dict(q) for q in record.get("questions", [])][:limit]
        if questions:
            return questions, record

        # An interview that ran and was satisfied has nothing to ask. Only fall
        # back when no interview produced a decision at all — an LLM plan that
        # omitted the node, or a run that failed closed before reaching it.
        if (record.get("verdict") or {}).get("verdict") in ("achieved", "stalled", "cap_reached"):
            return [], record

        fallback = [q for q in (state.open_questions if state else []) if q not in self.asked]
        fallback = fallback or [q for q in DEFAULT_QUESTIONS if q not in self.asked]
        return [
            {"axis_id": "", "label": "补充信息", "tier": "CORE", "question": q,
             "why": "", "options": [], "origin": "fallback"}
            for q in fallback[:limit]
        ], record

    def _compose(self, *, escalated: bool, extracted: dict[str, Any], ignored: list[str]) -> AgentReply:
        state = self.state
        delivered = render(state, self.role)
        awaiting = state.release_status not in TERMINAL_STATUSES
        # The interview record is read on every turn, including terminal ones: the
        # interview node still ran, and its coverage and verdict belong in the
        # audit trail even when there is nothing further to ask.
        record = dict(state.outputs.get("interview") or {})
        structured = self._next_questions()[0] if awaiting else []
        questions = [q["question"] for q in structured]

        body = self._template_reply(delivered, structured, escalated)
        composer = "template"
        rephrased = self._rephrase(body, delivered, questions)
        if rephrased:
            body, composer = rephrased, "llm_rephrase"

        return AgentReply(
            message=body,
            questions=questions,
            release_status=state.release_status,
            risk_mode=state.risk_mode,
            awaiting_answer=awaiting,
            escalated=escalated,
            extracted=extracted,
            ignored_keys=ignored,
            known_facts=dict(self.facts),
            still_missing=list(state.missing_information),
            delivered=delivered,
            composer=composer,
            interview=self._interview_meta(record),
            structured_questions=structured,
        )

    def _interview_meta(self, record: dict[str, Any]) -> dict[str, Any]:
        """The interview summary a surface can show without reading the audit."""
        verdict = dict(record.get("verdict") or {})
        coverage = dict(record.get("coverage") or {})
        return {
            "rounds_used": record.get("rounds_used", self.interview.rounds_used),
            "composer": record.get("composer", "not_run"),
            "verdict": verdict.get("verdict", ""),
            "verdict_reason": verdict.get("reason", ""),
            "judged_by": verdict.get("judged_by", ""),
            "blocking": verdict.get("blocking_labels", []),
            # Ids as well as labels: the console marks a required-but-open axis
            # differently from a merely open one, and it keys off the id.
            "blocking_axis_ids": verdict.get("blocking_axes", []),
            "still_missing_axes": verdict.get("missing_labels", []),
            "contradictions": verdict.get("contradictions", []),
            "coverage_ratio": coverage.get("ratio", 0.0),
            "answered_axes": [AXES_BY_ID[a].label for a in coverage.get("answered", []) if a in AXES_BY_ID],
            "open_axes": [AXES_BY_ID[a].label for a in coverage.get("open", []) if a in AXES_BY_ID],
            "model_claimed_complete": bool(record.get("model_claimed_complete")),
            "rejected": record.get("rejected", []),
        }

    def _template_reply(self, delivered: dict[str, Any], questions: list[dict[str, Any]], escalated: bool) -> str:
        """Deterministic prose built only from what the run already released."""
        lines: list[str] = []

        if self.state.risk_mode == "urgent" and delivered.get("urgent"):
            urgent = delivered["urgent"]
            if escalated:
                lines.append("你刚才描述的情况属于急症信号，我们必须先处理这一点。")
            for key in ("risk_judgement", "immediate_action", "transport_advice", "uncertainty"):
                if urgent.get(key):
                    lines.append(urgent[key])
            return "\n".join(lines)

        lines.append({
            "blocked": "本次结果没有通过安全审查，不能作为临床依据。",
            "failed_closed": "关键环节出现故障，系统已按故障关闭处理，本次不产出结论。",
            "needs_examination": "根据目前信息，建议先线下评估——有需要排除的风险点。",
            "insufficient_evidence": "目前证据还不足以支撑结论。",
            "treatment_advice_only": "已经可以给出不含剂量的治法方向。",
            "draft_for_physician": "已生成处方草案，但必须由医师逐味审核签名后才能作为处方。",
            "approved_by_physician": "处方已完成医师逐味审核与签名放行。",
        }.get(self.state.release_status, "我还需要一些信息才能给出有把握的判断。"))

        for warning in (delivered.get("medication_warnings") or [])[:2]:
            lines.append(f"用药提醒：{warning.get('combination', '')}——{warning.get('what_to_do', '')}")

        causes = delivered.get("what_this_might_be") or (delivered.get("biomedical") or {}).get("differentials") or []
        if causes:
            lines.append("目前考虑的方向：" + "、".join(str(c) for c in causes[:3]) + "。")

        for issue in (self.state.safety_issues or [])[:2]:
            lines.append(f"需要注意：{issue}")

        if questions:
            lines.append("为了更准确，还想请你回答：")
            for question in questions:
                line = f"· {question['question']}"
                if question.get("options"):
                    line += "（" + " / ".join(question["options"][:4]) + "）"
                lines.append(line)

        if delivered.get("disclaimer"):
            lines.append(delivered["disclaimer"])
        return "\n".join(line for line in lines if line)

    def _rephrase(self, body: str, delivered: dict[str, Any], questions: list[str]) -> str | None:
        """Let the model polish wording — never add content.

        Rejected outright if it contains a dose or comes back empty; rejection
        simply keeps the template text, so the conversation never depends on the
        model behaving. The urgent script is never sent here at all: its wording
        is safety-critical and must not drift.
        """
        if self.llm is None or not getattr(self.llm, "available", False):
            return None
        if self.state.risk_mode == "urgent":
            return None
        if not self.state.budget.reserve_llm():
            return None
        try:
            response = self.llm.chat(
                [
                    {"role": "system", "content": REPLY_SYSTEM_PROMPT.format(role=self.role)},
                    {"role": "user", "content": json.dumps(
                        {
                            "release_status": self.state.release_status,
                            "draft_reply": body,
                            "questions": questions,
                            "disclaimer": delivered.get("disclaimer", ""),
                        },
                        ensure_ascii=False,
                    )},
                ],
                temperature=0.2, max_tokens=600,
            )
            self.state.budget.charge_llm_tokens(response.total_tokens)
        except (LLMError, Exception):  # noqa: BLE001
            self.state.warn("回复改写失败，使用模板回复")
            return None

        text = (response.text or "").strip()
        if not text:
            return None
        if DOSE_RE.search(text):
            self.state.warn("模型改写的回复中出现剂量数值，已丢弃并使用模板回复")
            return None
        disclaimer = delivered.get("disclaimer", "")
        if disclaimer and disclaimer[:12] not in text:
            text = f"{text}\n{disclaimer}"
        return text
