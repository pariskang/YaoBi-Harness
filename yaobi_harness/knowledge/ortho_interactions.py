"""Orthopaedic drug-interaction rule pack.

This is the project's own encoding of well-established clinical facts, shipped
under the repository licence so it can be used without a third-party data
licence. It is **not** a substitute for a full interaction database: it covers
the combinations that matter most in an orthopaedic / osteoporosis setting, and
is meant to run *alongside* label-derived findings (openFDA / NMPA) and, where
licensed, a comprehensive DDI source.

Every rule must be reviewed and signed off by the deploying institution's
pharmacist before it is relied on clinically. Rules carry a mechanism and a
management action so an answer can explain itself rather than just flagging.

Matching is class-based and bilingual: patient medication lists arrive as free
text in Chinese or English, so each class lists both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

SEVERITY_ORDER = {"contraindicated": 0, "major": 1, "moderate": 2, "minor": 3}
BLOCKING_SEVERITIES = {"contraindicated", "major"}

#: Drug class -> member spellings (generic English + common Chinese).
DRUG_CLASSES: dict[str, tuple[str, ...]] = {
    "nsaid": (
        "ibuprofen", "naproxen", "diclofenac", "celecoxib", "etoricoxib", "meloxicam", "indomethacin",
        "ketorolac", "loxoprofen", "aceclofenac", "piroxicam", "nimesulide", "flurbiprofen",
        "布洛芬", "萘普生", "双氯芬酸", "扶他林", "塞来昔布", "依托考昔", "美洛昔康", "吲哚美辛",
        "酮咯酸", "洛索洛芬", "醋氯芬酸", "吡罗昔康", "尼美舒利", "氟比洛芬",
    ),
    "aspirin": ("aspirin", "acetylsalicylic", "阿司匹林", "乙酰水杨酸"),
    "anticoagulant": (
        "warfarin", "rivaroxaban", "apixaban", "edoxaban", "dabigatran", "heparin", "enoxaparin",
        "nadroparin", "dalteparin", "fondaparinux", "acenocoumarol",
        "华法林", "利伐沙班", "阿哌沙班", "依度沙班", "达比加群", "肝素", "依诺肝素", "那屈肝素",
        "达肝素", "磺达肝癸钠", "低分子肝素",
    ),
    "antiplatelet": (
        "clopidogrel", "ticagrelor", "prasugrel", "cilostazol", "dipyridamole", "tirofiban",
        "氯吡格雷", "替格瑞洛", "普拉格雷", "西洛他唑", "双嘧达莫", "替罗非班",
    ),
    "acei_arb": (
        "enalapril", "lisinopril", "ramipril", "perindopril", "captopril", "benazepril", "fosinopril",
        "losartan", "valsartan", "irbesartan", "olmesartan", "telmisartan", "candesartan", "sacubitril",
        "依那普利", "赖诺普利", "雷米普利", "培哚普利", "卡托普利", "贝那普利", "福辛普利",
        "氯沙坦", "缬沙坦", "厄贝沙坦", "奥美沙坦", "替米沙坦", "坎地沙坦", "沙库巴曲",
    ),
    "diuretic": (
        "furosemide", "hydrochlorothiazide", "indapamide", "torasemide", "torsemide", "spironolactone",
        "bumetanide", "chlortalidone",
        "呋塞米", "速尿", "氢氯噻嗪", "吲达帕胺", "托拉塞米", "螺内酯", "布美他尼", "氯噻酮",
    ),
    "corticosteroid": (
        "prednisone", "prednisolone", "methylprednisolone", "dexamethasone", "hydrocortisone",
        "betamethasone", "triamcinolone", "deflazacort",
        "泼尼松", "强的松", "泼尼松龙", "甲泼尼龙", "地塞米松", "氢化可的松", "倍他米松", "曲安奈德",
    ),
    "ssri_snri": (
        "fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram", "fluvoxamine",
        "venlafaxine", "duloxetine", "milnacipran",
        "氟西汀", "舍曲林", "帕罗西汀", "西酞普兰", "艾司西酞普兰", "氟伏沙明", "文拉法辛",
        "度洛西汀", "米那普仑",
    ),
    "opioid": (
        "morphine", "oxycodone", "hydrocodone", "fentanyl", "hydromorphone", "codeine", "buprenorphine",
        "methadone", "tapentadol", "pethidine", "meperidine", "dihydrocodeine", "sufentanil",
        "吗啡", "羟考酮", "氢可酮", "芬太尼", "氢吗啡酮", "可待因", "丁丙诺啡", "美沙酮",
        "他喷他多", "哌替啶", "杜冷丁", "双氢可待因", "舒芬太尼",
    ),
    "tramadol": ("tramadol", "曲马多", "曲马朵"),
    "benzodiazepine_sedative": (
        "diazepam", "alprazolam", "lorazepam", "clonazepam", "midazolam", "estazolam", "temazepam",
        "zolpidem", "zopiclone", "eszopiclone", "chloral",
        "地西泮", "安定", "阿普唑仑", "劳拉西泮", "氯硝西泮", "咪达唑仑", "艾司唑仑", "替马西泮",
        "唑吡坦", "佐匹克隆", "右佐匹克隆", "水合氯醛",
    ),
    "gabapentinoid": ("gabapentin", "pregabalin", "加巴喷丁", "普瑞巴林"),
    "alcohol": ("alcohol", "ethanol", "酒精", "乙醇", "饮酒"),
    "maoi": (
        "phenelzine", "tranylcypromine", "isocarboxazid", "selegiline", "rasagiline", "moclobemide",
        "linezolid", "methylene blue",
        "苯乙肼", "反苯环丙胺", "司来吉兰", "雷沙吉兰", "吗氯贝胺", "利奈唑胺", "亚甲蓝",
    ),
    "serotonergic_other": (
        "sumatriptan", "rizatriptan", "zolmitriptan", "ondansetron", "granisetron", "dextromethorphan",
        "st john", "hypericum",
        "舒马曲坦", "利扎曲坦", "佐米曲坦", "昂丹司琼", "格拉司琼", "右美沙芬", "圣约翰草", "贯叶连翘",
    ),
    "acetaminophen": ("acetaminophen", "paracetamol", "对乙酰氨基酚", "扑热息痛", "醋氨酚"),
    "enzyme_inducer": (
        "rifampicin", "rifampin", "rifabutin", "carbamazepine", "phenytoin", "phenobarbital",
        "primidone", "efavirenz", "st john", "hypericum",
        "利福平", "利福布汀", "卡马西平", "苯妥英", "苯巴比妥", "扑米酮", "依非韦伦", "圣约翰草",
    ),
    "bisphosphonate": (
        "alendronate", "risedronate", "ibandronate", "zoledronic", "pamidronate", "etidronate", "minodronic",
        "阿仑膦酸", "利塞膦酸", "伊班膦酸", "唑来膦酸", "帕米膦酸", "依替膦酸", "米诺膦酸",
    ),
    "polyvalent_cation": (
        "calcium", "ferrous", "iron", "magnesium", "aluminium", "aluminum", "antacid", "sucralfate",
        "钙", "碳酸钙", "枸橼酸钙", "铁", "硫酸亚铁", "琥珀酸亚铁", "镁", "铝", "抗酸", "硫糖铝",
    ),
    "denosumab": ("denosumab", "地舒单抗", "狄诺塞麦"),
    "teriparatide": ("teriparatide", "abaloparatide", "特立帕肽", "阿巴洛肽"),
    "romosozumab": ("romosozumab", "罗莫佐单抗", "罗莫单抗"),
    "vitamin_d_calcium_raising": (
        "calcitriol", "alfacalcidol", "cholecalciferol", "thiazide",
        "骨化三醇", "阿法骨化醇", "胆钙化醇", "维生素d", "噻嗪",
    ),
    "methotrexate": ("methotrexate", "甲氨蝶呤", "氨甲蝶呤"),
    "sulfamethoxazole_trimethoprim": (
        "sulfamethoxazole", "trimethoprim", "cotrimoxazole", "co-trimoxazole",
        "磺胺甲噁唑", "甲氧苄啶", "复方新诺明", "复方磺胺甲噁唑",
    ),
    "probenecid": ("probenecid", "丙磺舒"),
    "colchicine": ("colchicine", "秋水仙碱"),
    "strong_cyp3a4_pgp_inhibitor": (
        "clarithromycin", "erythromycin", "ketoconazole", "itraconazole", "voriconazole", "posaconazole",
        "ritonavir", "cobicistat", "cyclosporine", "ciclosporin", "verapamil", "diltiazem", "amiodarone",
        "grapefruit",
        "克拉霉素", "红霉素", "酮康唑", "伊曲康唑", "伏立康唑", "泊沙康唑", "利托那韦", "考比司他",
        "环孢素", "维拉帕米", "地尔硫䓬", "地尔硫卓", "胺碘酮", "西柚", "葡萄柚",
    ),
    "nephrotoxic": (
        "gentamicin", "amikacin", "tobramycin", "vancomycin", "amphotericin", "cisplatin",
        "contrast media", "iohexol", "iodixanol",
        "庆大霉素", "阿米卡星", "妥布霉素", "万古霉素", "两性霉素", "顺铂", "造影剂", "碘海醇", "碘克沙醇",
    ),
    "periop_antibiotic": (
        "cefazolin", "cefuroxime", "clindamycin", "metronidazole", "levofloxacin", "moxifloxacin",
        "头孢唑林", "头孢呋辛", "克林霉素", "甲硝唑", "左氧氟沙星", "莫西沙星",
    ),
}

#: Patient conditions / planned procedures a rule can key on.
KNOWN_CONDITIONS = {
    "renal_impairment", "esophageal_disorder", "hypocalcemia", "hypercalcemia", "severe_ckd",
    "recent_mi_or_stroke", "peptic_ulcer_history", "elderly", "planned_neuraxial_anesthesia",
    "planned_surgery", "hepatic_impairment", "pregnancy", "immobilized",
}


@dataclass(frozen=True)
class InteractionRule:
    rule_id: str
    title: str
    severity: str
    #: Every group must be satisfied by at least one of the patient's drugs.
    requires_all: tuple[tuple[str, ...], ...]
    mechanism: str
    management: str
    any_conditions: tuple[str, ...] = ()
    all_conditions: tuple[str, ...] = ()
    references: tuple[str, ...] = ("yaobi_ortho_rules",)
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "severity": self.severity,
            "mechanism": self.mechanism,
            "management": self.management,
            "tags": list(self.tags),
        }


def _g(*classes: str) -> tuple[str, ...]:
    return tuple(classes)


ORTHO_RULES: tuple[InteractionRule, ...] = (
    InteractionRule(
        "ORTHO-001", "NSAIDs 与抗凝/抗血小板药合用增加出血风险", "major",
        (_g("nsaid"), _g("anticoagulant", "antiplatelet", "aspirin")),
        "NSAIDs 抑制血小板环氧合酶并直接损伤胃黏膜，与抗凝/抗血小板作用叠加",
        "尽量避免；确需合用时使用最低有效剂量与最短疗程，加用 PPI，监测出血征象与 INR/血红蛋白，优先考虑对乙酰氨基酚或外用 NSAIDs",
        tags=("bleeding", "gi"),
    ),
    InteractionRule(
        "ORTHO-002", "NSAIDs + ACEI/ARB + 利尿剂 三重打击致急性肾损伤", "major",
        (_g("nsaid"), _g("acei_arb"), _g("diuretic")),
        "前列腺素依赖的入球小动脉扩张被抑制，同时出球小动脉张力下降且血容量不足",
        "避免三联；必须使用时限期最短、监测肌酐与电解质、维持水化，脱水或发热期间暂停 NSAIDs",
        tags=("renal",),
    ),
    InteractionRule(
        "ORTHO-003", "NSAIDs 与糖皮质激素合用增加消化道出血", "major",
        (_g("nsaid"), _g("corticosteroid")),
        "两者均削弱胃黏膜防御，溃疡与出血风险显著协同",
        "避免长期合用；必须合用时加用 PPI，评估幽门螺杆菌，监测黑便与贫血",
        tags=("bleeding", "gi"),
    ),
    InteractionRule(
        "ORTHO-004", "NSAIDs 与 SSRI/SNRI 合用增加上消化道出血", "major",
        (_g("nsaid"), _g("ssri_snri")),
        "SSRI/SNRI 耗竭血小板 5-HT 而削弱止血，与 NSAIDs 的黏膜损伤叠加",
        "评估必要性，加用 PPI，优先对乙酰氨基酚或外用 NSAIDs，告知出血征象",
        tags=("bleeding",),
    ),
    InteractionRule(
        "ORTHO-005", "阿片类与苯二氮䓬/镇静药/酒精/加巴喷丁类合用导致呼吸抑制", "contraindicated",
        (_g("opioid", "tramadol"), _g("benzodiazepine_sedative", "alcohol", "gabapentinoid")),
        "中枢与呼吸驱动的协同抑制，是阿片相关死亡的主要机制",
        "尽量避免同时处方；无法避免时取两者最低剂量、缩短疗程、配备纳洛酮并告知家属，禁止饮酒",
        tags=("respiratory", "fatal"),
    ),
    InteractionRule(
        "ORTHO-006", "曲马多与 SSRI/SNRI/MAOI 合用致血清素综合征与癫痫", "major",
        (_g("tramadol"), _g("ssri_snri", "maoi", "serotonergic_other")),
        "曲马多抑制 5-HT/NE 再摄取并降低癫痫阈值",
        "与 MAOI 合用属禁忌（含停药后 14 天内）；与 SSRI/SNRI 尽量改用其他镇痛方案，必须合用时监测躁动、高热、肌阵挛、反射亢进",
        tags=("serotonin", "seizure"),
    ),
    InteractionRule(
        "ORTHO-007", "对乙酰氨基酚长期高剂量与华法林合用升高 INR", "moderate",
        (_g("acetaminophen"), _g("anticoagulant")),
        "持续每日 ≥2 g 对乙酰氨基酚可干扰维生素 K 依赖凝血因子合成",
        "偶尔使用通常安全；连续使用超过数日应在 3-5 天内复查 INR 并相应调整华法林剂量",
        tags=("bleeding",),
    ),
    InteractionRule(
        "ORTHO-008", "酶诱导剂降低抗凝药与镇痛药疗效", "moderate",
        (_g("enzyme_inducer"), _g("anticoagulant", "opioid", "tramadol", "acetaminophen")),
        "CYP3A4/CYP2C9 与 P-gp 诱导加速代谢清除",
        "起始与停用诱导剂后均需重新评估：抗凝药监测 INR/抗Xa，阿片类监测镇痛不足或撤药症状",
        tags=("efficacy",),
    ),
    InteractionRule(
        "ORTHO-009", "双膦酸盐与钙/铁/镁/铝/抗酸剂同服显著降低吸收", "major",
        (_g("bisphosphonate"), _g("polyvalent_cation")),
        "多价阳离子在肠道与双膦酸盐螯合，生物利用度本已低于 1%",
        "口服双膦酸盐须晨起空腹以 200 ml 白水送服，服后至少 30-60 分钟保持直立且不进食其他药物、钙剂或食物",
        tags=("absorption",),
    ),
    InteractionRule(
        "ORTHO-010", "肾功能不全或食管疾病者使用双膦酸盐", "contraindicated",
        (_g("bisphosphonate"),),
        "经肾清除，肾功能不全时蓄积；食管排空延迟或狭窄时腐蚀性食管炎风险高",
        "CrCl < 30-35 ml/min 通常禁用；食管狭窄、贲门失弛缓或不能保持直立 30 分钟者禁用口服剂型，改评估其他抗骨吸收方案",
        any_conditions=("renal_impairment", "severe_ckd", "esophageal_disorder"),
        tags=("renal", "contraindication"),
    ),
    InteractionRule(
        "ORTHO-011", "地舒单抗在低钙血症或严重肾病中的重度低钙风险", "contraindicated",
        (_g("denosumab"),),
        "强效抑制破骨细胞骨吸收，钙释放锐减；肾功能不全者矫正能力更差",
        "给药前必须纠正低钙血症并确保充足钙与维生素 D 摄入；严重 CKD 者加密监测血钙（首月内多次）",
        any_conditions=("hypocalcemia", "severe_ckd", "renal_impairment"),
        tags=("electrolyte",),
    ),
    InteractionRule(
        "ORTHO-012", "特立帕肽与升高血钙的药物合用", "moderate",
        (_g("teriparatide"), _g("vitamin_d_calcium_raising")),
        "PTH 类似物本身升高血钙，与活性维生素 D 或噻嗪类叠加",
        "监测血钙与尿钙，避免同时给予高剂量活性维生素 D；已有高钙血症者禁用特立帕肽",
        tags=("electrolyte",),
    ),
    InteractionRule(
        "ORTHO-013", "罗莫佐单抗在近期心肌梗死或卒中患者中的心血管风险", "contraindicated",
        (_g("romosozumab"),),
        "临床试验观察到主要心血管不良事件增加",
        "近一年内发生心肌梗死或卒中者禁用；用药前评估心血管风险并与患者讨论替代方案",
        any_conditions=("recent_mi_or_stroke",),
        tags=("cardiovascular", "contraindication"),
    ),
    InteractionRule(
        "ORTHO-014", "长期糖皮质激素的骨丢失与感染风险", "major",
        (_g("corticosteroid"),),
        "抑制成骨细胞、增加骨吸收与钙排泄，同时抑制免疫",
        "预期泼尼松 ≥5 mg/d 持续 ≥3 个月者启动骨保护（钙、维生素 D，中高危加抗骨吸收药），并筛查感染与血糖",
        any_conditions=("immobilized", "elderly", "planned_surgery"),
        tags=("osteoporosis", "infection"),
    ),
    InteractionRule(
        "ORTHO-015", "抗凝/抗血小板药与椎管内麻醉或围术期管理", "contraindicated",
        (_g("anticoagulant", "antiplatelet"),),
        "椎管内穿刺或拔管时抗凝会导致硬膜外血肿与永久性截瘫",
        "按各药半衰期执行术前停药与穿刺/拔管时间窗（如低分子肝素预防量末次给药后 ≥12 h、治疗量 ≥24 h），由麻醉科主导决策，术后密切监测神经功能",
        any_conditions=("planned_neuraxial_anesthesia", "planned_surgery"),
        tags=("periop", "neuro"),
    ),
    InteractionRule(
        "ORTHO-016", "甲氨蝶呤与 NSAIDs 或复方新诺明合用致骨髓抑制", "major",
        (_g("methotrexate"), _g("nsaid", "sulfamethoxazole_trimethoprim", "probenecid", "aspirin")),
        "竞争性抑制肾小管排泌（NSAIDs、丙磺舒）或叠加抗叶酸作用（TMP-SMX）",
        "避免与复方新诺明合用；与 NSAIDs 合用时监测血常规、肝肾功能，保证叶酸补充，肾功能下降时减量或停用",
        tags=("myelosuppression",),
    ),
    InteractionRule(
        "ORTHO-017", "秋水仙碱与强 CYP3A4/P-gp 抑制剂合用致中毒", "contraindicated",
        (_g("colchicine"), _g("strong_cyp3a4_pgp_inhibitor")),
        "清除受阻使秋水仙碱蓄积，可致骨髓抑制、多器官衰竭甚至死亡",
        "肝或肾功能不全者禁止合用；其他人群须显著减量并延长给药间隔，避免西柚",
        tags=("toxicity", "fatal"),
    ),
    InteractionRule(
        "ORTHO-018", "围术期抗菌药与华法林或肾毒性药物的组合", "moderate",
        (_g("periop_antibiotic"), _g("anticoagulant", "nephrotoxic")),
        "抗菌药改变肠道菌群维生素 K 合成并抑制 CYP2C9；与肾毒性药物叠加损伤肾功能",
        "开始与停用抗菌药后 3-5 天复查 INR；联合肾毒性药物时监测肌酐与药物谷浓度，维持水化",
        tags=("periop", "renal"),
    ),
)


@dataclass
class RuleMatch:
    rule: InteractionRule
    matched_drugs: dict[str, list[str]] = field(default_factory=dict)
    matched_conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.rule.to_dict(),
            "matched_drugs": {k: sorted(set(v)) for k, v in self.matched_drugs.items()},
            "matched_conditions": self.matched_conditions,
            "source_id": self.rule.references[0],
        }


def classify(name: str) -> list[str]:
    """Return the classes a medication name belongs to."""
    text = (name or "").strip().lower()
    if not text:
        return []
    hits = []
    for class_name, members in DRUG_CLASSES.items():
        if any(member.lower() in text for member in members):
            hits.append(class_name)
    return hits


def classify_all(names: Iterable[str]) -> dict[str, list[str]]:
    """Map class -> the raw medication names that fall into it."""
    mapping: dict[str, list[str]] = {}
    for name in names:
        for class_name in classify(name):
            mapping.setdefault(class_name, []).append(name)
    return mapping


def evaluate(medications: Iterable[str], conditions: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Return every rule triggered by this medication list and patient state.

    Results are sorted most severe first. A rule with ``any_conditions`` only
    fires when at least one of those conditions is present, so a bisphosphonate
    alone is not flagged unless renal or oesophageal risk is recorded.
    """
    meds = [m for m in medications if m and str(m).strip()]
    present = classify_all(meds)
    condition_set = {str(c).strip().lower() for c in conditions if c}

    matches: list[RuleMatch] = []
    for rule in ORTHO_RULES:
        matched: dict[str, list[str]] = {}
        satisfied = True
        for group in rule.requires_all:
            group_hits = [name for cls in group for name in present.get(cls, [])]
            if not group_hits:
                satisfied = False
                break
            matched["/".join(group)] = group_hits
        if not satisfied:
            continue
        if rule.all_conditions and not set(rule.all_conditions).issubset(condition_set):
            continue
        matched_conditions = sorted(condition_set.intersection(rule.any_conditions))
        if rule.any_conditions and not matched_conditions:
            continue
        matches.append(RuleMatch(rule, matched, matched_conditions))

    matches.sort(key=lambda m: (SEVERITY_ORDER.get(m.rule.severity, 9), m.rule.rule_id))
    return [m.to_dict() for m in matches]


def blocking(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in matches if m.get("severity") in BLOCKING_SEVERITIES]


def rule_pack_summary() -> dict[str, Any]:
    return {
        "rule_count": len(ORTHO_RULES),
        "class_count": len(DRUG_CLASSES),
        "severities": sorted({r.severity for r in ORTHO_RULES}),
        "known_conditions": sorted(KNOWN_CONDITIONS),
        "review_required": "本规则包须经本机构药师/医师复核后启用",
    }
