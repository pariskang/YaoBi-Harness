"""Catalogue of external knowledge sources and their reuse terms.

Each entry records what the source covers, how it is accessed and — most
importantly — what this project is allowed to do with it. The terms below
reflect published licences at the time of writing; an operator deploying this
system is responsible for re-checking them, which is why every entry carries the
URL of the governing statement.

Nothing here downloads content. Enabling a source only makes its connector
available; ingestion is an explicit, operator-run step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .licensing import Reuse, SourceLicense


@dataclass(frozen=True)
class KnowledgeSource:
    source_id: str
    name: str
    kind: str  # guideline | label | terminology | interaction | pharmacopoeia | literature
    license: SourceLicense
    access: str  # api | bulk_download | operator_supplied
    coverage: str = ""
    connector: str = ""
    default_endpoint: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "kind": self.kind,
            "license": self.license.license_name,
            "reuse": self.license.reuse.value,
            "commercial_ok": self.license.allows_commercial,
            "access": self.access,
            "coverage": self.coverage,
            "connector": self.connector,
            "license_url": self.license.url,
            "notes": self.notes,
        }


def _src(**kwargs: Any) -> KnowledgeSource:
    return KnowledgeSource(**kwargs)


# --------------------------------------------------------------------- catalog

SOURCE_CATALOG: dict[str, KnowledgeSource] = {s.source_id: s for s in [
    # ---------------------------------------------------------------- open API
    _src(
        source_id="openfda",
        name="openFDA Drug Label API",
        kind="label",
        access="api",
        connector="openfda",
        default_endpoint="https://api.fda.gov/drug/label.json",
        coverage="FDA 结构化说明书：相互作用、禁忌、剂量、特殊人群、警告",
        license=SourceLicense(
            license_name="CC0 1.0 (public domain), excluding flagged third-party fields",
            reuse=Reuse.PUBLIC_DOMAIN,
            attribution="openFDA, U.S. Food and Drug Administration",
            notice="openFDA 数据不构成 FDA 背书；个别第三方字段除外。",
            url="https://open.fda.gov/license/",
        ),
        notes="最稳妥的开放药品知识底座；作为剂量与相互作用的一级证据。",
    ),
    _src(
        source_id="dailymed",
        name="DailyMed Structured Product Labeling",
        kind="label",
        access="api",
        connector="dailymed",
        default_endpoint="https://dailymed.nlm.nih.gov/dailymed/services/v2",
        coverage="美国现行说明书全量 SPL（XML），可按日/周/月/全量下载",
        license=SourceLicense(
            license_name="U.S. NLM public access",
            reuse=Reuse.PUBLIC_DOMAIN,
            attribution="DailyMed, U.S. National Library of Medicine",
            notice="保留原始标签版权声明与 setid/版本。",
            url="https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm",
        ),
    ),
    _src(
        source_id="rxnorm",
        name="RxNorm / RxClass (RxNav)",
        kind="terminology",
        access="api",
        connector="rxnorm",
        default_endpoint="https://rxnav.nlm.nih.gov/REST",
        coverage="药品标准名、成分、剂型、强度、品牌映射与 ATC 分类",
        license=SourceLicense(
            license_name="RxNorm core vocabulary — no licence required; mapped source vocabularies keep their own terms",
            reuse=Reuse.OPEN_ATTRIBUTION,
            attribution="RxNorm, U.S. National Library of Medicine",
            notice="仅使用 RxNorm 核心词表；映射到受限外部词表时须另行遵守其许可。",
            url="https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html",
        ),
        notes="药品身份层：解决商品名/通用名/盐型/剂型不一致。",
    ),
    _src(
        source_id="pmc_oa",
        name="PubMed Central Open Access Subset",
        kind="literature",
        access="api",
        connector="generic_file",
        default_endpoint="https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/",
        coverage="骨科指南、共识与系统综述全文",
        license=SourceLicense(
            license_name="Per-article CC BY / CC BY-NC / other — must be filtered article by article",
            reuse=Reuse.OPEN_ATTRIBUTION,
            attribution="PubMed Central Open Access Subset",
            notice="逐篇校验许可；NC 条款的文章在商业模式下不可用。",
            url="https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/",
        ),
    ),

    # ------------------------------------------------------- non-commercial
    _src(
        source_id="who_guidelines",
        name="WHO Guidelines and Technical Documents",
        kind="guideline",
        access="operator_supplied",
        connector="generic_file",
        coverage="肌骨健康、康复、慢性腰痛、跌倒预防、基本药物",
        license=SourceLicense(
            license_name="CC BY-NC-SA 3.0 IGO (typical; verify per publication)",
            reuse=Reuse.NONCOMMERCIAL,
            attribution="World Health Organization",
            share_alike=True,
            notice="非商业使用；须署名 WHO、采用相同或相近许可、附免责声明，且不得暗示 WHO 认可本系统结论。",
            url="https://www.who.int/about/policies/publishing/copyright",
        ),
    ),
    _src(
        source_id="who_intl_pharmacopoeia",
        name="The International Pharmacopoeia (WHO), 12th ed.",
        kind="pharmacopoeia",
        access="operator_supplied",
        connector="generic_file",
        coverage="原料药、制剂、检验方法、杂质与通则（以基本药物为主）",
        license=SourceLicense(
            license_name="WHO copyright, typically CC BY-NC-SA 3.0 IGO",
            reuse=Reuse.NONCOMMERCIAL,
            attribution="World Health Organization",
            share_alike=True,
            notice="覆盖面窄于《中国药典》/USP-NF/EP，主要面向质量控制。",
            url="https://www.who.int/teams/health-product-policy-and-standards/standards-and-specifications/norms-and-standards-for-pharmaceuticals/international-pharmacopoeia",
        ),
    ),
    _src(
        source_id="ddinter",
        name="DDInter 2.0",
        kind="interaction",
        access="operator_supplied",
        connector="ddinter",
        coverage="约 30 万条 DDI、机制、风险等级、处理建议、药物-食物相互作用",
        license=SourceLicense(
            license_name="CC BY-NC-SA 4.0",
            reuse=Reuse.NONCOMMERCIAL,
            attribution="DDInter 2.0, Central South University",
            share_alike=True,
            notice="仅限非商业使用；数据快照需与最新说明书交叉校验。",
            url="https://ddinter2.scbdd.com/",
        ),
        notes="开放型结构化 DDI 库首选；商业部署须另行取得许可。",
    ),
    _src(
        source_id="vadod_cpg",
        name="VA/DoD Clinical Practice Guidelines",
        kind="guideline",
        access="operator_supplied",
        connector="generic_file",
        coverage="腰痛、髋膝骨关节炎、截肢康复、围术期、疼痛与阿片类药物",
        license=SourceLicense(
            license_name="U.S. Government work (verify third-party material in each document)",
            reuse=Reuse.OPEN_ATTRIBUTION,
            attribution="U.S. Department of Veterans Affairs / Department of Defense",
            notice="逐份核查文档内嵌的第三方材料。",
            url="https://www.healthquality.va.gov/",
        ),
    ),

    # ----------------------------------------------------------- link-only
    _src(
        source_id="aaos_cpg",
        name="AAOS Clinical Practice Guidelines / AUC",
        kind="guideline",
        access="operator_supplied",
        connector="citation_only",
        coverage="骨折、关节炎、关节置换、运动损伤，骨科专科性最强",
        license=SourceLicense(
            license_name="AAOS Terms of Use — personal, non-commercial access; other uses need prior permission",
            reuse=Reuse.LINK_ONLY,
            attribution="American Academy of Orthopaedic Surgeons",
            notice="仅存标题、版本、链接与人工摘录的建议要点；不得批量入库全文或用于模型训练。",
            url="https://www.aaos.org/about/meet-aaos/aaos-policies/organizational-policies/website-disclaimer/terms-of-use/",
        ),
    ),
    _src(
        source_id="hk_dh_msk",
        name="香港衞生署 肌骨疾病基層醫療參考概覽",
        kind="guideline",
        access="operator_supplied",
        connector="citation_only",
        coverage="骨质疏松、腰痛、膝痛、跌倒，贴近中文临床场景",
        license=SourceLicense(
            license_name="Free to read; no explicit open redistribution licence found",
            reuse=Reuse.LINK_ONLY,
            attribution="Department of Health, HKSAR",
            notice="可引用与链接，不默认入库全文。",
            url="https://www.fhb.gov.hk/pho/",
        ),
    ),
    _src(
        source_id="cma_guidelines",
        name="中华医学会 / 中国医师协会 骨科相关指南",
        kind="guideline",
        access="operator_supplied",
        connector="citation_only",
        coverage="中国骨科实践、围术期、骨质疏松、创伤",
        license=SourceLicense(
            license_name="Copyright held by the issuing society or journal; free reading is not a reuse grant",
            reuse=Reuse.LINK_ONLY,
            attribution="中华医学会 / 中国医师协会",
            notice="临床本地化必需，但入库全文须先取得授权；未授权时仅保留引用与链接。",
            url="https://www.cma.org.cn/",
        ),
    ),
    _src(
        source_id="nmpa_documents",
        name="国家卫健委 / NMPA / CDE 公开文件",
        kind="guideline",
        access="operator_supplied",
        connector="citation_only",
        coverage="临床路径、合理用药、药物技术指导原则",
        license=SourceLicense(
            license_name="Government publication; no blanket bulk-reuse declaration equivalent to CC0/OGL",
            reuse=Reuse.LINK_ONLY,
            attribution="国家药品监督管理局 / 国家卫生健康委员会",
            notice="保留原文链接与版本号，不直接全文商业分发。",
            url="https://www.nmpa.gov.cn/",
        ),
    ),

    # --------------------------------------------------------- credentialed
    _src(
        source_id="nice",
        name="NICE Guidance (Syndication API)",
        kind="guideline",
        access="api",
        connector="nice",
        default_endpoint="https://api.nice.org.uk/services/syndication",
        coverage="骨关节炎、髋部骨折、脊柱损伤、腰痛与坐骨神经痛、关节置换、VTE 预防、骨质疏松、慢性疼痛",
        license=SourceLicense(
            license_name="NICE UK Open Content Licence / syndication agreement (application required)",
            reuse=Reuse.CREDENTIALED,
            attribution="National Institute for Health and Care Excellence (NICE)",
            notice="须经 NICE 申请并核实地域、用途与第三方内容限制；不得暗示 NICE 认可本系统结论。",
            url="https://www.nice.org.uk/reusing-our-content/nice-syndication-api",
        ),
        notes="骨科覆盖最完整，建议作为主指南库。",
    ),
    _src(
        source_id="chp_2025",
        name="《中华人民共和国药典》2025 年版",
        kind="pharmacopoeia",
        access="operator_supplied",
        connector="generic_file",
        coverage="中国法定药品标准，含中药材与饮片用量",
        license=SourceLicense(
            license_name="Copyright: 国家药典委员会 — registered/purchased access, no bulk reuse grant",
            reuse=Reuse.CREDENTIALED,
            attribution="国家药典委员会",
            notice="2025 年版自 2025-10-01 施行；法律效力不等于批量复制或模型训练授权。",
            url="https://english.nmpa.gov.cn/2025-06/11/c_1102157.htm",
        ),
        notes="中药剂量范围的本地权威依据，必须持授权后由部署方摄取。",
    ),
    _src(
        source_id="nmpa_labels",
        name="NMPA 批准药品说明书",
        kind="label",
        access="operator_supplied",
        connector="generic_file",
        coverage="中国适应证、剂量、禁忌、特殊人群",
        license=SourceLicense(
            license_name="Rights held by marketing authorisation holders / NMPA",
            reuse=Reuse.CREDENTIALED,
            attribution="国家药品监督管理局",
            notice="中国用药剂量的最终本地依据；须记录说明书版本与核准日期。",
            url="https://www.nmpa.gov.cn/",
        ),
    ),
    _src(
        source_id="usp_nf",
        name="USP–NF",
        kind="pharmacopoeia",
        access="operator_supplied",
        connector="generic_file",
        license=SourceLicense(
            license_name="Subscription, copyright protected",
            reuse=Reuse.CREDENTIALED,
            attribution="United States Pharmacopeia",
            url="https://www.usp.org/",
        ),
    ),
    _src(
        source_id="ph_eur",
        name="European Pharmacopoeia (12th ed.)",
        kind="pharmacopoeia",
        access="operator_supplied",
        connector="generic_file",
        license=SourceLicense(
            license_name="365-day online licence (EDQM)",
            reuse=Reuse.CREDENTIALED,
            attribution="EDQM, Council of Europe",
            url="https://www.edqm.eu/en/european-pharmacopoeia-all-you-need-to-know",
        ),
    ),
    _src(
        source_id="drugbank",
        name="DrugBank Clinical / DDI",
        kind="interaction",
        access="operator_supplied",
        connector="generic_file",
        coverage="DDI、机制、严重度、证据等级、处理建议",
        license=SourceLicense(
            license_name="Academic or commercial licence required for any use or redistribution",
            reuse=Reuse.CREDENTIALED,
            attribution="DrugBank",
            notice="网页可检索不等于可抓取或用于模型训练。",
            url="https://go.drugbank.com/academic_research",
        ),
    ),
    _src(
        source_id="bnf_stockley",
        name="BNF / Stockley's Drug Interactions",
        kind="interaction",
        access="operator_supplied",
        connector="generic_file",
        coverage="相互作用严重度、处理措施与临床意义",
        license=SourceLicense(
            license_name="Proprietary content / API licence",
            reuse=Reuse.CREDENTIALED,
            attribution="Pharmaceutical Press / BNF",
            url="https://www.medicinescomplete.com/",
        ),
    ),

    # ------------------------------------------------------------- in-repo
    _src(
        source_id="yaobi_ortho_rules",
        name="Yaobi 骨科相互作用规则包",
        kind="interaction",
        access="operator_supplied",
        connector="builtin",
        coverage="骨科高频高危组合：NSAIDs、抗凝/抗血小板、阿片类、双膦酸盐、骨质疏松生物制剂、围术期",
        license=SourceLicense(
            license_name="MIT (this repository)",
            reuse=Reuse.PUBLIC_DOMAIN,
            attribution="Yaobi-Harness contributors",
            notice="本规则包是对公开临床事实的自有编码，须由本机构药师/医师复核后启用。",
        ),
        notes="随仓库分发的唯一相互作用内容；引用的机制与处理建议应与说明书核对。",
    ),
    _src(
        source_id="yaobi_tcm_incompatibility",
        name="Yaobi 十八反/十九畏/妊娠禁忌规则包",
        kind="interaction",
        access="operator_supplied",
        connector="builtin",
        coverage="经典中药配伍禁忌",
        license=SourceLicense(
            license_name="MIT (this repository)",
            reuse=Reuse.PUBLIC_DOMAIN,
            attribution="Yaobi-Harness contributors",
        ),
    ),
]}

#: Sources that are safe to enable with no operator action at all.
DEFAULT_ENABLED = ("openfda", "dailymed", "rxnorm", "yaobi_ortho_rules", "yaobi_tcm_incompatibility")


def get_source(source_id: str) -> KnowledgeSource:
    source = SOURCE_CATALOG.get(source_id)
    if source is None:
        raise KeyError(f"unknown knowledge source {source_id!r}; known: {sorted(SOURCE_CATALOG)}")
    return source


def catalog_summary() -> list[dict[str, Any]]:
    return [source.to_dict() for source in SOURCE_CATALOG.values()]
