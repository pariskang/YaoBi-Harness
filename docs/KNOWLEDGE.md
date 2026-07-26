# 授权知识库接入

本仓库**只包含代码与许可模型，不包含任何第三方受版权内容**。指南、药典与相互作用数据由部署方按自身持有的授权自行摄取；
知识库在**写入时**强制执行许可，而不是在文档里提醒。

## 三条硬规则

1. **非商业数据不能进入商业部署。** `YAOBI_DEPLOYMENT_MODE=commercial` 时，CC BY-NC-SA 来源（WHO、DDInter）的
   任何写入都会抛 `LicenseError`。
2. **只读来源永不入库全文。** AAOS、中华医学会、NMPA 文件、香港衞生署等标记为 `link_only`：可存标题、版本、
   发布日期、链接和人工摘录的建议要点，`body` 字段会被存储层直接丢弃，且强制要求引用 URL。
3. **须授权来源默认关闭。** NICE、《中国药典》2025、USP–NF、EP、DrugBank、BNF/Stockley's、NMPA 说明书在
   未登记授权声明前完全不可用——不是降级，是拒绝。

## 来源目录

`python -m yaobi_harness knowledge sources` 会列出全部来源及当前部署是否可用。

| source_id | 来源 | 类型 | 复用等级 | 商用 |
| --- | --- | --- | --- | --- |
| `openfda` | openFDA Drug Label API | 说明书 | public_domain (CC0) | ✅ |
| `dailymed` | DailyMed SPL | 说明书 | public_domain | ✅ |
| `rxnorm` | RxNorm / RxClass | 术语 | open_attribution | ✅ |
| `pmc_oa` | PMC Open Access Subset | 文献 | open_attribution（逐篇校验） | ⚠️ 逐篇 |
| `vadod_cpg` | VA/DoD CPG | 指南 | open_attribution | ⚠️ 查第三方材料 |
| `who_guidelines` | WHO 指南 | 指南 | noncommercial (CC BY-NC-SA 3.0 IGO) | ❌ |
| `who_intl_pharmacopoeia` | WHO 国际药典 12 版 | 药典 | noncommercial | ❌ |
| `ddinter` | DDInter 2.0 | 相互作用 | noncommercial (CC BY-NC-SA 4.0) | ❌ |
| `aaos_cpg` | AAOS CPG/AUC | 指南 | link_only | ❌ 需授权 |
| `hk_dh_msk` | 香港衞生署肌骨参考框架 | 指南 | link_only | ❌ |
| `cma_guidelines` | 中华医学会/中国医师协会 | 指南 | link_only | ❌ 需授权 |
| `nmpa_documents` | 卫健委/NMPA/CDE 文件 | 指南 | link_only | ❌ |
| `nice` | NICE Syndication API | 指南 | credentialed | 需申请 |
| `chp_2025` | 《中国药典》2025 年版 | 药典 | credentialed | 需授权 |
| `nmpa_labels` | NMPA 批准说明书 | 说明书 | credentialed | 需授权 |
| `usp_nf` / `ph_eur` | USP–NF / 欧洲药典 | 药典 | credentialed | 需订阅 |
| `drugbank` | DrugBank | 相互作用 | credentialed | 需授权 |
| `bnf_stockley` | BNF / Stockley's | 相互作用 | credentialed | 需授权 |
| `yaobi_ortho_rules` | 内置骨科相互作用规则包 | 相互作用 | MIT（本仓库） | ✅ |
| `yaobi_tcm_incompatibility` | 内置十八反/十九畏/妊娠禁忌 | 相互作用 | MIT（本仓库） | ✅ |

## 配置

```bash
export YAOBI_DEPLOYMENT_MODE=research_noncommercial   # 或 commercial
export YAOBI_LICENSE_ATTESTATIONS='{
  "chp_2025": {"licensee": "某某医院", "license_reference": "CHP-2025-LIC-001", "expires": "2027-01-01"},
  "nice":     {"licensee": "某某医院", "license_reference": "NICE-SYND-2026-042"}
}'
```

## 构建

```bash
# 1) 抓取本部署有权抓取的一切（openFDA + DailyMed + 内置规则包；NICE 需授权）
python -m yaobi_harness knowledge build --store ./knowledge.db --cache-dir ./.kcache

# 2) 摄取自有授权文件
python -m yaobi_harness knowledge ingest-file --store ./knowledge.db \
    --source chp_2025 --kind dose_ranges --path ./authorized/chp2025_herbs.json --version 2025
python -m yaobi_harness knowledge ingest-file --store ./knowledge.db \
    --source cma_guidelines --kind citations --path ./authorized/cma_citations.json
python -m yaobi_harness knowledge ingest-file --store ./knowledge.db \
    --source ddinter --kind interactions --path ./downloads/ddinter_full.csv    # 仅非商业模式

# 3) 查看与自检
python -m yaobi_harness knowledge stats --store ./knowledge.db
python -m yaobi_harness knowledge check-interactions --medications 布洛芬 华法林 --conditions elderly

# 4) 在临床运行中使用
python -m yaobi_harness run --role physician --knowledge-store ./knowledge.db --complaint "..."
```

## 文件格式

摄取器接受 JSON（对象或列表）与 CSV/TSV，字段名有多个别名，便于直接映射已有导出。

```jsonc
// --kind dose_ranges
{"dose_ranges": [
  {"substance": "独活", "min": 3, "max": 10, "unit": "g", "population": "adult",
   "route": "oral", "basis": "《中国药典》2025 一部"}
]}

// --kind guidelines  /  --kind citations（link_only 来源用 citations，body 会被丢弃且必须给 url）
{"guidelines": [
  {"id": "NG59", "title": "Low back pain and sciatica in over 16s", "topic": "low back pain",
   "url": "https://www.nice.org.uk/guidance/ng59", "published": "2020-12-11",
   "evidence_grade": "NICE", "summary": "...", "recommendations": ["...", "..."]}
]}

// --kind interactions
{"interactions": [
  {"subject": "warfarin", "object": "ibuprofen", "severity": "major",
   "mechanism": "...", "management": "...", "evidence": "label"}
]}
```

## 数据进入临床决策的位置

| 层级 | 来源 | 生效点 |
| --- | --- | --- |
| 骨科决策 | NICE / VA-DoD / 授权 AAOS 与中国指南 | `clinical_guideline_search`：命中即为 `guideline_or_standard` 级证据；未命中仍标记 `stub_not_for_clinical_use` |
| 剂量与说明书 | 《中国药典》/ NMPA / openFDA / DailyMed | `pharmacopeia_check` 逐味比对**拟用剂量**；授权药典优先级高于本地配置表 |
| 药品身份 | RxNorm / RxClass (ATC) | `drug_normalize`；未联网时回退到规则包的中英双语类别匹配 |
| 相互作用一级 | 说明书直接记载 | `drug_label_lookup` 返回 `drug_interactions` / `contraindications` 章节及标签版本 |
| 相互作用二级 | 内置骨科规则包 + 授权 DrugBank/BNF/DDInter | `drug_interaction_check`；`contraindicated`/`major` 会阻断并把放行状态抬到 `needs_examination` |
| 中药配伍 | 内置十八反/十九畏/妊娠禁忌 | `interaction_check`；违反即禁止生成含剂量草案 |

优先级由 `store._SOURCE_PRIORITY` 决定：`chp_2025` > `nmpa_labels` > `ph_eur` > `usp_nf` >
`who_intl_pharmacopoeia` > `openfda` > `dailymed`。

## 内置骨科相互作用规则包

18 条规则、30 个药物类别，覆盖你列出的全部重点组合：

| 规则 | 组合 | 级别 |
| --- | --- | --- |
| ORTHO-001 | NSAIDs + 抗凝/抗血小板 | major |
| ORTHO-002 | NSAIDs + ACEI/ARB + 利尿剂（三重打击） | major |
| ORTHO-003 / 004 | NSAIDs + 糖皮质激素 / SSRI-SNRI | major |
| ORTHO-005 | 阿片类 + 苯二氮䓬/镇静药/酒精/加巴喷丁类 | contraindicated |
| ORTHO-006 | 曲马多 + SSRI/SNRI/MAOI | major |
| ORTHO-007 | 对乙酰氨基酚（长期高剂量）+ 华法林 | moderate |
| ORTHO-008 | 酶诱导剂 + 抗凝/镇痛药 | moderate |
| ORTHO-009 | 双膦酸盐 + 钙/铁/镁/铝/抗酸剂 | major |
| ORTHO-010 | 双膦酸盐 + 肾功能不全/食管疾病 | contraindicated |
| ORTHO-011 | 地舒单抗 + 低钙血症/严重肾病 | contraindicated |
| ORTHO-012 | 特立帕肽 + 升高血钙药物 | moderate |
| ORTHO-013 | 罗莫佐单抗 + 近期心梗/卒中 | contraindicated |
| ORTHO-014 | 长期糖皮质激素（骨丢失与感染） | major |
| ORTHO-015 | 抗凝/抗血小板 + 椎管内麻醉/围术期 | contraindicated |
| ORTHO-016 | 甲氨蝶呤 + NSAIDs / 复方新诺明 / 丙磺舒 | major |
| ORTHO-017 | 秋水仙碱 + 强 CYP3A4/P-gp 抑制剂 | contraindicated |
| ORTHO-018 | 围术期抗菌药 + 华法林 / 肾毒性药物 | moderate |

条件门控规则（ORTHO-010/011/013/014/015）只在 `facts.conditions` 提供相应状态时触发，可用条件见
`ortho_interactions.KNOWN_CONDITIONS`。

**这个规则包必须经本机构药师/医师复核后启用**，且不能替代完整的相互作用数据库——未接入授权 DDI 库时，
`drug_interaction_check` 会在结果里显式声明覆盖范围有限。

## 答案中的出处

`render()` 的医师视图与研究者视图包含 `citations`，逐条给出来源、许可、版本、发布日期、检索时间与链接；
`run_meta.knowledge` 记录本次运行启用了哪些来源以及当前许可模式。
