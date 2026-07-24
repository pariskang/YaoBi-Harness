# YaoBi-Harness

Yaobi-Harness 当前定位为 **V0.0 安全重构骨架**，不是可用于真实临床决策或患者处方的 V0.1 完整系统。它保留 Protocol V2.0 的方向：临床认知层由 Agent/LLM 规划，控制层负责权限、证据、预算、故障关闭和医师审核；本仓库当前实现的是可测试的离线控制骨架，后续可替换为 LangGraph + LLM 节点。

> 严禁把原始 Excel 身份数据提交、打包或直接返回给模型。病例检索只能使用脱敏 ETL 后的结构化字段；含剂量方剂只能以 `draft_for_physician` 作为医师草案，未逐味审核不得发布为最终处方。

## 快速体验

```bash
python -m yaobi_harness run --role physician --complaint "腰痛3月，久坐加重，右下肢麻木，无大小便异常" --allow-prescription
python -m yaobi_harness run --role patient --complaint "突发腰痛伴尿潴留和会阴麻木"
python -m yaobi_harness inspect-xlsx /path/to/authorized_deidentified_or_local_raw.xlsx
```

## 本次骨架具备的硬安全能力

* `ClinicalRunState`：保存问诊、风险、任务图、证据台账、Agent 轨迹、预算和发布状态。
* `CapabilityBroker`：按角色、风险模式、工具健康和预算动态授权；急症和患者端均禁止方剂/剂量工具。
* `ExpertCaseStore`：读取 Excel 后删除姓名、病案号、地址、医师工号等直接标识，只返回研究 ID 与授权临床字段。
* `red_flag_evidence_search`：支持否定语境过滤，并覆盖马尾、感染/肿瘤、骨折、进展神经缺损和胸痛呼吸困难等非腰痛急症信号。
* `DoseAgent`：按证型/年龄分层检索剂量；任一药味缺少剂量依据、特殊人群信息缺失、相互作用或风险药专项审查失败时降级为非处方建议。
* `SkillRegistry`：加载 Skill manifest 并执行角色、allowed tools 和 forbidden tools 约束。

## 尚未完成

真实 LLM 自主规划、LangGraph interrupt/resume、真实指南/药典/相互作用数据库、逐断言 CitationGuard、医师 UI 审批和大规模安全评测仍未实现；因此本项目不能对外宣称为临床可用系统。
