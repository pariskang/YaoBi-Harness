# YaoBi-Harness

Yaobi-Harness 是面向医师的腰痹辅助诊疗智能体原型。它参考 Shanghan-Harness 的显式 harness 思路，将临床认知自主层与安全控制层分离：模型/子智能体负责主动规划、问诊、鉴别、辨证、病例检索、方剂草案和修复；Capability Broker、Evidence Ledger、Safety Gate 负责权限、证据、剂量来源、急症能力收回和医师逐味审核。

> 本项目仅用于临床决策支持研发与离线评测，不面向患者自动开具处方；含剂量方剂只能以 `draft_for_physician` 发布，并要求医师逐味审核。

## 快速体验

```bash
python -m yaobi_harness run --role physician --complaint "腰痛3月，久坐加重，右下肢麻木，无大小便异常" --allow-prescription
python -m yaobi_harness run --role patient --complaint "突发腰痛伴尿潴留和会阴麻木"
python -m yaobi_harness inspect-xlsx "沈钦荣腰痹200例.xlsx"
```

## 架构要点

* `ClinicalRunState`：保存问诊、风险、任务图、证据台账、Agent 轨迹和发布状态。
* `YaobiGraphRunner`：显式状态图，包含 intake、planner、urgent/normal 子图、critic、安全审查和发布门。
* `CapabilityBroker`：按角色、风险模式、审批状态和工具健康动态授权；急症时禁止中药处方、剂量、推拿/牵引建议。
* `ExpertCaseStore`：把 Excel 专家门诊数据定位为单一专家经验记忆库、复诊轨迹库和剂量条件分布库，而非疗效因果证据。
* `DoseAgent`：只有存在工具证据时才生成明确剂量；否则失败关闭或降级为非处方建议。
