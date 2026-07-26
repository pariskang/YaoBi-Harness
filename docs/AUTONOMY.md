# 自主性：模型到底自己做了什么

## 一句话

**规划和执行都由模型驱动，剂量和急症不由模型驱动。** 这不是能力不足，是刻意的边界。

## 分层现状

| 环节 | 谁决定 | 说明 |
| --- | --- | --- |
| 任务图规划 | **模型** | 提案经规则层逐条校验（Agent 白名单、技能工具表、角色、风险模式、依赖环），越权即整体驳回 |
| 红旗语义补充 | **模型** | 只能追加信号，不能清除规则层命中 |
| 追问生成 | **模型** | 受 `max_questions` 约束 |
| 西医鉴别 | **模型（工具循环）** | `yaobi.biomedical_differential` |
| 中医辨证 | **模型（工具循环）** | `yaobi.tcm_pattern` |
| 专家经验综合 | **模型（工具循环）** | `yaobi.expert_case_reasoning` |
| 方剂组成 | **模型（工具循环）** | `yaobi.formula_design`，只出药名，不出克数 |
| 用药相互作用 | 确定性 | 规则包可复核、可版本化，不需要模型的不确定性 |
| **逐味剂量** | **确定性，永不交给模型** | 分层中位数 → 最小样本量 → 离散度 → 授权药典范围逐味比对 |
| 急症行动计划 | 确定性 | 固定安全脚本是特性，不是缺陷 |
| 安全审查 | 确定性 + 模型追加异议 | 模型只能加问题，不能清除 |

## 工具调用循环（ReAct）

标记 `autonomous: true` 的技能不走硬编码逻辑：

```
system: 角色 + 风险模式 + 技能说明(instructions) + 输出 schema + 硬约束
  ↓
模型选工具 → CapabilityBroker 校验 → 执行 → 结果入证据台账 → evidence_id 回传
  ↓  (循环，上限 max_steps=6)
模型输出 JSON → schema 校验 → 剂量扫描 → 绑定 citations
```

五道闸门，任何一道不过就**整体回退确定性逻辑**，而不是带病放行：

1. **可见性**：工具 schema 按 `allowed_tools − forbidden_tools` 过滤——模型连越权工具的名字都看不到；即使编造了，Broker 也会拒绝。
2. **输出契约**：必须通过技能声明的 `output_schema`；不合规 → `schema_violation`。
3. **禁剂量**：输出中出现任何 `数字 + 克/g/mg` → `dose_in_output`，整体作废。
4. **预算**：LLM 调用数与 token 数受 `Budget` 限制；耗尽 → 回退。
5. **步数**：`max_steps` 封顶；超限 → 回退。

### 参数错误是可恢复的，不是失败

模型第一次常常传错参数。系统把这类错误与真正的工具故障**严格区分**：

| | 计入证据台账 | 触发熔断 | 阻断放行 | 返回给模型 |
| --- | --- | --- | --- | --- |
| 参数错误 / 未知工具 | ❌ | ❌ | ❌ | 参数 schema + 提示，可重试 |
| 工具执行失败 | ✅（`failed_tool_event`） | ✅ | ✅ | 错误信息 |

否则一次拼错字段就会让整份病例作废——这是接入真实模型后最先炸的地方。

### 观察结果按结构压缩

检索类工具可能返回几百条病例。观察结果若超过 `MAX_OBSERVATION_CHARS` 会被**按结构压缩**成计数与样例，
而不是截断 JSON 字符串——后者会给模型送去无法解析的半截 JSON。

## 技能：既是权限，也是规程

一个技能同时是两样东西：

```yaml
- skill_id: yaobi.tcm_pattern
  allowed_tools: [tcm_pattern_knowledge_search]   # ← 能力经纪强制执行
  forbidden_tools: []
  output_schema: PatternAssessment                 # ← 输出契约
  autonomous: true                                 # ← 是否交给模型自主执行
  instructions: |                                  # ← 模型读取的规程
    基于主诉与已知四诊信息辨证。
    1. primary_pattern 只给一个……
```

* `allowed_tools` 为空 = 不得调用任何工具；技能不在 manifest 中 = 无任何工具权限（fail-closed）。
* `instructions` 进入该 Agent 的 system prompt；`description` 进入规划器的技能目录。
* `autonomous` 决定是否启用工具循环——这个开关在**经过复核的策略文件**里，不在代码里。

查看：

```bash
python -m yaobi_harness skill list
python -m yaobi_harness skill show yaobi.tcm_pattern
```

## 专家 xlsx → 技能

以前专家病例只被用于"检索相似病例"和"剂量分布"——**经验并没有进入推理**。
`skill build-expert` 把语料挖掘成技能：

```bash
python -m yaobi_harness skill build-expert --xlsx ./authorized.xlsx --merge
python -m yaobi_harness run --skill-manifest yaobi_harness/skills/manifest.yaml --complaint "..."
```

挖掘内容（按证型分组）：核心药（出现率 ≥60%）、随证加减药、常用治法、合并西药、常做检查、
常见合并症/术史、年龄分布、反例例数，以及语料级的复诊轨迹。

生成的技能 `skill_id` 就是 `yaobi.expert_case_reasoning`——它**替换**内置技能，
所以重新生成等于升级已有 Agent，而不是新增一条并行路径。

### 隐私与安全约束

* 只输出**聚合统计**，绝不含个体自由文本；低于 `min_support`（默认 2）的取值被抑制。
* 生成前跑 PHI 自检（超长行、手机号、身份证、邮箱），不通过则拒绝写文件。
* **剂量不进入技能说明**。推理 Agent 被禁止输出克数，把中位数放进它的提示既自相矛盾，
  又会诱发被闸门拒绝的输出。剂量统计只经 `herb_dose_distribution` 进入确定性剂量链路。

```
expert_practice_profile  →  {herb, count, share}          →  推理 Agent
herb_dose_distribution   →  {median_g, p25, p75, n, ...}  →  剂量链路（仅 yaobi.dose_generation）
```

## 还不是自主的部分

* **多轮问诊**仍是单轮 + `resume`，模型不能主动发起对话轮次。
* **工具集是固定的**——模型不能创建新工具或修改自己的技能。
* **急症脚本固定**：红旗命中后走确定性行动计划，不交给模型措辞。
* **规则包与药典校验不可被模型绕过**，这是设计目标而非待办。
