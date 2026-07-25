# YaoBi-Harness

Yaobi-Harness 是一个 **证据受控的骨科/腰痹多智能体临床决策支持骨架**。V0.1 在 V0.0 安全骨架的基础上补齐了
自主规划层和 LLM 接入，同时保持"认知层可换、控制层不可绕过"的分层：

* **认知层**：可由 LLM 驱动的规划、语义红旗补充、追问生成与对抗式安全审查。
* **控制层**：能力经纪（角色/风险/技能/熔断/预算）、证据台账与等级、逐断言引用校验、放行状态机、医师审核闭环。

LLM 在本系统中是**只能加安全、不能减安全**的顾问角色：它可以提议计划、追加红旗、追加安全异议，但不能发明
Agent、不能触及技能未授权的工具、不能清除规则层命中的风险信号、不能生成剂量。未配置模型时，系统以完全确定性的
规则路径运行。

> 严禁把原始 Excel 身份数据提交、打包或直接返回给模型。病例检索只能使用脱敏 ETL 后的结构化字段；含剂量方剂只能以
> `draft_for_physician` 作为医师草案，未逐味审核签名不得发布为最终处方。

## 快速开始

```bash
pip install -e .

# 稳定假名化密钥是硬性要求：缺失时加载病例库会直接报错，而不是静默使用随机密钥
export YAOBI_DEID_KEY="$(openssl rand -hex 32)"

python -m yaobi_harness run --role physician \
  --complaint "腰痛3月，久坐加重，右下肢麻木，无大小便异常" --allow-prescription
python -m yaobi_harness run --role patient --complaint "突发腰痛伴尿潴留和会阴麻木"
python -m yaobi_harness inspect-xlsx /path/to/authorized_deidentified_or_local_raw.xlsx
python -m yaobi_harness llm-check
```

结构化事实（年龄、肝肾功能、用药过敏、医师签名等）通过 `--facts` / `--facts-file` 传入：

```bash
python -m yaobi_harness run --role physician --allow-prescription \
  --complaint "腰痛3月，刺痛固定" \
  --facts '{"special_population":{"pregnancy":false,"age":63,"renal":"normal","liver":"normal"},
            "medications_confirmed":true,"allergies_confirmed":true}'
```

## 接入 LLM

支持 **Azure OpenAI / Poe / MiniMax / LiteLLM**，四家共用 OpenAI 兼容协议，适配器只用标准库 HTTP，
不引入额外依赖。端点与 API 版本均可通过环境变量覆盖。

| provider | 必需变量 | 可选变量 |
| --- | --- | --- |
| `azure` | `AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_ENDPOINT`、`AZURE_OPENAI_DEPLOYMENT` | `AZURE_OPENAI_API_VERSION`（默认 `2024-10-21`） |
| `poe` | `POE_API_KEY` | `POE_MODEL`（默认 `Claude-Sonnet-4.5`）、`POE_BASE_URL` |
| `minimax` | `MINIMAX_API_KEY` | `MINIMAX_MODEL`、`MINIMAX_BASE_URL`、`MINIMAX_GROUP_ID` |
| `litellm` | `LITELLM_MODEL` | `LITELLM_API_KEY`、`LITELLM_BASE_URL`（默认 `http://localhost:4000/v1`） |

```bash
export YAOBI_LLM_PROVIDER=azure
export AZURE_OPENAI_API_KEY=...  AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com  AZURE_OPENAI_DEPLOYMENT=gpt-4o
python -m yaobi_harness llm-check
python -m yaobi_harness run --role physician --complaint "..." 

# 或在命令行临时指定
python -m yaobi_harness run --llm-provider poe --llm-model Claude-Sonnet-4.5 --complaint "..."
```

通用调节项：`YAOBI_LLM_TIMEOUT`(秒)、`YAOBI_LLM_RETRIES`、`YAOBI_LLM_CA_BUNDLE`。
未设置 `YAOBI_LLM_PROVIDER` 时使用 `NullLLMClient`，全流程确定性运行；显式指定了 provider 但凭据不全会**直接报错**，
避免配置错误伪装成"正常的规则输出"。

## 安全能力

* **红旗筛查**：子句级否定/家族史/假设语境判定，覆盖马尾、心肺、DVT/PE、感染肿瘤、化脓性关节炎/骨髓炎、骨折、
  进行性神经缺损、脊髓型颈椎病、骨筋膜室综合征；硬信号立即升级，弱信号走"需线下检查"而非丢弃；不确定时向上升级。
* **剂量安全**：分层专家剂量 → 最小样本量/离散度/异常值 → **拟用剂量与授权药典范围逐味比对** → 特殊人群 →
  用药过敏确认 → 十八反/十九畏/妊娠禁忌；任一不通过即降级为 `treatment_advice_only`，不产出克数。
* **能力经纪**：角色/风险模式/技能清单/熔断/预算按序检查，**被拒绝的调用不扣预算**；技能缺失即拒绝（fail-closed）。
* **证据台账**：等级由**工具自身**声明，占位数据源记为 `stub_not_for_clinical_use`，不可被提升为指南级；
  逐断言引用校验（CitationGuard）对高风险结论强制要求可放行证据。
* **终结安全审查**：CriticAgent 在**所有路径**（含故障关闭、全部跳过）无条件执行，并独立复核剂量范围与配伍禁忌；
  发现问题可触发有界修复循环（`max_loops`）。
* **医师审核闭环**：`draft_for_physician` → 逐味审批 + 签名 → `approved_by_physician`。
* **角色化输出**：患者视图不返回他人病例与证据台账；研究者视图只返回聚合统计；`--debug-state` 才输出完整内部状态。
* **检查点与续跑**：每个节点落盘，`resume` 子命令可从检查点继续。

## 仍未完成

真实授权指南/药典/相互作用数据库、LangGraph 原生 interrupt/resume、医师审批 UI、多轮问诊状态机、
大规模对抗性安全评测与红旗召回率基线仍未实现。**本项目不能对外宣称为临床可用系统。**

## 测试

```bash
python -m unittest discover -s tests    # 77 个用例，无需 pytest
```
