# YaoBi-Harness

Yaobi-Harness 是一个 **证据受控的骨科/腰痹多智能体临床决策支持骨架**。V0.1 在 V0.0 安全骨架的基础上补齐了
自主规划层和 LLM 接入，同时保持"认知层可换、控制层不可绕过"的分层：

* **认知层**：LLM 驱动的**规划**与**执行**——鉴别诊断、辨证、专家经验综合、方剂组成均由模型
  自主选择工具、自主决定参数、自我纠错并绑定证据（ReAct 工具调用循环）。
* **控制层**：能力经纪（角色/风险/技能/熔断/预算）、证据台账与等级、逐断言引用校验、放行状态机、医师审核闭环。

LLM 在本系统中是**只能加安全、不能减安全**的执行者：它可以规划、选工具、下结论，但看不到技能未授权的工具、
不能发明 Agent、不能清除规则层命中的风险信号、**永远不能生成剂量**。任何一步越界（越权/输出不合 schema/
出现克数/预算或步数耗尽）都会**整体回退**到确定性逻辑，而不是带病放行。未配置模型时，全流程确定性运行。

自主性的确切边界见 [docs/AUTONOMY.md](docs/AUTONOMY.md)。

> 严禁把原始 Excel 身份数据提交、打包或直接返回给模型。病例检索只能使用脱敏 ETL 后的结构化字段；含剂量方剂只能以
> `draft_for_physician` 作为医师草案，未逐味审核签名不得发布为最终处方。

## 可视化控制台

```bash
python -m yaobi_harness ui --port 8000 --knowledge-store ./knowledge.db
```

控制台是一个**智能体运行检查器**，不是聊天界面：放行状态是视觉主角，
`计划 → 执行 → 证据 → 裁决` 全部可见，「交付内容」与「操作者审计」在页面上明确分离——
切换交付对象（患者/医师/研究者）能直接看到输出裁剪的差异。零依赖、零 CDN、单文件页面、
明暗双主题，可在离线院内网络运行。详见 [docs/CONSOLE.md](docs/CONSOLE.md)。

三个视图：**诊疗运行**（完整病例）、**用药速查**（只做相互作用筛查）、
**知识库**（来源目录、许可状态、规则包全文）。后端是普通 JSON API，可被其他前端复用。

需要分享给同事评审时可映射成公开链接（会**强制**生成访问令牌并拼进 URL）：

```bash
export NGROK_AUTHTOKEN=...        # https://dashboard.ngrok.com/get-started/your-authtoken
pip install pyngrok
python -m yaobi_harness ui --public --knowledge-store ./knowledge.db
```

> ⚠️ 公网链接是**演示/评审链接，不是临床部署**：令牌只是演示级门禁，没有逐用户身份、没有访问审计、
> 没有院内网络边界。**不要在公开实例里输入任何真实患者可识别信息。** 默认仍只监听 `127.0.0.1`。

**Colab**：`notebooks/Yaobi_Harness_Colab.ipynb` 是完整走查（安装自检 → 确定性运行 → 骨科规则包 →
实时构建知识库 → 授权药典如何改变放行 → 接入 LLM → 内嵌控制台），可直接在 Colab 打开运行。

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

## 把专家 xlsx 变成技能

专家病例不该只用于"检索相似病例"。`skill build-expert` 把语料挖掘成**技能**——在本系统里技能同时是
能力经纪强制执行的**权限授权**，和模型执行时读取的**规程说明**：

```bash
python -m yaobi_harness skill build-expert --xlsx ./authorized.xlsx --merge
python -m yaobi_harness skill show yaobi.expert_case_reasoning
python -m yaobi_harness run --skill-manifest yaobi_harness/skills/manifest.yaml --complaint "..."
```

按证型挖掘核心药（出现率 ≥60%）、随证加减、常用治法、合并西药、常做检查、合并症、反例例数与复诊轨迹。
生成的技能直接**替换**内置的 `yaobi.expert_case_reasoning`，因此重新生成即升级已有 Agent。

* 只输出聚合统计，绝不含个体自由文本；低于 `min_support` 的取值被抑制；生成前跑 PHI 自检。
* **剂量不进入技能说明**——推理 Agent 禁止输出克数，剂量只经 `herb_dose_distribution` 走确定性链路。

## 接入授权知识库

指南、药典与相互作用数据不随仓库分发——仓库只提供连接器和**在写入时强制执行的许可模型**。
完整来源目录、文件格式与摄取配方见 [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md)。

```bash
export YAOBI_DEPLOYMENT_MODE=research_noncommercial     # 或 commercial
python -m yaobi_harness knowledge sources               # 各来源当前是否可用及原因
python -m yaobi_harness knowledge build --store ./knowledge.db --cache-dir ./.kcache
python -m yaobi_harness knowledge check-interactions --medications 布洛芬 华法林
python -m yaobi_harness run --role physician --knowledge-store ./knowledge.db --complaint "..."
```

* **开箱可用（CC0/公有领域）**：openFDA 说明书、DailyMed SPL、RxNorm/RxClass，以及内置的骨科相互作用规则包
  （18 条规则 / 30 个药物类别）和十八反十九畏规则包。
* **非商业**：WHO 指南与国际药典、DDInter 2.0 —— 在 `commercial` 模式下写入直接被拒绝。
* **只读**：AAOS、中华医学会、NMPA 文件、香港衞生署 —— 只存标题/版本/链接/摘录要点，全文永不入库。
* **须授权**：NICE、《中国药典》2025、NMPA 说明书、USP–NF、EP、DrugBank、BNF/Stockley's —— 未登记
  `YAOBI_LICENSE_ATTESTATIONS` 前完全禁用。

接入后的行为变化：授权指南命中即为 `guideline_or_standard` 级证据（不再是 stub）；授权药典范围优先于本地配置表
并逐味比对拟用剂量；医师与研究者视图输出 `citations`，逐条给出来源、许可、版本、发布日期与检索时间。

## 安全能力

* **红旗筛查**：子句级否定/家族史/假设语境判定，覆盖马尾、心肺、DVT/PE、感染肿瘤、化脓性关节炎/骨髓炎、骨折、
  进行性神经缺损、脊髓型颈椎病、骨筋膜室综合征；硬信号立即升级，弱信号走"需线下检查"而非丢弃；不确定时向上升级。
* **剂量安全**：分层专家剂量 → 最小样本量/离散度/异常值 → **拟用剂量与授权药典范围逐味比对** → 特殊人群 →
  用药过敏确认 → 十八反/十九畏/妊娠禁忌；任一不通过即降级为 `treatment_advice_only`，不产出克数。
* **用药安全**：`MedicationSafetyAgent` 对患者现有西药做骨科相互作用筛查（NSAIDs+抗凝、三重打击、阿片+镇静、
  曲马多+SSRI、双膦酸盐+钙剂、秋水仙碱+CYP3A4 抑制剂、围术期抗凝与椎管内麻醉等）；`contraindicated`/`major`
  会阻断并把放行状态抬到 `needs_examination`，患者视图给出通俗的处理建议。
* **能力经纪**：角色/风险模式/技能清单/熔断/预算按序检查，**被拒绝的调用不扣预算**；技能缺失即拒绝（fail-closed）。
* **证据台账**：等级由**工具自身**声明，占位数据源记为 `stub_not_for_clinical_use`，不可被提升为指南级；
  逐断言引用校验（CitationGuard）对高风险结论强制要求可放行证据。
* **终结安全审查**：CriticAgent 在**所有路径**（含故障关闭、全部跳过）无条件执行，并独立复核剂量范围与配伍禁忌；
  发现问题可触发有界修复循环（`max_loops`）。
* **医师审核闭环**：`draft_for_physician` → 逐味审批 + 签名 → `approved_by_physician`。
* **角色化输出**：患者视图不返回他人病例与证据台账；研究者视图只返回聚合统计；`--debug-state` 才输出完整内部状态。
* **检查点与续跑**：每个节点落盘，`resume` 子命令可从检查点继续。

## 仍未完成

LangGraph 原生 interrupt/resume、医师审批 UI、**模型主动发起的多轮问诊**、中文指南的结构化推荐抽取、
大规模对抗性安全评测与红旗召回率基线仍未实现。内置骨科规则包不能替代完整的相互作用数据库，
且须经本机构药师/医师复核后启用。**本项目不能对外宣称为临床可用系统。**

## 测试

```bash
python -m unittest discover -s tests    # 200 个用例，无需 pytest 与网络
```
