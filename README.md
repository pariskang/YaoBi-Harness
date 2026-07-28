# YaoBi-Harness

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/psknlr/YaoBi-Harness/blob/main/notebooks/Yaobi_Harness_Colab.ipynb)
[![Tests](https://img.shields.io/badge/tests-722%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

> 👆 **点上面的 Colab 徽章即可一键运行**，无需本地安装。合并前想先试这个 PR 的版本：
> [在 Colab 打开 PR 分支](https://colab.research.google.com/github/psknlr/YaoBi-Harness/blob/claude/orthopedic-agent-review-yupwfu/notebooks/Yaobi_Harness_Colab.ipynb)

Yaobi-Harness 是一个 **证据受控的骨科/腰痹多智能体临床决策支持骨架**。V0.1 在 V0.0 安全骨架的基础上补齐了
自主规划层和 LLM 接入，同时保持"认知层可换、控制层不可绕过"的分层：

* **认知层**：LLM 驱动的**规划**、**执行**与**追问**——鉴别诊断、辨证、专家经验综合、方剂组成均由模型
  自主选择工具、自主决定参数、自我纠错并绑定证据（ReAct 工具调用循环）；问诊由模型通过
  `ask_patient` 工具自主组织，多学科会诊由若干独立子体各自运行；临床图片由多模态模型判读。
* **控制层**：能力经纪（角色/风险/技能/熔断/预算）、证据台账与等级、逐断言引用校验、放行状态机、医师审核闭环。

## 分工：规则采集信号，模型做判断

**规则是给模型的参考，不是对模型输出的裁决。** 这一条是被一个真实 bug 逼出来的：
输入「我腰痛1个月，乏力」，系统回了「立即拨打120」——关键词筛查把"乏力"匹配成了
感染/肿瘤的全身症状，而 harness 把这次**关键词命中当成了分诊结论**。
一个月病程的腰痛伴乏力是门诊，不是救护车；**在常规病例上乱响的急症警报，
会让人在真急症时不再相信它**。

现在的分工：

| 环节 | 谁决定 |
| --- | --- |
| 分诊等级 | **模型**。关键词筛查的命中、软信号、模型自己的语义发现，全部作为**素材**交给模型；模型给出等级和理由。两个方向的分歧都逐条记进台账 |
| 问什么、怎么问 | **模型**。它问出的问题**原样送达**——不改写、不用题库替换、不因为漏了某条必答轴就插进别的问题 |
| **什么时候不再问** | **模型**。返回空问题列表即结束。审核者的 `achieved`/`stalled` 只作为 `reviewer_opinion` 进提示词——此前只要审核者说"够了"，模型**根本不会被咨询** |
| **急症轮次要不要提问** | **模型**。此前终态状态会把模型的提问整批丢掉，而「你现在还能自己走吗？」正是分诊 |
| **必答轴算不算已闭合** | **模型（审核者）**。关键词读不出口语否认（"大便一直很正常"）时，审核者可凭病史原话关闭该轴，每次都记录依据 |
| 回复怎么写 | **模型**。包括急症回复。此前急症走固定模板且不许模型参与，所以它分不清疑似马尾和一个月的疲劳，对两者一样喊 |
| 任务图 | **模型**（提案经 Agent 白名单与技能工具表校验） |
| 逐味剂量 | **确定性，永不交给模型**。克数需要签名，不需要模型 |
| 终结安全审查 | **确定性**。它的立场是证伪模型刚才产出的东西，交给模型写就没有意义了 |

未配置模型时，全流程按规则确定性运行——那是回退路径，不是常态路径。

### 只保留两处硬边界，其余都是建议

* **含剂量内容必须经医师逐味审核签名**才能作为处方。这是法定环节，不是表达偏好。
  实现上是**只隐去数值、保留模型的全部文字**——为了不打印"9克"而丢掉整段推理，
  患者付出的代价远大于收益。
* **能力经纪按技能裁剪工具**。模型看不到技能未授权的工具，这是权限，与它的临床判断无关。

其余全部改为建议：必答轴未闭合只影响**能不能进入含剂量环节**，不影响问什么；
规则与模型分诊不一致时**采纳模型的判断并把分歧记下来**；模型的回复只做**追加**
（急症指令、免责声明缺了就补上），不做替换。

模型抽取到清单外的临床信息（吸烟史、工作姿势、既往影像）也不再被删除——
治理字段仍走类型校验，其余落在 `facts["_extra"]`，下游剂量与放行链路不读它，
但模型下一轮看得到、台账里也在。之前它们直接消失，只因为一份在对话发生之前写好的清单。

一轮最多 6 个问题，这是"患者读不完"的限制，不是对问题内容的判断——超出的**顺延到下一轮**
并说明，不静默丢弃。

### 推理模型：思考过程既不展示也不解析

推理型模型（`<think>…</think>`）暴露了两个缺陷，其中一个远比另一个严重。

**展示层面**：开场白里印出了模型对着系统提示词自言自语的一整段。

**解析层面要命得多**：思考过程里几乎必然出现示例 JSON（"我应该输出 `{"age": 99}`"），
而抽取器取的是**第一个**找到的对象——于是一个**凭空捏造的年龄被写进了临床事实**。
编造出来的事实，比难看的输出严重得多。

现在在**每个 provider 都会经过的那一个边界**上分离：`<think>` / `<thinking>` /
`<reasoning>` 等写法，以及 DeepSeek 式的 `reasoning_content` 旁路字段，
都进 `LLMResponse.reasoning`（保留在台账里），`text` 只留真正的回答。
截断在思考中途 → 没有回答，返回空而不是把半截思考当答案。
`extract_json` 与对话层各自再挡一次，因为自定义客户端不走 provider 适配器。

### JSON 修复：让模型真实写出的输出能被用上

模型输出的*形状*被宽容对待，*结构*被严格校验（Agent 白名单、工具表、权限）。
形状这一半现在由 `yaobi_harness/llm/jsonrepair.py` 承担——一个**逐字符扫描器**，
不是一堆正则。这个区别不是风格问题：字符串里的花括号、中文句子里的撇号、URL 里的 `//`，
每一个都会让正则式的"修复"变成**能解析的损坏**，那比解析失败糟得多。

修复的（全部来自真实输出）：代码块围栏与前后散文、尾随逗号与重复逗号、单引号字符串与
裸键、Python 字面量 `True`/`False`/`None`、`NaN`/`Infinity`、`//` 与 `/* */` 注释、
中文输入法的全角与弯引号 “ ” ‘ ’ ，：、字符串里的裸换行制表符，以及**截断**——
被 `max_tokens` 砍断的输出会被补上未闭合的字符串与括号，末尾无法闭合的残片则丢弃后重试。
截断修复价值最高：被砍断的回答通常已经包含了要紧的部分。

**绝不修复**任何需要猜测语义的东西：缺值、无冒号的键、两个对象直接相连——
这些是歧义而不是笔误，替它选一种读法等于把编造的内容写进临床记录。它们照样失败。

修复过程会被记录：`autonomy.<Agent>.json_repairs` 与 `state.notes` 都会写出
用了哪些修复。这条信息有用——`unclosed` 基本等同于"这次回复撞上了 max_tokens"，
是配置问题而不是模型问题，而从一个看起来正常的结果里是看不出来的。计划可以写成 `tasks` /
`plan` / `task_plan` / `steps` / 裸列表，Agent 可以写在 `agent` / `agent_name` / `name`，
依赖可以写成 `depends_on` / `dependencies` / `after`；作答可以带代码块、前后散文、
尾随逗号、单引号字典。理由是内容在后面每一层都会被校验（schema 查字段与类型，
计划校验器查 Agent 白名单与技能工具表，能力经纪查权限），所以为一个尾随逗号拒收
一份提案买不到任何安全，只会把整条模型驱动路径静默降级成规则路径。
含义则绝不猜测：裸字符串只在**精确命中 Agent 目录**时才接受。
回退时 `note` 会说清是哪一种——未配置 / 提案被驳回 / 回复无法解析 / 预算用尽，
而不是只显示 `rule`。

自主性的确切边界见 [docs/AUTONOMY.md](docs/AUTONOMY.md)；
问诊追问见 [docs/INTERVIEW.md](docs/INTERVIEW.md)，
会诊子体与技能库见 [docs/PANEL.md](docs/PANEL.md)，
视觉判读见 [docs/VISION.md](docs/VISION.md)，
病历摘要见 [docs/SUMMARY.md](docs/SUMMARY.md)。

> 严禁把原始 Excel 身份数据提交、打包或直接返回给模型。病例检索只能使用脱敏 ETL 后的结构化字段；含剂量方剂只能以
> `draft_for_physician` 作为医师草案，未逐味审核签名不得发布为最终处方。

## 自主追问：十问歌 × 骨科专科问诊

```bash
python -m yaobi_harness chat --role patient
# 或脚本化：
python -m yaobi_harness chat --role patient \
  --message "腰痛3个月，久坐加重" \
  --message "大小便正常，没有发烧，腿不麻，晚上不痛" \
  --message "63岁，没怀孕，肝肾功能正常，在吃布洛芬和华法林" \
  --message "这两天突然尿不出来，会阴发麻"
```

追问不是念问卷。**模型决定问什么、怎么问、往哪个方向追下去**——它通过 `ask_patient`
工具提问，每个问题必须声明它要闭合哪条**问诊轴**。28 条轴覆盖十问歌全十条与骨科专科的
六条鉴别轴（炎症性/机械性、节段定位、间歇性跛行、髓性症状、FRAX 骨脆性、外伤植入物），
分五个层级：

| 层级 | 未闭合的后果 |
| --- | --- |
| 红旗（5 条） | **阻断任何含剂量放行**，永不可豁免 |
| 核心（4 条） | 阻断含剂量放行 |
| 专科（6 条） | 不阻断，但缺了就分不开病因 |
| 中医四诊（10 条） | **开方前必答** |
| 背景（3 条） | 影响方案，不阻断 |

**模型问出的问题一定会原样问出去。** 这一条曾经是坏的，而且用户看得见：一轮追问会返回
「提问已被拦下：必答轴被模型遗漏，已按题库补回」，患者读到的是一句罐头问句，
而不是模型真正拟的追问。轴表、必答范围、题库现在**全部是给模型的建议**：

* `axis_id` 填错或不填，**照原文问**，只在台账里记一句。
* 必答轴模型没问，**不补题**——下一轮把它作为建议再告诉模型一次，
  未闭合的后果留在充分性裁决里（不得进入含剂量环节），不改写对话。
* 问题里带推断（"我在排除椎管狭窄，所以想知道……"）**照原文问**。医生就是这么问的。
* 问题里出现剂量，**只隐去数值**，问题照问。

审核者仍然独立：模型说"问够了"只是提案，必答轴未闭合判 `blocked`（不得进入含剂量环节），
连续三轮问不出新信息判 `stalled`（带缺口继续，不把病人困在无尽问卷里）。

**否定回答也是回答**：说"大小便正常、没有发烧"会闭合对应的红旗轴，
否则系统会反复追问同一件事，最后把一个完全配合的病人判成"病史不足"。
分类复用 `safety/red_flags.py` 既有的从句级否定逻辑——顺带修掉了筛查层的两个真实缺陷
（口语否定"不会痛醒"被读成阳性；被否认的佐证仍在把软信号促成硬信号）。

```bash
python -m yaobi_harness interview --tier RED_FLAG      # 看全部红旗轴
python -m yaobi_harness interview --complaint "68岁女性腰痛3月，走远了要停"
```

完整说明见 [docs/INTERVIEW.md](docs/INTERVIEW.md)。

## 智能体自己调图，和最后的病历摘要

**需要看图时，模型主动要。** 它调用 `request_image`（或在 `ask_patient` 里带
`image_requests`），界面随即打开上传模块并**预选好类型**，同时把模型给的理由展示出来
（"舌象能把气滞血瘀和寒湿分开，直接影响用药方向"）。什么时候值得要写在技能说明里：
舌象决定辨证、患肢外观提示血管或骨筋膜室问题、报告单翻拍把"患者转述"换成原始记录。
什么时候不要：为了完整而收集影像——那是负担不是帮助。

两条边界，都不是对模型判断的限制：

* **去标识化仍由人确认。** 模型不能替对方勾选「已去标识化」，也不能声称已确认——
  它没看过那个文件，由它断言等于让这项声明失去意义。
* **没配置视觉模型时请求会被拦下并记录**。上传了也读不了的图，只是白费对方的力气。
  没单独配视觉模型时，控制台会借用会话自己的对话模型（多模态对话模型就是能用的视觉模型），
  并在徽章上标明；无论如何，**运行结束时若有图而无判读结果，必须说清是哪一种情况**——
  一张片子被接收、存下、再也没被看过，而答复读起来像什么都没附，是唯一不可接受的结局。

**得出结论后自动生成结构化病历摘要**，按门诊病历的节次排列：主诉 / 现病史 / 既往史与用药 /
中医四诊 / 查体与量表 / 辅助检查与影像 / 西医诊断（鉴别）/ 中医诊断与证型 / 风险评估与红旗 /
用药安全 / 治疗计划 / 处方草案 / 医嘱 / 随访 / 不确定性与局限，附引用来源与证据台账。

```bash
python -m yaobi_harness run --complaint "..." --facts '{...}' --summary   # 直接打印病历文本
python -m yaobi_harness chat --role patient        # 对话里输入 /note 查看
```

控制台里，**摘要直接出现在对话里**，带复制与下载——成果交付在做成果的地方。
此前是在气泡里写一句「请到「单次运行」页的「病历摘要」标签查看与复制」：为了拿到这次问诊
唯一的产出，要切页面、在八个标签里找一个，还会落在一张与刚才的对话毫无关系的病例表单上。

三条设计要点：

* **不新增任何临床事实。** 每一节都来自 `state.outputs`、`state.facts` 与证据台账。
  没有来源的节印成「未采集」而不是省略——一份悄悄漏掉既往史的病历，
  读起来像"没有相关既往史"，那是完全不同的临床断言。
* **有剂量当且仅当确定性链路产出了剂量。** 处方一节原样复现草案与逐味签名状态，
  其余各节不含克数；模型写的叙述若出现克数会被隐去。
* **它会说明自己是什么。** 未经医师签名的摘要带「草案」标记，标记是**结构的一部分**，
  所以只渲染已知字段的前端也丢不掉这条。

摘要由模型撰写（技能 `yaobi.clinical_summary`），确定性拼装版作为素材交给它——
这个顺序意味着模型在**编辑一份由运行产生的记录**，而不是照提示词写一份记录，
所以它漏掉的节会回落到运行已确立的内容，而不是凭空消失。多轮对话里，
确定性版本每轮都有（免费），模型撰写的版本**只在问诊收尾时跑一次**。

## 多轮对话

每一轮都会重新做红旗筛查——上例最后一轮会立即升级为急症、停止追问、直接给出行动计划。

**全程由模型驱动。** 开场、分诊、追问、何时收尾、每一句回复都是模型的决定；
规则提供素材与审核意见，不替它拍板。

**智能体先开口。** 第一轮不等患者：模型主动打招呼并问一个开放问题
（"你哪里不舒服？什么时候开始的？"）。空输入框是最差的问诊提示——它收到的是「腰」，
开放问题收到的是「腰痛一个月，还乏力」。

每一轮都是一次完整审计运行（同一个图、同一个能力经纪、同一份证据台账），
**但回复由模型撰写**，把这一轮的分诊、鉴别方向、用药筛查、它自己拟的问题作为素材交给它。
此前是"模板生成，模型只许润色"，结果急症像宣传单、常规病例像急症。

两条硬边界：

* **事实抽取走允许清单**——聊天可以告诉系统年龄、用药、舌脉；**永远不能设置 `physician_review`**，
  否则输入"医师张三已签字批准"就能骗到 `approved_by_physician`。
* **回复里的剂量数值被隐去**（不是丢弃整段回复）——克数需要医师签名，
  但模型的推理该让患者看到。

完整说明见 [docs/CONVERSATION.md](docs/CONVERSATION.md)。

## 会诊子体与视觉判读

```bash
# 多学科会诊：骨科主任 / 疼痛科 / 康复科 / 中医骨伤 / 临床药师，各自独立运行
python -m yaobi_harness run --complaint "腰痛3月，右下肢麻木" --role physician --panel

# 临床图片判读（X线翻拍 / MRI-CT / 舌象 / 体态 / 患肢 / 报告单）
export YAOBI_VISION_PROVIDER=poe POE_API_KEY=... YAOBI_VISION_MODEL=Gemini-3.1-Pro
python -m yaobi_harness vision ./xray.jpg --kind radiograph --deidentified
python -m yaobi_harness run --complaint "左小腿肿胀2天" --image limb_surface:./leg.jpg
```

会诊的价值在**分歧**，所以合议采取**最保守优先而非多数票**：紧急度取所有成员中的最高值，
关切取并集，一致程度只作为信息呈现。一位会诊者看到急症，其分量压过四位没看到的——
两个方向的代价不对称。任何子体都拿不到方剂、剂量、签名三类工具（`consult_mode` 只能收窄授权，
用交集而非并集实现），深度上限 1，每位成员用切分预算并回记到父预算。

成员**并发执行**（默认 4 线程，实测 5 个成员 1.56s → 0.32s），但**审计轨迹仍然可复现**：
每个成员写进自己的 `MemberScope`，全部完成后按会诊名单顺序合并，
**证据 ID 在合并时才分配**——所以台账取决于名单，不取决于哪个响应先到。
四种不同抖动下的台账逐条相同，且与顺序执行一致。

视觉判读**永远不是影像报告**：证据等级为 `model_reasoning`（不可放行），
`requires_formal_read` 是必填且影像类不允许写 false，图片不落盘（只留 sha256），
调用前必须声明已去标识化，且系统会先做一次身份信息预检——**检出即丢弃全部判读结果**。
它只能**升级**风险（患肢发紫/张力高 → 血管或骨筋膜室信号），不能撤销规则已判定的红旗。

详见 [docs/PANEL.md](docs/PANEL.md) 与 [docs/VISION.md](docs/VISION.md)。

## 重放日志：过去的决策可以被离线复核

检查点记录的是**状态曾是什么**，但审计要问的是"在恰好这些证据之下，为什么放行了那条建议"。
回答它必须用当时的工具返回和当时的模型输出重跑，而不是用今天的端点重跑。

```bash
# 录制：每一次工具执行与模型补全都按内容地址追加到 JSONL
python -m yaobi_harness run --complaint "…" --journal ./audit/case-001.jsonl

python -m yaobi_harness journal ./audit/case-001.jsonl --entries   # 看里面有什么

# 离线重放：不需要网络，不需要配置模型
python -m yaobi_harness run --complaint "…" --replay ./audit/case-001.jsonl
```

三条设计要点：

* **日志不是授权。** 授权在重放时重新推导——以医师身份录的日志，
  在以患者身份重放时拿不到方剂结果，因为经纪在触及记录之前就拒绝了。
  被篡改的日志最多让重放**偏离**，不可能把草案提升为已签名处方。
* **偏离必须响亮。** Agent 用宽泛 `except Exception` 包住模型调用以便回退，
  这会把偏离异常吞成一句"已回退规则"。所以偏离被**锁存**，运行结束时检查并**故障关闭**，
  退出码 3——一个把退出码当作"重放确认了原决策"的脚本不能在偏离时被告知"是"。
* **耗尽 ≠ 偏离。** 同序号不同调用是硬失败；日志用完还有新调用只是告警
  "本次结果不是纯离线重放"。

详见 [docs/REPLAY.md](docs/REPLAY.md)。

## 可视化控制台

```bash
python -m yaobi_harness ui --port 8000 --knowledge-store ./knowledge.db
```

控制台是一个**智能体运行检查器**，不是聊天界面：放行状态是视觉主角，
`计划 → 执行 → 证据 → 裁决` 全部可见，「交付内容」与「操作者审计」在页面上明确分离——
切换交付对象（患者/医师/研究者）能直接看到输出裁剪的差异。零依赖、零 CDN、单文件页面、
明暗双主题，可在离线院内网络运行。详见 [docs/CONSOLE.md](docs/CONSOLE.md)。

四个视图：**对话问诊**（多轮）、**单次运行**（完整病例检查器）、**用药速查**（只做相互作用筛查）、
**知识库**（来源目录、许可状态、技能表、专家语料、规则包全文）。后端是普通 JSON API，可被其他前端复用。

单次运行页把本文档里的每一项能力都做成了控件：**召集多学科会诊**勾选后出现
**会诊并发线程**（1–8，并发只改调用时序不改证据台账）；**录制可复核日志**勾选后
多出一个**离线复核**标签页——用录制的日志重新推导这次决策，先给结论
（放行状态、风险模式、规划来源、证据条数逐项对照），并且可以**改写主诉再复核**，
此时日志按请求内容寻址、病历一变就对不上，运行故障关闭并列出偏离的请求哈希。
「规划与执行」页现在还会说明**为什么**用了确定性计划，而不只是显示「规则」。
控制台的日志只存在内存里、不落盘——它和聊天记录一样是临床内容。

**对话轮次在后台跑，页面轮询。** 一轮问诊是十余次串行模型调用；推理型模型下这是几分钟，
把 HTTP 请求挂那么久正是「出错了：Failed to fetch」的成因——那是**浏览器**对连接中断的
用词，不是本服务发出的错误，Colab 的 iframe 代理尤其不会让请求开那么久。
现在 `POST /api/chat/start` 立刻返回任务号，页面每 1.5 秒轮询一次
`/api/chat/poll`。所有请求都带超时、一次静默重试，
以及说明问题的错误文案，而不是把浏览器的原话直接抛给用户。

**执行过程是流式可见的。** 轮询带回的不只是计数，还有从上次游标之后的每一步：
哪个子体开始了、调了什么工具（含**被技能策略拒掉的**调用）、工具回了什么、
模型这次想了什么。页面实时渲染，答复到达后折叠成「本轮执行过程」留在气泡里。
是**按步**流式而不是按 token：逐字流式只会让第十三次调用一个字一个字地出现，
而前十二次——真正的等待来源——仍然一片空白。工具参数**只在取值属于代码掌握的封闭词表时
才显示原值**（轴 id、图片类型、已知合并症），其余只显示参数名和形状：这个流会渲染在
可能被人从旁看到的标签页里，而模型驱动的工具循环自己挑参数，按参数名做白名单只是猜。

**一轮问诊从十三次调用降到七次。** 六次是鉴别/辨证/病例检索，而那一轮的产出
可能只是一句「您疼多久了？」。要不要现在就展开这套推理，由问诊充分性审核者回答
（`workup_now`，两个方向都算数）；安全筛查从不推迟，单轮 `run` 也从不推迟——
没有下一轮可推迟到时，推迟就等于静默丢弃。剩下的调用里，互不依赖的子体并行执行，
按计划顺序合并，审计轨迹与串行跑出来的逐字节相同。

**影像走单独的上传通道。** 选中文件即以原始字节 `POST /api/image/upload`，
换回一个 handle，之后每轮只带这几十个字节。此前页面把整张片子转成 base64
塞进每一条聊天消息，附件看起来传成功了（其实什么都还没离开浏览器），
下一次提问才炸出「请求体过大」。

需要分享给同事评审时可映射成公开链接（会**强制**生成访问令牌并拼进 URL）：

```bash
export NGROK_AUTHTOKEN=...        # https://dashboard.ngrok.com/get-started/your-authtoken
pip install pyngrok
python -m yaobi_harness ui --public --knowledge-store ./knowledge.db
```

> ⚠️ 公网链接是**演示/评审链接，不是临床部署**：令牌只是演示级门禁，没有逐用户身份、没有访问审计、
> 没有院内网络边界。**不要在公开实例里输入任何真实患者可识别信息。** 默认仍只监听 `127.0.0.1`。

**Colab**：点顶部徽章直接打开 `notebooks/Yaobi_Harness_Colab.ipynb`，十六节完整走查——
安装自检 → 确定性运行 → 骨科规则包 → 实时构建知识库 → 授权药典如何改变放行 →
xlsx 变技能 → 模型自主执行 → 多轮对话 → **自主追问（十问歌 × 专科）** →
**会诊子体** → **视觉判读** → **技能库** → **并发会诊** → **重放日志** →
接入 LLM（含模型输出形状容忍度的逐项验证）→ 内嵌控制台（含离线复核与 ngrok 公开链接）。

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
| `minimax` | `MINIMAX_API_KEY` | `MINIMAX_MODEL`（默认 `MiniMax-M3`）、`MINIMAX_REGION`（`china`/`global`）、`MINIMAX_BASE_URL`、`MINIMAX_GROUP_ID` |
| `litellm` | `LITELLM_MODEL` | `LITELLM_API_KEY`、`LITELLM_BASE_URL`（默认 `http://localhost:4000/v1`） |

> **MiniMax 的两个区域地址不能互换**：国内 `https://api.minimaxi.com/v1`，
> 海外 `https://api.minimax.io/v1`。默认走国内，用 `MINIMAX_REGION=global` 切换。
> 已废弃的 `api.minimax.chat` 会**直接报错并给出正确地址**，而不是超时。

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

## 技能：既是授权，也是规程

技能在本系统里同时是两样东西：能力经纪强制执行的**权限授权**，和模型执行时读取的**规程说明**。
这两样东西的载体需求正好相反，所以有两种写法：

* **`manifest.yaml`** —— 工具授权、输出契约、角色限制集中在一份文件里，合规审查者一眼看完。
* **`SKILL.md`** —— 一个目录加一份 markdown：frontmatter 放策略，正文放流程。
  一份完整的骨科问诊协议有几百行，塞进 YAML 标量就不可读，而这恰恰最需要临床评审。
  布局与 Grok Build、Claude Code 相同，同一份文件可在几个 harness 之间移植。

随包发布四份专业技能：`yaobi.interview`（十问歌 + 骨科专科全流程）、
`yaobi.vision_read`（七类图片各自的判读边界）、`yaobi.consult_panel`（五个专科视角 + 14 组用药风险核对表）、
`yaobi.osteoporosis_risk`（FRAX 要素、椎体骨折线索、治疗顺序硬约束、跌倒风险）。

优先级从高到低：`$YAOBI_SKILL_PATH` / `--skill-dir` → `./.yaobi/skills/` → 随包库 → `manifest.yaml`。
同 `skill_id` 高优先级**替换**低优先级，所以医院能钉住自己的问诊协议而不用改包。

```bash
python -m yaobi_harness skill list                    # 全部技能与来源（manifest / SKILL.md）
python -m yaobi_harness skill show yaobi.interview    # 完整流程文本
python -m yaobi_harness run --skill-dir ./my-skills --complaint "..."
```

### 把专家 xlsx 变成技能

`skill build-expert` 把语料挖掘成技能：

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
* **问诊范围是建议，问诊内容是模型的**：模型跳过的必答轴不会被题库补回，只会在下一轮
  再被建议一次；`blocked` 裁决（红旗轴未获答复）仍然不可豁免，但它拦的是**含剂量环节**，
  不是对话。
* **分诊分歧双向留痕**：规则关键词筛查与模型判断不一致时采纳模型的判断，
  并把"规则报了什么、模型为什么不同意"逐条记进台账——这是本设计的代价，
  它不能被无声地付出。
* **子体永不出处方**：`consult_mode` 用交集收窄授权，方剂/剂量/签名三类工具对任何会诊子体不可达；
  深度上限 1；预算切分并回记父预算；合议取最高紧急度而非多数票。
* **急症计划由模型撰写**：紧急程度要与本例相称；模型漏写 `immediate_action` 时
  由系统**追加**而非替换，所以指令一定到达患者，措辞仍然是模型的。
* **视觉不做诊断**：判读等级 `model_reasoning`（不可放行），影像类不允许声明
  `requires_formal_read=false`，图片不落盘，PHI 预检命中即丢弃全部结果。
* **并发下台账仍可复现**：成员写各自的 `MemberScope`，按名单顺序合并分配证据 ID；
  `Budget` 与 `ToolHealth` 都是读-改-写，已加锁——否则"硬上限"和"两次熔断"在并发下都名不副实。
* **重放要么复现，要么明说**：调用按内容地址记录，同序号不同调用即判偏离并故障关闭；
  偏离被锁存，不依赖异常穿透那些合法的 catch。日志耗尽后实跑的尾部不计入"已复现"。
* **格式失误给一次重提，安全失误不给**：作答不合 schema 时补发一轮"只输出 JSON"并附上
  schema，因为不该为少一个代码块丢掉模型已经做完的取证；但**输出含克数不重提**——
  重提只会换个说法再泄一次。重提次数记录在自主执行台账里。

## 仍未完成

**模型误判分诊的风险现在落在模型身上。** 规则改为建议之后，一个把真实马尾综合征
判成 routine 的模型会决定结果——分歧会被记录并在控制台和 CLI 里可见，但不再被拦下。
这是"规则只作参考"这一要求的直接代价，部署前必须结合模型能力评估。

LangGraph 原生 interrupt/resume、医师审批 UI、中文指南的结构化推荐抽取、
大规模对抗性安全评测与红旗召回率基线仍未实现。模型**不能主动发起对话轮次**（它会答会问，
但不会自己开口）；persona 的 I/O 契约是声明性的、未被强制校验；
重放日志能证明"重放与记录一致"，不能证明"记录未被修改"（需存储层签名）；
并发只覆盖会诊，图内任务执行仍是串行。内置骨科规则包不能替代完整的相互作用数据库，
且须经本机构药师/医师复核后启用。**本项目不能对外宣称为临床可用系统。**

## 测试

```bash
python -m unittest discover -s tests    # 722 个用例，无需 pytest 与网络
```
