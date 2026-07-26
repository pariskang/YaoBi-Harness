# 操作者控制台与 Colab

## 设计理念

控制台不是聊天界面，而是一个**智能体运行检查器**。三条原则：

1. **放行状态是视觉主角。** `urgent_action_plan` / `blocked` / `draft_for_physician` 等以整块状态条呈现，
   并说明"为什么是这个状态"。颜色只用于临床语义，不用于装饰——同一个颜色永远表示同一件事。
2. **推理过程是一等公民。** `计划 → 执行 → 证据 → 裁决` 全部可见：任务图带每个 Agent 的状态与工具调用，
   证据台账标注等级与是否可放行，安全审查列出已执行的检查与裁决。
3. **交付与审计分离。** 「交付内容」标签页是该角色**实际会收到**的东西（由 `render()` 产出）；
   其余标签页是操作者审计视图，页面上明确标注"不对该角色展示"。切换角色能直接看到输出裁剪的差异。

零依赖：标准库 `http.server` + 单文件页面，无框架、无 CDN、无构建步骤。可在离线院内网络运行。

## 启动

```bash
export YAOBI_DEID_KEY="$(openssl rand -hex 32)"
python -m yaobi_harness ui --port 8000 --knowledge-store ./knowledge.db
```

| 参数 | 说明 |
| --- | --- |
| `--host` | 默认 `127.0.0.1`。**控制台没有身份认证**，除非前置了自有的认证代理，否则不要暴露到共享网络 |
| `--port` | 默认 8000 |
| `--knowledge-store` | 授权知识库路径（见 [KNOWLEDGE.md](KNOWLEDGE.md)） |
| `--xlsx` | 本地授权专家病例 Excel |
| `--llm-provider` / `--llm-model` | 覆盖 `YAOBI_LLM_PROVIDER`；密钥仍只从环境变量读取 |
| `--checkpoint-dir` | 每个节点落盘，可用 `resume` 子命令续跑 |
| `--skill-manifest` | 指定技能清单（例如合并了生成的专家技能的那份） |
| `--open` | 启动后打开浏览器 |
| `--public` | 经 ngrok 映射为公开链接，**强制启用访问令牌**（仅演示/评审） |
| `--access-token` | 固定访问令牌；`--public` 时会自动生成一个。启动时打印的链接**已包含 `?t=`**，请用那个链接打开 |
| `--ngrok-authtoken` / `--ngrok-region` | 覆盖 `NGROK_AUTHTOKEN` / 选择区域 |
| `--no-vision` | 即使配置了视觉模型也禁用 |
| `--skill-dir` | 追加 `SKILL.md` 根目录（最高优先级），可重复 |
| `--panel-concurrency` | 会诊并发线程的默认值（页面可逐次覆盖；1 为顺序执行） |

## 四个视图

**诊疗运行** — 左侧填写病例，右侧看结果。
左侧支持：交付对象（患者/医师/研究者）、主诉、五个示例病例、当前用药（中英逗号分隔）、
患者状态标签（决定条件门控规则是否触发）、结构化事实 JSON，以及四个开关：

| 开关 | 作用 |
| --- | --- |
| 允许生成含剂量草案 | 仅医师角色生效，仍需逐味审核签名 |
| 启用 LLM 规划与审查 | 关掉即可与确定性路径对照 |
| 召集多学科会诊 | 勾选后展开**会诊并发线程**（1–8）。并发只改调用时序，不改证据台账 |
| 录制可复核日志 | 记录每一次工具与模型调用，运行后在「离线复核」页重放 |

右侧标签页：

| 标签 | 内容 |
| --- | --- |
| 交付内容 | 该角色实际收到的答复，带角色对应的免责声明 |
| 规划与执行 | **自主执行**面板（模型选了哪个工具、传了什么参数、格式重提了几次、绑定了哪条证据）、任务图、每个 Agent 的状态，以及**规划来源的诊断** |
| 问诊与会诊 | 问诊轴覆盖率、实际问出的问题、充分性裁决；会诊成员意见与合议 |
| 用药安全 | 相互作用发现，含严重度、机制、处理措施、命中药物与类别映射、触发条件 |
| 离线复核 | 仅在录制后出现：用日志重新推导这次决策并逐项对照；可改写主诉再复核 |
| 证据台账 | 每条证据的等级与可放行性；结论↔证据的绑定；外部来源的许可/版本/发布日期/检索时间 |
| 安全审查 | 终结节点裁决、告警、修复请求、缺证据的结论、已执行的检查清单 |
| 原始 JSON | 完整响应，便于排查 |

### 规划来源要能被诊断

「规划与执行」页此前在回退时只显示「规则」，而这三件完全不同的事都会显示成「规则」：
没有配置模型、提案被规则层驳回、模型有回复但解析不出任务。操作者无法区分，
也就无法处理。现在 `audit.plan.note` 随响应返回，页面把每一种翻译成一句人话，
并把原始 note 一并显示——见 [AUTONOMY.md](AUTONOMY.md#格式失误给一次重提安全失误不给)。

### 离线复核

结论放在最前面，因为一次重放的价值在**对照**，而不在第二个答案：
放行状态、风险模式、规划来源、任务状态、按台账顺序的证据逐项比对。
时间戳与 run_id 被刻意排除——它们必然不同，比对它们会把每次重放都报成偏离。
「已复现」还额外要求日志没有耗尽：耗尽后实跑的尾部是重新推导，不是重放。

**改写主诉再复核**是这一页真正的用途："如果病史读起来不一样，这份日志还能支撑那个结论吗？"
日志按请求内容寻址，所以答案是响亮的失败，并列出偏离的请求哈希。

日志**只存在内存里、从不落盘**，保留最近 20 次；理由和控制台不持久化聊天记录相同——
都是临床内容。要落盘用 CLI 的 `--journal`。

**用药速查** — 不跑完整病例，只做相互作用筛查。适合门诊快速核对。

**知识库** — 知识库统计、来源目录与许可状态（哪些启用、哪些被禁用及原因）、内置规则包全文。

## 访问令牌怎么流转

配置了令牌之后，页面必须**自己把令牌带上**——这一点曾经是坏的，值得写清楚。

服务端接受四种通道：`Authorization: Bearer`、`X-Yaobi-Token` 头、`?t=` 查询串、
`yaobi_token` cookie。但页面此前一个都不发：只有最初那次带 `?t=` 的 HTML 请求能过，
随后每个 XHR 都 401。于是 Colab 内嵌与 ngrok 公开链接下，控制台会**把界面渲染出来**，
然后在第一个动作上失败并提示"缺少或错误的访问令牌"。

现在页面按 URL → `sessionStorage` → cookie 的顺序取令牌，并在每个请求上带
`X-Yaobi-Token`。三条通道都留着，是因为在第三方 iframe 里 storage 与 cookie
都可能被浏览器拦掉，而 URL 一定在——也正因如此**不从地址栏抹掉 `?t=`**：
storage 被拦时，刷新只能靠它。

已验证：跨域 iframe 下正常、阻止第三方 cookie 时正常、cookie 与 storage 全部阻止时仍正常；
无令牌与错误令牌一律 401。

## API

控制台的后端是普通 JSON API，可以被其他前端复用。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 存活检查 |
| GET | `/api/bootstrap` | LLM/知识库/许可策略/规则摘要/示例病例 |
| GET | `/api/rules` | 完整规则包与药物类别表 |
| POST | `/api/run` | `{complaint, role, facts, allow_prescription, use_llm, enable_panel, panel_concurrency, record_journal, images}` → `{delivered, audit, meta, journal?}` |
| POST | `/api/replay` | `{run_id, complaint?, facts?}` → `{fidelity, journal, delivered, audit, meta}` |
| POST | `/api/interactions` | `{medications, conditions}` → 相互作用筛查结果 |
| POST | `/api/chat` | `{message, role, session_id?}` → `{session_id, reply, audit, meta}` |
| POST | `/api/chat/reset` | `{session_id}` → 清空该会话 |

`/api/run` 的返回结构固定为三段：`delivered`（角色化答复）、`audit`（推理记录，仅操作者）、
`meta`（放行状态、规划来源、预算、会诊并发、知识库与许可模式）；
`record_journal` 时多一段 `journal`（`run_id`、条目数、重放提示）。

`/api/replay` 的 `fidelity` 先给结论：`reproduced`、`against`（`recording` / `modified`）、
`differences`、`before` / `after` 指纹、`divergences`、`live_after_exhaustion`。
`run_id` 未知时返回 400 而不是 500——那是请求错误，不是服务故障。

```bash
curl -sS -X POST http://127.0.0.1:8000/api/run \
  -H 'Content-Type: application/json' \
  -d '{"complaint":"腰痛3月，久坐加重","role":"physician",
       "facts":{"medications":["布洛芬","华法林"],"conditions":["elderly"]}}'
```

## 限制

* **无身份认证、无多用户隔离。** 这是本地操作者工具。
* 会话创建与淘汰在锁内完成，上限 50 个（`MAX_SESSIONS`）；accept 队列深度 128——
  默认的 5 会让一次页面加载的并发请求被重置，而连接重置看起来像控制台崩了，
  比"你的请求在排队"是坏得多的诊断。
* 对话会话只存在内存中，进程退出即丢失；持久化涉及临床记录留存，须另行设计。
* 一次只跑一个病例：服务端用锁把运行串行化。会诊在**运行内部**并发（默认 4 线程），
  所以串行化的是病例，不是成员。
* 浏览器端不持久化任何病例数据；刷新即清空。

## Colab

README 顶部有 Colab 徽章，点开即可运行。`notebooks/Yaobi_Harness_Colab.ipynb` 是十节完整走查：
安装自检 → 确定性运行 → 骨科规则包 → 实时构建知识库 → 授权药典如何改变放行 →
xlsx 变技能 → 模型自主执行 → 多轮对话 → 接入 LLM → 内嵌控制台（含 ngrok 公开链接）。

内嵌方式：

```python
import threading, time
from yaobi_harness.ui.server import ConsoleService, create_server

service = ConsoleService(knowledge_store_path="/content/knowledge.db")
httpd = create_server(service, "127.0.0.1", 8000)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
time.sleep(1)

from google.colab import output
output.serve_kernel_port_as_iframe(8000, path=f"/?t={service.access_token}", height=1100)
```

公开链接（会强制要求访问令牌）：

```python
from yaobi_harness.ui.tunnel import banner, new_token, open_ngrok

tunnel = open_ngrok(8000, token=service.access_token, authtoken="<NGROK_AUTHTOKEN>")
print(banner(tunnel, local_url="http://127.0.0.1:8000/"))
```

本地 Jupyter 用 `IPython.display.IFrame` 指向 `http://127.0.0.1:8000/` 即可。
