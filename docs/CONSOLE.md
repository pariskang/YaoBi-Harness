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
| `--access-token` | 固定访问令牌；`--public` 时会自动生成一个 |
| `--ngrok-authtoken` / `--ngrok-region` | 覆盖 `NGROK_AUTHTOKEN` / 选择区域 |

## 四个视图

**诊疗运行** — 左侧填写病例，右侧看结果。
左侧支持：交付对象（患者/医师/研究者）、主诉、五个示例病例、当前用药（中英逗号分隔）、
患者状态标签（决定条件门控规则是否触发）、结构化事实 JSON、是否允许含剂量草案、是否启用 LLM 规划。
右侧标签页：

| 标签 | 内容 |
| --- | --- |
| 交付内容 | 该角色实际收到的答复，带角色对应的免责声明 |
| 规划与执行 | 规划来源（LLM 提案是否被采纳）、任务图、每个 Agent 的状态、工具与输出摘要 |
| 用药安全 | 相互作用发现，含严重度、机制、处理措施、命中药物与类别映射、触发条件 |
| 证据台账 | 每条证据的等级与可放行性；结论↔证据的绑定；外部来源的许可/版本/发布日期/检索时间 |
| 安全审查 | 终结节点裁决、告警、修复请求、缺证据的结论、已执行的检查清单 |
| 原始 JSON | 完整响应，便于排查 |

**用药速查** — 不跑完整病例，只做相互作用筛查。适合门诊快速核对。

**知识库** — 知识库统计、来源目录与许可状态（哪些启用、哪些被禁用及原因）、内置规则包全文。

## API

控制台的后端是普通 JSON API，可以被其他前端复用。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 存活检查 |
| GET | `/api/bootstrap` | LLM/知识库/许可策略/规则摘要/示例病例 |
| GET | `/api/rules` | 完整规则包与药物类别表 |
| POST | `/api/run` | `{complaint, role, facts, allow_prescription, use_llm}` → `{delivered, audit, meta}` |
| POST | `/api/interactions` | `{medications, conditions}` → 相互作用筛查结果 |
| POST | `/api/chat` | `{message, role, session_id?}` → `{session_id, reply, audit, meta}` |
| POST | `/api/chat/reset` | `{session_id}` → 清空该会话 |

`/api/run` 的返回结构固定为三段：`delivered`（角色化答复）、`audit`（推理记录，仅操作者）、
`meta`（放行状态、规划来源、预算、知识库与许可模式）。

```bash
curl -sS -X POST http://127.0.0.1:8000/api/run \
  -H 'Content-Type: application/json' \
  -d '{"complaint":"腰痛3月，久坐加重","role":"physician",
       "facts":{"medications":["布洛芬","华法林"],"conditions":["elderly"]}}'
```

## 限制

* **无身份认证、无多用户隔离。** 这是本地操作者工具。
* 对话会话只存在内存中，进程退出即丢失；持久化涉及临床记录留存，须另行设计。
* 一次只跑一个病例：共享的 `ToolRegistry` 与熔断器不为并发运行设计，服务端用锁串行化。
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
