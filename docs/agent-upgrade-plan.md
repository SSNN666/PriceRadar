# Agent 化改造计划

> 制定：2026-10-08 ｜ 状态：**待执行**（未动一行代码）
> 目标岗位：AI 应用开发 / Agent 开发 ｜ 相关：[next-steps.md](./next-steps.md)

---

## 一、为什么做：项目叫 Agent，却没有 Agent

`agent/pipeline.py` 的 `run()` 是一条**完全写死的确定性管线**（检索 → 匹配 → 算价）。LLM 只在两个边缘干活：`llm_nodes.extract_keyword`（提取关键词）与 `compose_summary`（写报告导语）。

> 面试官第一个问题「你的 Agent 在哪一步自主决策、怎么决定用哪个工具」——**现在答不上来。**

### 调研结论（2026-10-08）

| 来源 | 结论 |
|---|---|
| [AgentGuide 简历指南](https://github.com/adongwanai/AgentGuide/blob/main/docs/04-interview/20-resume-guide.md) | L1 认知 / L2 能跑 / L3 优化 / L4 生产工程；要求能被追问架构、数据、指标、取舍 |
| [InfoQ 字节 AI 应用岗复盘](https://xie.infoq.cn/article/61369d59b176e61b2e5d370be) | 加分项 = 错误码带**可重试语义**、安全边界、长期记忆去重 |
| [Agent 工程追问清单](https://www.gankinterview.cn/blog/ai-agent-engineering-interview-tool-callingmemoryragplanning--executionfailure-r) | 高频追问：工具超时重试、schema 校验、降级兜底、防死循环、量化评测 |
| [腾讯云多轮对话面试题](https://cloud.tencent.cn/developer/article/2745023) | 必须区分 Chat History / Chat Memory / Session State / Long-Term Memory |
| [招聘 JD 实测](https://career.hebut.edu.cn/correcruit/content/id/80121.html) | 明写要求：**流式处理**、**长短期记忆管理**、**多工具调用**、**MCP** |

**关键判断：`智能客服 + LangGraph + RAG + 准确率提升 15%` 这类简历已高度同质化，可能 0 约面。** 要的是「被真实环境验证过、处理过失败场景」。

### 本项目的真实优势（不能丢）

真实 API、真实账号限流、真实类目空洞、真实跨平台命中率只有 ~11%、真实降级路径——这些是同质化项目**编都编不出来**的。

> **问题不是「AI 含量不够」，而是这些真实工程能力没有被 AI 的方式组织起来。**

### 改造后的定位

> **确定性内核 + Agent 外壳**
> 钱仍然不由 LLM 算（到手价引擎是纯函数），但 LLM 真正**自主决定调哪个工具、调几次、何时停**；失败有明确语义、有轮次上限、有降级回确定性管线的兜底。

---

## 二、红队修订（**这几条推翻了初版设计，务必先读**）

初版计划经红队复核，以下五处结论被推翻或修正：

### ① 「retryable 交给模型决定」是错的 —— 必须按工具定策略

- pdd/jd 客户端**内部已经重试 3 次**（`sources/retry.py`）。到 agent 层时，transient 失败已经是「重试后仍失败」。
- 再让模型重试 = **3 倍配额** + 触发已知的**拼多多限流**（`pdd-api-findings.md §10`，限制 >20 分钟不自动恢复，判据是「调用量 vs 出单量」而本项目账号天然不出单）。
- 而且 `is_transient` 对 `PddApiError` / `JdApiError` / `BochaError` / `LlmError` **一律返回 False**——它们是包装错误，cause 藏在 message 字符串里。

**改为**：**按工具定义重试策略**（pdd 系 no-retry；bocha/llm 可 retry-once），`is_transient` 只作为输入之一（检查 `__cause__`），并用「**单请求外部调用总预算**」兜底。

### ② 「复用 execute_similar」写错了 —— 它在服务层且已序列化

`execute_similar` 位于 `service/api.py:256`，返回**已经过 `offer_payload` 序列化的 dict**。agent 层调它 = 反向依赖 service 层，并把 Web 展示结构带进 agent。

**改为**：把核心（`recommend_similar` + `compute_price` + `merge_user_coupons`）**下沉**到 agent 层；`service/api.py` 的 `execute_similar` 改成薄委托，**名字与签名不变**（`tests/test_service.py` 里 6 处 `monkeypatch.setattr` 一处都不用改）。

### ③ 降级会重复打外部 API —— 必须做单请求结果 memo

若 `compare_prices` 已跑过 `pipeline.run`，触顶降级时再跑一遍 = **双倍拼多多调用 + 双倍延迟**，正是限流高危场景。

**改为**：工具执行器在**单请求内按 `(tool, args)` memo 完整结果对象**——摘要只给 LLM，全量结构留在服务端。降级 = 「复用最后一次可用的完整结果 + 确定性渲染」；只有完全没有可用结果时才真跑 `pipeline.run()`。memo 顺带解决「模型用相同参数把同一工具调两次」。

### ④ 单轮工具数上限有协议坑

助手消息带 N 个 `tool_calls`，若只执行前 K 个，**剩余的必须补上对应的 `tool` 消息**（如 skipped 说明），否则下一轮请求因 `tool_call_id` 不成对直接 400。

**截断调用 ≠ 截断消息。**

### ⑤ ScriptedTransport 测不出「工具选择正确率」

脚本是自己写的，准确率必然 100%，面试官一问就穿。

**必须两层**：
- **(a) 离线机制测试**（配对补齐、预算截断、降级不重跑）——用 `ScriptedTransport`
- **(b) 真实响应录制回放**——把真实 DeepSeek 的 `tool_calls` 响应录成 jsonl，离线回放打分。**仓库目前没有「录制回放」范式，这是能报出真数字的关键**

---

## 三、已实测验证的技术假设

在**本项目实际安装的版本**（starlette 1.7.0 / fastapi 0.142.2）上跑过验证，不是纸上推演：

| 假设 | 结论 | 证据 |
|---|---|---|
| 同步生成器 + `StreamingResponse` 不阻塞事件循环 | ✅ | 源码 `self.body_iterator = iterate_in_threadpool(content)` |
| uvicorn 不会把流缓冲成一坨 | ✅ | 起真实 uvicorn + 真实 HTTP 客户端：4 帧在 **0.4 / 0.8 / 1.2 / 1.61 s** 分别到达；`Transfer-Encoding: chunked` |
| 断连时生成器 `finally` 会被及时执行 | ❌ **推翻** | `iterate_in_threadpool` 源码里**没有** `close()` 底层迭代器 → 取消逻辑**不能建在 `finally` 上** |
| DeepSeek 支持 `tools`/`tool_calls` | ✅ | 兼容 OpenAI；支持并行调用、最多 128 工具；**`tool_call_id` 强校验** |
| DashScope embedding 可用 | ✅ | OpenAI 兼容端点；`text-embedding-v3/v4`；每批最多 10 条、单条 8192 token |

---

## 四、实施方案（按执行顺序）

### 批 1 · actions 工具层（地基）

新建 `src/priceradar/agent/actions.py`，**4 个意图级工具**：

| 工具 | 内部复用 |
|---|---|
| `compare_prices(query, jd_channel=0, use_risk=False)` | `agent/pipeline.run` |
| `find_similar(goods_sign, ...)` | `PddClient.recommend_similar`（自服务层下沉） |
| `scan_risk(product)` | `risk/scanner` |
| `check_price_history(keyword)` | `insights/history` |

**砍掉两个**（红队建议，采纳）：
- ❌ `parse_link` 不做成 LLM 工具——链接的铁律是**原样透传、绝不清洗 URL**（`sources/links.py`），而 `extract_keyword` 已有链接短路。暴露给 LLM 只会多一次往返 + 多一个「模型改写 URL」的风险。
- ❌ `list_user_coupons` 是 UI 管理功能，agent 拿到没有可执行动作。

其余要求：
- `find_similar` 用**独立参数**，不用 `goods_sign | query` 联合参数（对 tool schema 不友好）
- 顺手修掉既存不一致：**MCP 的 `_compare_text` 不含用户券、不做 risk 聚类**，而 `execute_search` 两者都有
- 服务层 `execute_search` / `execute_similar` 改**薄委托**，**签名一字不动**（`tests/test_service.py` 6 处 monkeypatch 不受影响）
- 保持**模块级名字 + 可注入**（`tests/test_mcp_server.py` 打的是 `server.run` / `server.scan_with_env` / `server._llm_client`）
- 处理 `agent/tools.py` 的死注册表 `TOOLS`（零消费者、无 schema，注释里「LangGraph 节点消费」的说法已过期）；注意底层原语**不止管线在用**，`jobs/snapshot.py` 也在用

### 批 2 · Agent 循环（核心）

**2.1 `llm.py` 新增 `chat(messages, *, tools=None, ...)`**
- 返回原始 message + `usage`
- **`complete()` 原样保留**（它的 `(system, user)` 签名表达不了 `assistant.tool_calls` / `role=tool`，改它会让既有节点与 `tests/test_llm.py` 承担风险）

**2.2 `loop.py`**
- 从 actions 的 pydantic 模型导出 OpenAI 兼容 tool schema
- 硬约束：`max_steps`（默认 8）、单轮工具数上限、**单请求外部调用总预算**
- **摘要必须「可行动」**（否则链条断裂）：带工具名 + 关键 ID（`goods_sign`/`sku`/keyword）+ **确定性数值原样文本**（不让 LLM 转述数字）+ 可用状态。`find_similar` 要 `goods_sign`、`check_price_history` 要关键词——摘要只留「标题+价格」的话模型第二轮就调不动下一个工具
- 工具执行器 memo（见红队修订 ③）
- 截断调用必须补齐 tool 消息（见红队修订 ④）
- `on_event` 回调逐事件外发——**一套机制两个消费者**：SSE 转发前端 / 测试收集断言
- token 预算：**事前按字符粗估**（中文约 1 token/字的上界）+ **事后用真实 usage 校准**，超预算即停。不做假精确
- 工具描述里**写明成本与延迟**（「风险扫描约 ¥0.2/次、5-10s」），让模型把成本算进选择
- **快路径**：单关键词（当前 ~90% 用法）直接走 `compare_prices`，只有多意图/追问才进 loop——少一次 LLM 往返、少一个失败面

**2.3 CLI**：`python -m priceradar.agent.loop "帮我看看QCY N20哪里买划算，顺便有没有踩坑的"`

### 批 3 · SSE 流式 + 前端面板

- **同步生成器** + `StreamingResponse`（已实测；内部全是阻塞 urllib，**绝不能写 async 生成器**）
- **断连不建在 `finally` 上**（见上表推翻的那条）→ 取消不做，改为**每步落库**（见批 4）
- ⚠️ **`EventSource` 重连必然重跑查询**：服务端正常结束流 = 连接关闭，浏览器按标准当作断线，**约 3 秒后自动重发同一个 GET**。三层防护：
  1. 前端收到 `final`/`error` **立即 `es.close()`**
  2. 所有事件带 `id:`，重连时用 `Last-Event-ID` 回放已存事件而非重跑
  3. 响应头 `Cache-Control: no-cache` + **`X-Accel-Buffering: no`**（部署到 nginx 时没有它流会被缓冲成一坨，见 [deployment-runbook.md](./deployment-runbook.md)）
- 首 chunk 之后 app 级异常处理器**失效** → 流中途异常必须自己转成 `error` 事件
- 前端**另开 `#agent-out` 容器**，不碰 `#out`（避免与 `render()` 的整块 `innerHTML` 覆盖冲突；沿用既有 `#vision-out`/`#similar-out` 先例）
- 最终 payload 仍走既有 `render(d)`，复用全部卡片
- **显式决定**：agent 面板是**替换** `search()` 还是**并行模式**？注意 `filters` 是跨查询全局保留的，agent 结果进入后会继承筛选状态
- 路由参数名与 `/api/search` 对齐（q / jd_channel / llm / risk）

### 批 4 · 多轮会话（最小形态）

**结论：多轮需求真，但浅。** 核心链路「看短视频 → 搜图 → 找同款」本身是**线性工作流，不是对话**。真正需要多轮的只有两处：结果卡后的追问（「这价是历史低位吗」）、歧义澄清。

**最小实现**：
- `session_id` + **槽位**（`last goods_sign` / keyword / 商品名）+ **最近 1-2 个完整 turn**
- **不做滚动摘要**——64K 上下文 + 这类短会话，超预算要很久才发生，是纯成本
- Turn 完整性靠**显式 `turn_id`**，裁剪只允许**整 turn 丢弃**（滚动摘要路径最容易产生「半 turn」，那是 `tool_call_id` 配对校验的炸点）
- 会话库照 `user_store.py` 模式（sqlite + 内联 SCHEMA + 函数接收 connection），独立 `data/sessions.db`（与观测、用户资产同样刻意分库）

⚠️ **前置依赖**：PRD §4.2 承诺了「帮我盯着降价」，但**目前没有 `POST /api/watchlist`**（watchlist 靠手改 `data/watchlist.json`）。多轮 UI 一上线，点它就是**死胡同** → 要么先加这个 API，要么先不放这个按钮。

### 批 5 · 治理与评测

- **真实 token 计量**：`trace.py` 现在用 `llm_deepseek: ¥0.002/次` 拍脑袋估算 → 读 `usage`。**新增 token 字段，不改 `est_cost` 语义**（否则 `tests/test_trace.py:23` 的断言要连带改）
- **缓存前缀优化**（简历上最硬的一条）：
  DeepSeek 定价中**缓存命中 vs 未命中差 30–50 倍**，而峰谷价差只有 2 倍（[官方定价](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)）。对策：**固定长前缀（system prompt + 工具定义）前置，易变内容（时间戳/query/轮次）后移**。
  → 话术：「工具定义前置 + 易变内容后移，缓存命中率 X%，单次成本从 ￥A 降到 ￥B」
- **`evaluation/agent_eval.py`**：照 `run_eval.py` 的 `evaluate(records)->dict` + `render_report(...)` 复用，产出 `results_agent.md`
  - 指标：工具选择正确率 / 平均轮次 / 降级率 / 错误恢复率
  - **必须含真实响应 fixture 回放**（见红队修订 ⑤）
  - 护栏照 `tests/test_eval.py:40 test_real_dataset_accuracy_floor`：正确率 ≥X%、降级率 ≤Y%

### 批 6 · 向量层：**先测量再决定**（默认不做）

红队评估：作为「分档第 3 层」**测不出增益，且有明确掉分风险**。

| 候选落点 | 判断 |
|---|---|
| 分档（A/B/excluded） | ❌ 大概率**掉**准确率——`N20`/`N60` 余弦近正是 `model_swap` 困难负例 |
| `_GROUP_SIMILARITY` 折叠 | ❌ 已有「同型号 + 同价」双前置，向量无感 |
| SIMILAR 档排序 / `_pick_source` | ❌ 现评测不测排序，无法归因 |
| **`channel_type=3` 推荐集去噪** | ✅ **唯一可测量**——现精度仅 33-47%（仓库自记）。以「锚点→推荐集」为标注单元复用 `label_tool.py` 盲标范式，测 **precision@5/@10 过滤前后** |

⚠️ **叙事一致性问题**：PRD 与 `resume-strategy.md` 承诺的是 **bge-small-zh 本地**（延迟/成本/隐私），改用 DashScope API 会与面试标准答案**自相矛盾**，需同步改说法；且**不为此引入 torch（约 2GB）**——零第三方依赖是本仓库难得的一致性优势。

**若测完无增益**：把**负结果**写进 docs。面试话术是「我实现并**评估**了向量层，用数据把它限制在召回/排序」，远强于「我加了向量库」——**但前提是这个实验真的跑了**。

---

## 五、执行顺序（按简历价值 / 单位工作量）

1. **actions + loop + `chat()` + 离线测试** ← 地基，每条测试都是面试可讲的边界设计
2. **agent_eval**（机制测试 + 真实 fixture 回放 + floor）← **没有它，第 1 件的面试价值打对折**
3. **SSE + 前端面板** ← 演示物（GIF/截图是面试第一眼）
4. **MCP 去重** ← 很便宜，但**必须等 actions 语义定型**，否则改两遍
5. **多轮会话**
6. **向量层**（先测量）

### 两天最简切法（每步可独立展示）

| 时段 | 内容 | 展示物 |
|---|---|---|
| Day1 上午 | 抽 actions：下沉 `execute_similar` 核心与 `merge_user_coupons`；统一 compare 语义；MCP 签名与注入点不变 | diff 证明只剩一份实现 |
| Day1 下午 | `llm.py` 加 `chat()`（`complete` 不动）+ ScriptedTransport 测试 | 回放一条真实 tool_calls 响应 |
| Day1 晚 | `loop.py`：schema、max_steps、外部调用预算、结果 memo、AgentStep、降级复用 | 一条「工具失败 → 改道/降级」的 trace |
| Day2 上午 | `execute_agent` + `/api/agent/stream`（同步生成器、错误转事件、`id:`、no-cache 头）+ TestClient stream 测试；MCP 委托 actions | curl 里逐事件进度 |
| Day2 下午 | 前端 `#agent-out` + EventSource（final 即 `es.close()`）+ 复用 `render(d)`；`evaluation/agent_eval.py`；更新 README/next-steps | agent_eval 指标表 |

**这两天明确不做**：session 表、滚动摘要、向量层、SSE 断线事件回放（先用 `es.close()`）。

---

## 六、两条不能破的约束

1. **所有新路径必须可离线注入打桩**（我的沙箱无 DNS/外网，真实调用只能你来跑）
2. **`execute_search` / `to_payload` 的签名与契约测试不动**

---

## 七、不做（及面试时怎么答）

- **不硬塞 RAG / 向量库**：知识源是**结构化实时价格**而非文档；「检索增强」在这里的正确形态是结构化过滤 + 平台原生相似推荐。风险扫描本身就是 Retrieval-Augmented，讲清楚即可。
- **不上 LangGraph**：现有流程是「线性 + 循环」，图的真正价值在条件边 / checkpoint / 人机协同，这里用不满 → 会被问「是不是过度设计」。自研循环每一行都能讲清，且能正面回答「为什么不用框架」。
- **不碰任何有副作用的动作**：全部工具都是只读检索与计算，Agent 循环没有任何可造成外部影响的操作（这也是「安全边界」那一条的回答）。

---

## 八、开工前必做（不依赖本计划）

- ⏰ **每日价格快照别断**——10 月底~双11 期间的涨价起点是唯一「过期不补」的资产（见 [next-steps.md](./next-steps.md)）
