# Agent 行为评测

> ⚠️ **这些指标来自真实录制，不是当前模型的实时表现。**
> 录制于 2026-10-09T13:27:40 ~ 2026-10-09T14:11:49（分批录制），模型 `deepseek-chat`，共 8 个场景。
> 要刷新就重跑 `python -m evaluation.agent_eval --live`。
> 期望的工具集合由人工撰写（`evaluation/agent_scenarios.jsonl`），可随时改。
> 回放全程零网络、零 API 花费。

## 总体指标

| 指标 | 值 | 说明 |
|---|---|---|
| 工具选择正确率 | 100.0% | 调用集合与期望**完全一致**的比例 |
| 工具召回率 | 100.0% | 期望的工具里被调到的比例（宽松） |
| 无多余调用率 | 100.0% | 没有调期望之外的工具 |
| 降级率 | 0.0% | 未走到 `finish`（触顶/LLM 失败/未调工具） |
| 平均轮次 | 1.12 | 发生过工具调用的轮数 |
| 平均提示 token | 3229 | 多轮累计 |
| 平均缓存命中率 | 79.9% | 命中与未命中单价差 30-50 倍 |
| 回放异常 | 0 | 非 0 说明循环行为与录制时不一致 |
| 硬失败后不重复试探 | 100.0% | 2 个真实失败场景里，没有多余重试的比例 |

## 逐条

| 场景 | 期望工具 | 实际调用 | 完全一致 | 轮次 | 结束原因 |
|---|---|---|---|---|---|
| category_gap | compare_prices | compare_prices | ✅ | 1 | finish |
| multi_intent | compare_prices、scan_risk | compare_prices、scan_risk | ✅ | 1 | finish |
| no_extra_call | compare_prices | compare_prices | ✅ | 1 | finish |
| non_original_link | compare_prices | compare_prices | ✅ | 1 | finish |
| price_only | compare_prices | compare_prices | ✅ | 1 | finish |
| risk_only | scan_risk | scan_risk | ✅ | 1 | finish |
| similar | compare_prices、find_similar | compare_prices、find_similar | ✅ | 2 | finish |
| timing | check_price_history | compare_prices、check_price_history | ✅ | 1 | finish |

## 失败路径（这些场景里工具**真的失败了**，看模型怎么处理）

| 场景 | 硬失败次数 | 多余重试 | 模型最终怎么说的 |
|---|---|---|---|
| category_gap | 1 | 0 | 这个品类比不了价，如实说明一下：  **显示器（27寸 4K）在拼多多侧没有可比的商品**，工具明确说这不是故障，而是整机与外设类目（手机、平板、显示器、电视、键盘、鼠标等）在数据… |
| non_original_link | 1 | 0 | 这个链接查不了，不是我不给力——拼多多那边直接拒了。  原因：你这条链接缺少 `uin` 这类参数，不是拼多多 App「分享 → 复制链接」出来的原始链接，平台会拒绝反查。这个错误… |
