"""Agent 工具调用循环（自研，不依赖编排框架）。

这是项目从「写死的管线」变成「真 Agent」的地方：由 **LLM 自己决定**调哪个工具、
调几次、什么时候停；工具是 `agent/actions.py` 里的意图级能力。

分工原则（PRD §5.2 的延伸）：
- **LLM 只做决策与表达**，不产生任何数字结论——金额一律来自确定性工具
- 循环本身是确定性的：轮次上限、调用预算、上下文预算、降级路径全部写死

为什么不用 LangGraph 等框架：这里的形态是「一条循环 + 工具调用」，
图的真正价值在条件边 / checkpoint / 人机协同，本场景用不满，引入即过度设计；
自研循环每一行都能讲清，也便于把失败语义做精确。

三个容易踩的坑（都有测试盯着，改动前先读）：
1. **截断调用 ≠ 截断消息**。助手消息里带 N 个 `tool_calls`，若只回 K 个 `tool`
   消息，下一轮请求会因 `tool_call_id` 不成对直接 400。被跳过的调用也必须补一条。
2. **只有成功结果才可复用，失败必须能重试**。若把失败也 memo 住，
   模型看到 `retryable=True` 想重试时会被缓存挡住，恢复能力直接失效。
   反过来，**不可重试的失败**也不再重打（省配额，hint 里已说明别试）。
3. **降级绝不重跑外部调用**。拼多多账号限流是已知生产事故（判据是
   「调用量 vs 出单量」，本项目账号天然不出单），所以触顶时优先复用循环内
   已取得的 `PipelineResult`，只有一次都没成功过才真跑兜底。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from priceradar.agent.actions import ACTIONS, _llm_client, compare_prices, openai_tools, run_action
from priceradar.agent.llm import ChatReply, DeepSeekClient, LlmError, Usage
from priceradar.agent.pipeline import PipelineResult
from priceradar.trace import Trace

MAX_STEPS = 8              # 轮次上限（防死循环）
MAX_CALLS_PER_STEP = 3     # 单轮工具调用上限（超出即跳过，但**必须补 tool 消息**）
MAX_TOOL_CALLS = 12        # 单请求外部调用总预算（防模型反复试探烧配额）
CHAR_BUDGET = 24000        # 上下文预算（字符；中文约 1 token/字的上界）
FEEDBACK_LIMIT = 1500      # 单条工具结果回灌上限
MAX_TOKENS = 1024

# ⚠️ 必须是**固定常量**、不含任何查询插值：它是每轮都重复的长前缀，
# 而 DeepSeek 的缓存命中与未命中单价差 30-50 倍（峰谷差只有 2 倍）。
# 易变内容（用户查询、轮次）一律放到它后面。
SYSTEM_PROMPT = """你是「真价雷达」的购物决策助手，帮用户比较各平台的**到手价**并给出判断。

工作方式：
- 你只能通过工具获取事实。**绝不允许编造、估算或"回忆"任何价格、销量、评分数字。**
- 引用数字时**原样照抄**工具返回的结果，不要四舍五入、不要换算、不要合并。
- 用户只问比价时，直接调 compare_prices，然后给结论即可，不要调无关的工具。
- 用户有多个诉求（例如"哪里买划算，顺便有没有什么坑"）时，可以在一轮里同时调用多个工具。
- 想找"一样的但更便宜的"时，用 compare_prices 结果里的 goods_sign 调 find_similar。
- 问到"是不是好价/该不该等"用 check_price_history；问到"口碑/踩坑"用 scan_risk。

工具失败时看返回里的 retryable 与建议：
- retryable 为 true：可以原样重试一次；若仍失败就改用其它工具或如实说明。
- retryable 为 false：原样重试毫无意义，请换关键词/换工具，或直接如实告诉用户这条路走不通。
- **不要**为了凑结论而在失败后反复换着参数试探，试两次不行就收手。

诚实要求：
- 工具说"未找到同款""样本不足""检索结果不代表全部"时，必须如实转达，不要粉饰。
- 金额均为平台公开优惠口径的测算值，最终以结算页为准；提到价格时请带上这个前提。

语言与风格（重要）：
- **全程用中文**。包括调用工具前的那句简短说明，也不要写英文。
- 简洁、口语化，先给结论再给依据。不要输出 markdown 表格。"""

_STOP_REASON = {
    "llm_unavailable": "未配置 LLM，已直接走确定性比价",
    "llm_error": "LLM 调用失败，已降级为确定性比价",
    "max_steps": "达到轮次上限，已降级为确定性比价",
    "budget": "达到调用/上下文预算，已降级为确定性比价",
    "no_tools": "模型未调用任何工具，已降级为确定性比价",
}

_SKIPPED_NOTE = "该调用未被执行（超出本轮或本次请求的调用预算）。如需，请精简后在下一轮重新发起。"


@dataclass
class ToolExecution:
    id: str
    name: str
    arguments: dict
    ok: bool
    summary: str = ""
    error: str = ""
    retryable: bool = False
    hint: str = ""
    from_memo: bool = False
    # 工具的**完整返回值**（领域对象）。只有 `summary` 会进模型上下文，
    # 全量结果留在服务端——降级时就不必重跑外部调用（拼多多限流是已知事故）。
    value: Any = None

    def feedback(self, limit: int = FEEDBACK_LIMIT) -> str:
        """回灌给模型的内容（定长）。失败时必须带上 retryable 与建议。"""
        if self.ok:
            return _clip(self.summary, limit)
        text = f"[工具失败] {self.error}\nretryable: {str(self.retryable).lower()}"
        if self.hint:
            text += f"\n建议: {self.hint}"
        return _clip(text, limit)


@dataclass
class AgentStep:
    index: int
    content: str | None
    tool_calls: list[ToolExecution] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


@dataclass
class AgentRun:
    query: str
    stopped_by: str
    final_text: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    result: PipelineResult | None = None       # 确定性比价结果（降级或复用而来）
    usage: Usage = field(default_factory=Usage)
    trace: Trace | None = None
    # 本轮实际发出去的完整消息列表（含历史）。调试时能一眼看到模型究竟看到了什么；
    # 多轮场景下调用方靠它把新产生的 assistant / tool 消息落库。
    messages: list[dict] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return self.stopped_by != "finish"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    suffix = "…（已截断）"
    return text[: limit - len(suffix)] + suffix


def _canonical(arguments: dict) -> str:
    """参数规范化：键序不同但内容相同要算同一个调用（否则 memo 失效）。"""
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)


def _usage_of(trace: Trace) -> Usage:
    """`AgentRun.usage` 是 `trace.tokens` 的**视图**，不是第二个累加器。

    两处各累加一遍同样的数字迟早会漂（而且没人会发现）。trace 已经是这份数据的
    唯一持久化去处，那就让它当唯一来源。
    """
    tokens = trace.tokens
    return Usage(
        prompt_tokens=tokens.prompt,
        completion_tokens=tokens.completion,
        total_tokens=tokens.prompt + tokens.completion,
        cache_hit_tokens=tokens.cache_hit,
        cache_miss_tokens=tokens.cache_miss,
    )


def _assistant_message(reply: ChatReply) -> dict:
    """回填助手消息。arguments 用**原始字符串**，不重新序列化。"""
    return {
        "role": "assistant",
        "content": reply.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.raw_arguments},
            }
            for call in reply.tool_calls
        ],
    }


def _emitter(on_event: Callable[[dict], None] | None) -> Callable[[dict], None]:
    """事件外发：**一套机制两个消费者**（SSE 转发前端 / 测试收集断言）。

    词汇表（本层只发**进度**，不发终止）：
    - `{"type": "step", "index": n, "content": str|None}` —— 某一轮模型给出了工具调用
    - `{"type": "tool", ...}` —— 单个工具执行完毕（含 ok / summary / from_memo）

    终止信息在 `AgentRun` 的返回值里；由调用方决定怎么表达（SSE 层用 `final`）。
    **不要在这里新增 `final` / `error` 这类名字**——会与传输层撞车。

    回调失败必须吞掉——前端断连时回调会抛异常，绝不能因此中断 Agent 运行。
    """
    if on_event is None:
        return lambda event: None

    def emit(event: dict) -> None:
        try:
            on_event(event)
        except Exception:
            pass

    return emit


def _execute(memo: dict, call, trace: Trace) -> ToolExecution:
    """执行一个工具并把结果放进 memo。

    复用规则（三条都是刻意的）：
    - 成功结果 → 复用：模型用相同参数重复调同一工具时不重打外部接口
    - **不可重试的失败 → 也复用**：重打没用，省配额（hint 里已写明别试）
    - **可重试的失败 → 不复用**，允许模型再试一次；否则它看到 retryable=true
      想重试时会被缓存挡回去，恢复能力直接失效
    """
    key = (call.name, _canonical(call.arguments))
    cached = memo.get(key)
    if cached is not None and (cached.ok or not cached.retryable):
        return ToolExecution(
            id=call.id,
            name=call.name,
            arguments=call.arguments,
            ok=cached.ok,
            summary=cached.summary,
            error=cached.error,
            retryable=cached.retryable,
            hint=cached.hint,
            value=cached.value,
            from_memo=True,
        )

    with trace.span(f"tool:{call.name}", _canonical(call.arguments)[:60]):
        outcome = run_action(call.name, call.arguments)
    fresh = ToolExecution(
        id=call.id,
        name=call.name,
        arguments=call.arguments,
        ok=outcome.ok,
        summary=outcome.summary,
        error=outcome.error,
        retryable=outcome.retryable,
        hint=outcome.hint,
        value=outcome.value,
    )
    memo[key] = fresh
    return fresh


def _memoized_result(memo: dict) -> PipelineResult | None:
    """从 memo 里取出最后一次成功的比价结果（降级时复用，避免重跑外部调用）。"""
    for outcome in reversed(list(memo.values())):
        if outcome.ok and isinstance(outcome.value, PipelineResult):
            return outcome.value
    return None


def _degrade(
    query: str,
    stopped_by: str,
    *,
    memo: dict,
    steps: list[AgentStep],
    trace: Trace,
    compare_kwargs: dict,
    messages: list[dict] | None = None,
) -> AgentRun:
    """降级：优先复用循环内已取得的比价结果，**绝不重跑**；一次都没有才真跑兜底。"""
    reason = _STOP_REASON.get(stopped_by, stopped_by)
    result = _memoized_result(memo)
    reused = result is not None
    if result is None:
        with trace.span("fallback_pipeline", "确定性兜底"):
            result = compare_prices(query, **compare_kwargs)
    result.notes.append(reason + ("（复用本轮已取得的结果）" if reused else ""))
    return AgentRun(
        query=query,
        stopped_by=stopped_by,
        final_text=result.summary or "",
        steps=steps,
        result=result,
        usage=_usage_of(trace),
        trace=trace,
        messages=messages or [],
    )


def run_agent(
    query: str,
    *,
    client: Any = None,
    messages: list[dict] | None = None,
    on_event: Callable[[dict], None] | None = None,
    max_steps: int = MAX_STEPS,
    max_calls_per_step: int = MAX_CALLS_PER_STEP,
    max_tool_calls: int = MAX_TOOL_CALLS,
    char_budget: int = CHAR_BUDGET,
    trace: Trace | None = None,
    **compare_kwargs: Any,
) -> AgentRun:
    """跑一轮 Agent。永不抛异常——失败一律降级为确定性比价（降级不阻塞的延续）。

    `client` 为 None 时按「未配置 LLM」处理；不传则尝试 `DeepSeekClient.from_env()`。
    `compare_kwargs` 透传给兜底的 `compare_prices`（如 jd_channel / use_risk）。

    `messages` 用于**多轮**：由 `context.build_messages` 组装好（固定 system →
    结构化槽位 → 最近若干完整轮 → 本轮提问）。不传则是单轮（system + query）。
    传入的列表会被复制，循环只在副本上追加。
    """
    trace = trace or Trace()
    emit = _emitter(on_event)
    messages = list(messages) if messages else [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    if client is None:
        client = _llm_client()
    if client is None:
        return _degrade(
            query, "llm_unavailable", memo={}, steps=[], trace=trace,
            compare_kwargs=compare_kwargs, messages=messages,
        )

    tool_schemas = openai_tools()
    memo: dict[tuple[str, str], ToolExecution] = {}
    steps: list[AgentStep] = []       # 只记**发生过工具调用**的轮次；最终答复不算一轮
    executed_total = 0

    for index in range(max_steps):
        # 上下文预算按实际消息体积算（不做增量记账——那是最容易算错的地方）
        if len(json.dumps(messages, ensure_ascii=False, default=str)) > char_budget:
            return _degrade(query, "budget", memo=memo, steps=steps, trace=trace, compare_kwargs=compare_kwargs, messages=messages)

        with trace.span("llm_chat", f"第 {index + 1} 轮"):
            try:
                reply = client.chat(messages, tools=tool_schemas, max_tokens=MAX_TOKENS)
            except LlmError as exc:
                trace.count("llm_deepseek")
                return _degrade(query, "llm_error", memo=memo, steps=steps, trace=trace, compare_kwargs=compare_kwargs, messages=messages)
        trace.count("llm_deepseek")
        # 真实 token 计量入 trace（缓存命中/未命中分项）——这是成本的大头，
        # 也是「固定前缀前置」这套设计唯一能被验证的地方
        trace.record_usage(
            prompt=reply.usage.prompt_tokens,
            completion=reply.usage.completion_tokens,
            cache_hit=reply.usage.cache_hit_tokens,
            cache_miss=reply.usage.cache_miss_tokens,
        )

        if not reply.tool_calls:
            if executed_total == 0:
                return _degrade(query, "no_tools", memo=memo, steps=steps, trace=trace, compare_kwargs=compare_kwargs, messages=messages)
            # ⚠️ 这里**刻意不发终止事件**：事件词汇表按层分开——本循环只发"进度"
            # （step / tool），终止由调用方按返回值自己决定怎么表达。
            # 曾经在这里 emit 过一个 {"type":"final"}，结果与 SSE 层的真·final 撞名，
            # 前端收到第一个（没有 payload）就 close 了，比价卡片永远出不来。
            return AgentRun(
                query=query,
                stopped_by="finish",
                final_text=(reply.content or "").strip(),
                steps=steps,
                result=_memoized_result(memo),
                usage=_usage_of(trace),
                trace=trace,
                messages=messages,
            )

        emit({"type": "step", "index": index, "content": reply.content})
        messages.append(_assistant_message(reply))

        step = AgentStep(index=index, content=reply.content)
        skipped_ids: list[str] = []
        total_budget_exhausted = False
        for call in reply.tool_calls:
            # 单轮上限是**限流**：跳过并把原因喂回模型，循环继续（它可以下一轮再来）
            if len(step.tool_calls) >= max_calls_per_step:
                step.skipped.append(call.id)
                skipped_ids.append(call.id)
                continue
            # 总量上限是**硬顶**：本次请求的外部调用额度用完了，收手
            if executed_total >= max_tool_calls:
                step.skipped.append(call.id)
                skipped_ids.append(call.id)
                total_budget_exhausted = True
                continue
            execution = _execute(memo, call, trace)
            if not execution.from_memo:
                executed_total += 1
            step.tool_calls.append(execution)
            emit(
                {
                    "type": "tool",
                    "index": index,
                    "name": execution.name,
                    "arguments": execution.arguments,
                    "ok": execution.ok,
                    "summary": execution.feedback(),
                    "from_memo": execution.from_memo,
                }
            )

        # ⚠️ 截断调用 ≠ 截断消息：被跳过的调用也必须补 tool 消息，否则下一轮 400
        for execution in step.tool_calls:
            messages.append({"role": "tool", "tool_call_id": execution.id, "content": execution.feedback()})
        for call_id in skipped_ids:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": _SKIPPED_NOTE})

        steps.append(step)
        if total_budget_exhausted:
            return _degrade(query, "budget", memo=memo, steps=steps, trace=trace, compare_kwargs=compare_kwargs, messages=messages)

    return _degrade(query, "max_steps", memo=memo, steps=steps, trace=trace, compare_kwargs=compare_kwargs, messages=messages)


def render(run: AgentRun) -> str:
    """CLI 文本输出（终端可读）。"""
    lines = [f"==== 真价雷达 · Agent 循环 ====", f"查询：{run.query}"]
    for step in run.steps:
        if step.content:
            lines.append(f"\n[第 {step.index + 1} 轮] {step.content.strip()}")
        for execution in step.tool_calls:
            mark = "✓" if execution.ok else "✗"
            memo = "（复用）" if execution.from_memo else ""
            lines.append(f"  {mark} {execution.name}{memo} {_canonical(execution.arguments)[:80]}")
            if execution.ok:
                head = execution.summary.strip().splitlines()[:3]
                lines.extend(f"      {line.strip()}" for line in head)
            else:
                lines.append(f"      {execution.error}（retryable={execution.retryable}）")
        if step.skipped:
            lines.append(f"  △ 跳过 {len(step.skipped)} 个调用（超出预算）")

    # 最终答复是**最重要的产出**，必须显式打印（它不属于任何"工具轮"）
    if run.stopped_by == "finish":
        lines.append(f"\n【最终答复】\n{run.final_text or '（模型未给出文字答复）'}")
    else:
        lines.append(f"\n※ {_STOP_REASON.get(run.stopped_by, run.stopped_by)}")
    if run.result is not None:
        from priceradar.agent.pipeline import render as render_pipeline

        lines.append("")
        lines.append(render_pipeline(run.result))
    if run.trace is not None:
        # summary_line 已含 token 分项、缓存命中率与成本——不另起一行重复一遍
        lines.append("")
        lines.append(run.trace.summary_line())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="真价雷达 · Agent 循环（LLM 自主决定调哪些工具）")
    parser.add_argument("query", help="自然语言，如：帮我看看QCY N20哪里买划算，顺便有没有踩坑的")
    parser.add_argument("--jd-channel", type=int, default=0, help="京东京粉频道（0=自动路由，-1=关闭京东侧）")
    parser.add_argument("--no-llm", action="store_true", help="禁用 LLM（直接走确定性兜底，用于对比）")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="轮次上限")
    args = parser.parse_args(argv)

    client = None
    if not args.no_llm:
        client = _llm_client()

    run = run_agent(
        args.query,
        client=client,
        max_steps=args.max_steps,
        on_event=lambda event: print(f"  · {event['type']}: {event.get('name') or event.get('stopped_by') or ''}", file=sys.stderr),
        jd_channel=args.jd_channel,
    )
    print(render(run))


if __name__ == "__main__":
    main()
