"""Agent 工具调用循环测试（ScriptedTransport 脚本化多轮，全离线）。

这里测的是**机制**：轮次上限、结果 memo、截断补齐、降级不重跑、事件外发。
「工具选择是否正确」属于模型行为，ScriptedTransport 自写脚本必然 100%，
测不出真实水平——那部分要用真实响应录制回放（见 docs/agent-upgrade-plan.md 批 ⑤）。
"""

import json
from decimal import Decimal

import pytest

import priceradar.agent.loop as loop
from priceradar.agent.actions import ACTIONS, ActionResult, Action
from priceradar.agent.llm import DeepSeekClient, LlmError
from priceradar.agent.pipeline import ConfirmedItem, OfferGroup, PipelineResult
from priceradar.domain.models import Offer, Platform
from priceradar.engine.pricing import compute_price
from priceradar.risk import RiskSummary
from tests.test_llm import ScriptedTransport

D = Decimal


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """把降级兜底的 `loop.compare_prices` **默认打桩**。

    起因：有两个用例只桩了 `run_action`，脚本化的「无工具调用」答复会走到
    `no_tools` 降级，而那里的 `compare_prices` 是真的——于是每次 `pytest`
    都会发出真实的 DeepSeek 与拼多多请求。花钱、烧配额，而且拼多多账号正是
    项目文档里点名限流敏感的那个。

    需要断言兜底行为的用例，在自己那里再 `monkeypatch.setattr` 覆盖即可
    （后设置的赢）。
    """
    monkeypatch.setattr(loop, "compare_prices", lambda keyword, **kwargs: make_result(keyword))


# ---------- 夹具 ----------

def make_result(keyword="QCY N20") -> PipelineResult:
    offer = Offer(platform=Platform.PDD, sku_id="a", title="QCY N20 降噪耳机", list_price=D("99"),
                  source_extra={"goods_sign": "SIGN_A"})
    return PipelineResult(
        keyword=keyword,
        pdd_count=20,
        jd_count=0,
        source=offer,
        confirmed=[ConfirmedItem(offer=offer, reason="源商品")],
        groups=[OfferGroup(representative=offer, members=[offer], final_price=D("99"), reason="源")],
        similar_count=0,
        excluded_summary={},
        breakdowns=[(offer, compute_price(offer))],
    )


def call_tool(call_id: str, name: str, **args) -> dict:
    return {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "prompt_cache_hit_tokens": 800,
                  "prompt_cache_miss_tokens": 200},
    }


def answer(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


def patch_actions(monkeypatch, outcomes: dict) -> list:
    """按 name 返回预设结果，并记录每次调用（用于断言 memo 与预算）。"""
    calls: list[tuple[str, dict]] = []

    def fake_run_action(name, arguments, **kwargs):
        calls.append((name, dict(arguments)))
        outcome = outcomes.get(name)
        if outcome is None:
            return ActionResult(ok=False, action=name, error="未知工具", hint="换一个")
        return outcome() if callable(outcome) else outcome

    monkeypatch.setattr(loop, "run_action", fake_run_action)
    return calls


def ok(name: str, value, summary: str = "摘要") -> ActionResult:
    return ActionResult(ok=True, action=name, value=value, summary=summary)


def make_agent(monkeypatch, *responses, **kwargs):
    transport = ScriptedTransport(*responses)
    client = DeepSeekClient("k", transport=transport)
    return transport, client


# ============================================================
# 1. 基本循环
# ============================================================

def test_single_tool_then_answer(monkeypatch):
    patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result(), "比价摘要")})
    transport, client = make_agent(monkeypatch, call_tool("c1", "compare_prices", query="QCY N20"), answer("最低 ￥99"))

    run = loop.run_agent("QCY N20 哪里买划算", client=client)
    assert run.stopped_by == "finish"
    assert run.final_text == "最低 ￥99"
    assert len(run.steps) == 1
    assert run.steps[0].tool_calls[0].name == "compare_prices"
    # 降级/复用用的确定性结果已被捕获
    assert run.result is not None and run.result.keyword == "QCY N20"


def test_parallel_tool_calls_in_one_step(monkeypatch):
    """多意图：一轮里同时要比价与风险扫描。"""
    patch_actions(monkeypatch, {
        "compare_prices": ok("compare_prices", make_result()),
        "scan_risk": ok("scan_risk", RiskSummary(product="x", queries=["q"], items=[], degraded=True)),
    })
    combined = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "compare_prices", "arguments": '{"query":"x"}'}},
                    {"id": "c2", "type": "function", "function": {"name": "scan_risk", "arguments": '{"product":"x"}'}},
                ],
            },
            "finish_reason": "tool_calls",
        }],
    }
    transport, client = make_agent(monkeypatch, combined, answer("两件事都办了"))
    run = loop.run_agent("比价顺便看看坑", client=client)
    assert [call.name for call in run.steps[0].tool_calls] == ["compare_prices", "scan_risk"]


def test_system_prompt_is_a_fixed_prefix(monkeypatch):
    """system prompt 必须是固定常量、不含 query 插值——它是每轮重复的长前缀，
    而 DeepSeek 的缓存命中与未命中单价差 30-50 倍。"""
    patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    transport, client = make_agent(monkeypatch, call_tool("c1", "compare_prices", query="x"), answer("好的"))
    loop.run_agent("随便什么查询", client=client)

    first = transport.calls[0]["messages"]
    assert first[0]["role"] == "system"
    assert "随便什么查询" not in first[0]["content"]
    assert first[1] == {"role": "user", "content": "随便什么查询"}


def test_custom_messages_are_used_for_multi_turn(monkeypatch):
    """多轮：调用方（Context Builder）组装好完整消息列表，循环只在副本上追加。"""
    patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    transport, client = make_agent(monkeypatch, call_tool("c1", "compare_prices", query="x"), answer("好"))
    history = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "system", "content": "【本会话已知的当前对象】商品「QCY N20」"},
        {"role": "user", "content": "上一问"},
        {"role": "assistant", "content": "上一答"},
        {"role": "user", "content": "那有没有踩坑的"},
    ]
    run = loop.run_agent("那有没有踩坑的", client=client, messages=history)

    assert transport.calls[0]["messages"][:5] == history       # 原样送出
    assert history[-1]["content"] == "那有没有踩坑的"           # 传入的列表未被就地改写
    assert len(run.messages) > len(history)                    # 返回值带完整消息列表


def test_run_messages_start_with_system_in_single_turn(monkeypatch):
    patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    transport, client = make_agent(monkeypatch, answer("好"))
    run = loop.run_agent("x", client=client)
    assert run.messages[0]["role"] == "system"
    assert run.messages[1] == {"role": "user", "content": "x"}


def test_tools_schema_is_sent_every_round(monkeypatch):
    """工具定义同样属于缓存前缀，必须每轮原样重发（顺序固定）。"""
    patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    transport, client = make_agent(monkeypatch, call_tool("c1", "compare_prices", query="x"), answer("好"))
    loop.run_agent("x", client=client)
    names_round_0 = [t["function"]["name"] for t in transport.calls[0]["tools"]]
    names_round_1 = [t["function"]["name"] for t in transport.calls[1]["tools"]]
    assert names_round_0 == names_round_1 == list(ACTIONS)


# ============================================================
# 2. 工具结果回灌：摘要而非全量
# ============================================================

def test_only_summary_is_fed_back_not_full_result(monkeypatch):
    """全量结果留在服务端 memo，进上下文的只有定长摘要（token 治理落点）。"""
    huge = "很长很长的摘要" * 2000
    patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result(), huge)})
    transport, client = make_agent(monkeypatch, call_tool("c1", "compare_prices", query="x"), answer("好"))
    loop.run_agent("x", client=client)

    tool_messages = [m for m in transport.calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert len(tool_messages[0]["content"]) <= loop.FEEDBACK_LIMIT
    assert tool_messages[0]["tool_call_id"] == "c1"


def test_assistant_tool_calls_are_echoed_verbatim(monkeypatch):
    """回填助手消息时用**原始** arguments 字符串，不重新序列化。"""
    patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    transport, client = make_agent(monkeypatch, call_tool("c9", "compare_prices", query="耳机"), answer("好"))
    loop.run_agent("x", client=client)
    echoed = [m for m in transport.calls[1]["messages"] if m.get("tool_calls")][0]
    assert echoed["tool_calls"][0]["id"] == "c9"
    assert json.loads(echoed["tool_calls"][0]["function"]["arguments"]) == {"query": "耳机"}


# ============================================================
# 3. 协议坑：截断调用必须补齐 tool 消息
# ============================================================

def test_truncated_tool_calls_still_get_tool_messages(monkeypatch):
    """单轮工具数超上限时，**截断调用 ≠ 截断消息**。

    助手消息里带 N 个 tool_calls，若只回 K 个 tool 消息，下一轮请求会因
    tool_call_id 不成对直接 400 —— 必须给被跳过的调用也补一条说明。
    """
    patch_actions(monkeypatch, {
        "compare_prices": ok("compare_prices", make_result()),
        "find_similar": ok("find_similar", None),
        "scan_risk": ok("scan_risk", None),
        "check_price_history": ok("check_price_history", None),
    })
    five_calls = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [
                    {"id": f"c{i}", "type": "function",
                     "function": {"name": name, "arguments": "{}"}}
                    for i, name in enumerate(["compare_prices", "find_similar", "scan_risk",
                                              "check_price_history", "compare_prices"])
                ],
            },
            "finish_reason": "tool_calls",
        }],
    }
    transport, client = make_agent(monkeypatch, five_calls, answer("好"))
    run = loop.run_agent("x", client=client, max_calls_per_step=2)

    assistant_msg = [m for m in transport.calls[1]["messages"] if m.get("tool_calls")][0]
    tool_msgs = [m for m in transport.calls[1]["messages"] if m["role"] == "tool"]
    assert len(assistant_msg["tool_calls"]) == 5
    assert len(tool_msgs) == 5                       # 一条不少
    assert {m["tool_call_id"] for m in tool_msgs} == {f"c{i}" for i in range(5)}
    assert len(run.steps[0].tool_calls) == 2         # 实际只执行了 2 个
    assert len(run.steps[0].skipped) == 3


# ============================================================
# 4. 失败恢复
# ============================================================

def test_tool_error_with_retryable_is_fed_back(monkeypatch):
    """工具失败不打断循环：错误 + retryable + hint 喂回模型，由它决定重试/换路。"""
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            return ActionResult(ok=False, action="scan_risk", error="URLError: dns down",
                                retryable=True, hint="网络瞬时故障，可稍后重试")
        return ok("scan_risk", RiskSummary(product="x", queries=["q"], items=[], degraded=True))

    patch_actions(monkeypatch, {"scan_risk": flaky})
    transport, client = make_agent(
        monkeypatch,
        call_tool("c1", "scan_risk", product="x"),
        call_tool("c2", "scan_risk", product="x"),
        answer("重试后拿到了"),
    )
    run = loop.run_agent("x 有没有坑", client=client)
    assert run.stopped_by == "finish"
    assert len(attempts) == 2
    first_tool_msg = [m for m in transport.calls[1]["messages"] if m["role"] == "tool"][0]
    assert "retryable" in first_tool_msg["content"] and "dns down" in first_tool_msg["content"]


def test_non_retryable_error_still_reaches_model(monkeypatch):
    patch_actions(monkeypatch, {
        "compare_prices": ActionResult(ok=False, action="compare_prices", error="PddApiError: 类目空洞",
                                       retryable=False, hint="换关键词或品类"),
    })
    transport, client = make_agent(monkeypatch, call_tool("c1", "compare_prices", query="显示器"), answer("这个品类没有"))
    run = loop.run_agent("显示器", client=client)
    assert run.stopped_by == "finish"
    tool_msg = [m for m in transport.calls[1]["messages"] if m["role"] == "tool"][0]
    assert "换关键词" in tool_msg["content"]


# ============================================================
# 5. 结果 memo（降级不重跑 —— 拼多多限流是已知生产事故）
# ============================================================

def test_identical_tool_call_is_memoized(monkeypatch):
    """模型用相同参数把同一工具调两次时，只执行一次。"""
    calls = patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    transport, client = make_agent(
        monkeypatch,
        call_tool("c1", "compare_prices", query="x"),
        call_tool("c2", "compare_prices", query="x"),
        answer("好"),
    )
    loop.run_agent("x", client=client)
    assert [name for name, _ in calls] == ["compare_prices"]     # 只执行一次


def test_non_retryable_failure_is_not_re_executed(monkeypatch):
    """不可重试的失败不再重打——hint 已写明别试，重打只是白烧配额。

    注意与上一条的区别：可重试的失败**必须**能重来（否则恢复能力失效），
    不可重试的失败**必须**缓存住（否则模型反复试探会烧掉配额）。这两条方向相反。
    """
    calls = patch_actions(monkeypatch, {
        "compare_prices": ActionResult(ok=False, action="compare_prices", error="PddApiError: 无权限",
                                       retryable=False, hint="换工具"),
    })
    transport, client = make_agent(
        monkeypatch,
        call_tool("c1", "compare_prices", query="x"),
        call_tool("c2", "compare_prices", query="x"),
        answer("好"),
    )
    monkeypatch.setattr(loop, "compare_prices", lambda keyword, **kw: make_result(keyword))
    loop.run_agent("x", client=client)
    assert len(calls) == 1          # 第二次直接复用失败结果，没有重打


def test_argument_order_does_not_defeat_memo(monkeypatch):
    """dict 键序不同但内容相同 → 仍算同一个调用。"""
    calls = patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    first = {"id": "c1", "type": "function",
             "function": {"name": "compare_prices", "arguments": '{"query":"x","jd_channel":0}'}}
    second = {"id": "c2", "type": "function",
              "function": {"name": "compare_prices", "arguments": '{"jd_channel":0,"query":"x"}'}}
    response = {"choices": [{"message": {"content": None, "tool_calls": [first]}, "finish_reason": "tool_calls"}]}
    response2 = {"choices": [{"message": {"content": None, "tool_calls": [second]}, "finish_reason": "tool_calls"}]}
    transport, client = make_agent(monkeypatch, response, response2, answer("好"))
    loop.run_agent("x", client=client)
    assert [name for name, _ in calls] == ["compare_prices"]


# ============================================================
# 6. 触顶与降级
# ============================================================

def test_max_steps_degrades_using_memoized_result(monkeypatch):
    """触顶时复用已取得的比价结果，**绝不重跑**（重跑 = 双倍拼多多调用 + 限流风险）。"""
    calls = patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    reruns = []
    monkeypatch.setattr(loop, "compare_prices", lambda *a, **kw: reruns.append(1) or make_result())

    responses = [call_tool(f"c{i}", "compare_prices", query=f"第{i}次") for i in range(4)]
    transport, client = make_agent(monkeypatch, *responses)
    run = loop.run_agent("x", client=client, max_steps=3)

    assert run.stopped_by == "max_steps"
    assert run.result is not None
    assert reruns == []                     # 没有重跑
    assert len(calls) == 3                  # 三次不同的调用各执行一次


def test_max_steps_without_any_result_falls_back(monkeypatch):
    """一次比价都没成功过 → 兜底真跑一次，否则用户什么都拿不到。"""
    patch_actions(monkeypatch, {"scan_risk": ok("scan_risk", RiskSummary(product="x", queries=[], items=[]))})
    fallback = []
    monkeypatch.setattr(loop, "compare_prices", lambda keyword, **kw: fallback.append(keyword) or make_result(keyword))

    responses = [call_tool(f"c{i}", "scan_risk", product="x") for i in range(4)]
    transport, client = make_agent(monkeypatch, *responses)
    run = loop.run_agent("QCY N20", client=client, max_steps=2)

    assert run.stopped_by == "max_steps"
    assert fallback == ["QCY N20"]
    assert run.result is not None


def test_no_tool_called_falls_back_to_pipeline(monkeypatch):
    """模型只聊不干活时，必须兜底出确定性结果，不能空手而归。"""
    patch_actions(monkeypatch, {})
    fallback = []
    monkeypatch.setattr(loop, "compare_prices", lambda keyword, **kw: fallback.append(keyword) or make_result(keyword))
    transport, client = make_agent(monkeypatch, answer("我觉得你说得对"))
    run = loop.run_agent("QCY N20", client=client)
    assert run.stopped_by == "no_tools"
    assert run.result is not None


def test_llm_unavailable_degrades_directly(monkeypatch):
    """无 key / 未注入 client 时不报错，直接走确定性管线（降级不阻塞的延续）。"""
    fallback = []
    monkeypatch.setattr(loop, "compare_prices", lambda keyword, **kw: fallback.append(keyword) or make_result(keyword))
    monkeypatch.setattr(loop, "_llm_client", lambda: None)
    run = loop.run_agent("QCY N20")
    assert run.stopped_by == "llm_unavailable"
    assert fallback == ["QCY N20"]
    assert run.result is not None


def test_llm_error_degrades(monkeypatch):
    class Boom:
        def chat(self, *a, **kw):
            raise LlmError("HTTP 500: 服务端错误")

    fallback = []
    monkeypatch.setattr(loop, "compare_prices", lambda keyword, **kw: fallback.append(keyword) or make_result(keyword))
    run = loop.run_agent("QCY N20", client=Boom())
    assert run.stopped_by == "llm_error"
    assert fallback == ["QCY N20"]


def test_external_call_budget_is_enforced(monkeypatch):
    """单请求外部调用总预算——超出即停，防止模型反复试探把配额烧掉。"""
    calls = patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    monkeypatch.setattr(loop, "compare_prices", lambda keyword, **kw: make_result(keyword))
    responses = [call_tool(f"c{i}", "compare_prices", query=f"第{i}次") for i in range(10)]
    transport, client = make_agent(monkeypatch, *responses)
    run = loop.run_agent("x", client=client, max_steps=10, max_tool_calls=3)
    assert len(calls) <= 3
    assert run.stopped_by in ("budget", "max_steps")


# ============================================================
# 7. 事件外发（SSE 与测试共用的同一套机制）
# ============================================================

def test_events_are_emitted_in_order(monkeypatch):
    patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result(), "比价摘要")})
    transport, client = make_agent(monkeypatch, call_tool("c1", "compare_prices", query="x"), answer("最低 ￥99"))
    events: list[dict] = []
    loop.run_agent("x", client=client, on_event=events.append)

    kinds = [event["type"] for event in events]
    assert kinds[0] == "step"
    assert "tool" in kinds
    tool_event = next(event for event in events if event["type"] == "tool")
    assert tool_event["name"] == "compare_prices"
    assert tool_event["ok"] is True
    assert tool_event["summary"] == "比价摘要"


def test_loop_never_emits_transport_level_event_names(monkeypatch):
    """本层只发**进度**事件。`final` / `error` 属于传输层（SSE）。

    曾经这里 emit 过一个 {"type":"final"}，与 SSE 层的真·final 撞名——
    前端收到第一个（没有 payload）就把连接关了，比价卡片永远出不来。
    这个坑只在真实端到端跑的时候才暴露，单测当时全绿。
    """
    patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    transport, client = make_agent(monkeypatch, call_tool("c1", "compare_prices", query="x"), answer("好"))
    events: list[dict] = []
    loop.run_agent("x", client=client, on_event=events.append)
    assert {event["type"] for event in events} <= {"step", "tool"}


def test_event_callback_failure_does_not_break_the_run(monkeypatch):
    """前端断连时回调可能抛异常——绝不能因此中断 Agent 运行。"""
    patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    transport, client = make_agent(monkeypatch, call_tool("c1", "compare_prices", query="x"), answer("好"))

    def boom(event):
        raise RuntimeError("客户端断了")

    run = loop.run_agent("x", client=client, on_event=boom)
    assert run.stopped_by == "finish"


# ============================================================
# 8. token 计量
# ============================================================

def test_usage_is_accumulated_across_rounds(monkeypatch):
    patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    second = dict(answer("好"))
    second["usage"] = {"prompt_tokens": 500, "completion_tokens": 10}
    transport, client = make_agent(monkeypatch, call_tool("c1", "compare_prices", query="x"), second)
    run = loop.run_agent("x", client=client)
    assert run.usage.prompt_tokens == 1500
    assert run.usage.completion_tokens == 30
    assert run.usage.cache_hit_tokens == 800


def test_usage_is_recorded_into_the_trace(monkeypatch):
    """算出来的真实 token 必须进 trace——否则计量白做。

    曾经 loop 只 count 了调用次数，把 `reply.usage` 丢在 run 对象里没人接。
    """
    from priceradar.trace import Trace

    patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    trace = Trace()
    transport, client = make_agent(monkeypatch, call_tool("c1", "compare_prices", query="x"), answer("好"))
    loop.run_agent("x", client=client, trace=trace)

    assert trace.tokens.calls == 2                 # 两轮 LLM 调用
    assert trace.tokens.prompt == 1000             # 只有第一轮带了 usage
    assert trace.tokens.cache_hit == 800
    assert trace.tokens.cache_miss == 200
    assert trace.tokens.cache_hit_rate == pytest.approx(0.8)
    assert trace.llm_cost_is_estimate is False     # 拿到了 token → 不是粗估


def test_trace_without_usage_falls_back_and_says_so(monkeypatch):
    from priceradar.trace import Trace

    patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    trace = Trace()
    transport, client = make_agent(monkeypatch, answer("好"))   # 响应里没有 usage
    loop.run_agent("x", client=client, trace=trace)
    assert trace.llm_cost_is_estimate is True
    assert "按次粗估" in trace.summary_line()


def test_tool_defaults_reach_the_tool_call(monkeypatch):
    """用户在界面上设的 jd_channel 必须真的到工具里。

    模型不会主动传 `jd_channel`（它不在工具描述里被强调），所以循环要把它补进
    模型**没给**的参数里——否则 Web 上「京东频道 = -1（关闭京东）」这种设置
    在 Agent 模式下是摆设，同样的查询在降级路径上却生效，行为还不一致。
    """
    calls = patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    transport, client = make_agent(monkeypatch, call_tool("c1", "compare_prices", query="x"), answer("好"))
    loop.run_agent("x", client=client, tool_defaults={"jd_channel": -1})

    assert calls[0][1]["jd_channel"] == -1


def test_tool_defaults_do_not_leak_into_tools_that_reject_them(monkeypatch):
    """回归：**只影响比价的设置不能把别的工具弄坏。**

    `tool_defaults={"jd_channel": …}` 曾经被**无差别**塞进每个工具调用，于是
    `scan_risk` 收到一个它不认的参数、被参数白名单拒绝，整个风险扫描报废——
    模型只好说「工具一直报参数错误，重试也没用」，然后放弃。

    ⚠️ 这个 bug **只有真的在浏览器里跑才看得到**：CLI 不传 `tool_defaults`，
    所以命令行一路正常；这里故意走**真实的** `run_action`（不桩它），
    才能覆盖到参数白名单那一层。
    """
    import priceradar.agent.actions as actions
    from priceradar.risk import RiskSummary

    monkeypatch.setattr(actions, "_llm_client", lambda: None)
    monkeypatch.setattr(
        actions,
        "scan_with_env",
        lambda product: RiskSummary(product=product, queries=["q"], items=[], note="ok", degraded=True),
    )

    _, client = make_agent(monkeypatch, call_tool("c1", "scan_risk", product="x"), answer("好"))
    run = loop.run_agent("x", client=client, tool_defaults={"jd_channel": -1})

    execution = run.steps[0].tool_calls[0]
    assert execution.ok is True, execution.error
    assert "jd_channel" not in execution.arguments


def test_accepted_params_knows_each_tool(monkeypatch):
    """默认值过滤的依据：`jd_channel` 属于比价，不属于风险扫描。"""
    from priceradar.agent.actions import accepted_params

    assert "jd_channel" in accepted_params("compare_prices")
    assert "jd_channel" not in accepted_params("scan_risk")
    assert "db_path" in accepted_params("check_price_history")   # 白名单内部参数
    assert accepted_params("不存在的工具") == frozenset()


def test_model_supplied_argument_wins_over_default(monkeypatch):
    """模型明确给了就听模型的——defaults 只补空缺。"""
    calls = patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    transport, client = make_agent(monkeypatch, call_tool("c1", "compare_prices", query="x", jd_channel=24), answer("好"))
    loop.run_agent("x", client=client, tool_defaults={"jd_channel": -1})
    assert calls[0][1]["jd_channel"] == 24


def test_trace_context_does_not_break_memo(monkeypatch):
    """注入的 trace 不能进 memo 键——否则同一个调用永远不命中缓存。"""
    calls = patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result())})
    transport, client = make_agent(
        monkeypatch,
        call_tool("c1", "compare_prices", query="x"),
        call_tool("c2", "compare_prices", query="x"),
        answer("好"),
    )
    loop.run_agent("x", client=client)
    assert len(calls) == 1


def test_char_budget_stops_the_loop(monkeypatch):
    """上下文预算兜底：按字符粗估（中文约 1 token/字的上界），超了就不再进下一轮。"""
    patch_actions(monkeypatch, {"compare_prices": ok("compare_prices", make_result(), "长" * 5000)})
    responses = [call_tool(f"c{i}", "compare_prices", query=f"第{i}次") for i in range(5)]
    transport, client = make_agent(monkeypatch, *responses)
    run = loop.run_agent("x", client=client, max_steps=5, char_budget=3000)
    assert run.stopped_by in ("budget", "max_steps")
    assert len(run.steps) < 5
