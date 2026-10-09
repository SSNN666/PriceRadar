"""`DeepSeekClient.chat()` 测试：工具调用协议 + usage 读取（transport 注入，全离线）。

`complete()` 的既有行为由 tests/test_llm.py 覆盖，这里只补**不回归**的那条。
"""

import json

import pytest

from priceradar.agent.llm import ChatReply, DeepSeekClient, LlmError, ToolCall, Usage
from tests.test_llm import ScriptedTransport, fake_response


def tool_call_response(*calls: tuple[str, str, dict], finish_reason: str = "tool_calls") -> dict:
    """构造一个 `tool_calls` 形状的响应（calls 为 (id, name, arguments) 三元组）。"""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                        }
                        for call_id, name, args in calls
                    ],
                },
                "finish_reason": finish_reason,
            }
        ]
    }


def with_usage(response: dict, **usage) -> dict:
    return {**response, "usage": usage}


# ---------- 请求构造 ----------

def test_chat_sends_messages_and_tools():
    transport = ScriptedTransport(fake_response("好的"))
    client = DeepSeekClient("k", transport=transport)
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {}}}]

    client.chat([{"role": "user", "content": "hi"}], tools=tools, tool_choice="auto")
    body = transport.calls[0]
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"
    assert body["model"] == "deepseek-chat"


def test_chat_omits_tools_when_not_given():
    """不带工具时不能传 `tools: null`——部分兼容实现会因此报错。"""
    transport = ScriptedTransport(fake_response("好的"))
    DeepSeekClient("k", transport=transport).chat([{"role": "user", "content": "hi"}])
    assert "tools" not in transport.calls[0]


# ---------- 响应解析 ----------

def test_chat_parses_content_only_reply():
    transport = ScriptedTransport(fake_response("最终答复"))
    reply = DeepSeekClient("k", transport=transport).chat([{"role": "user", "content": "hi"}])
    assert isinstance(reply, ChatReply)
    assert reply.content == "最终答复"
    assert reply.tool_calls == []


def test_chat_parses_tool_calls_with_parsed_arguments():
    transport = ScriptedTransport(tool_call_response(("call_1", "compare_prices", {"query": "耳机"})))
    reply = DeepSeekClient("k", transport=transport).chat([{"role": "user", "content": "hi"}])
    assert reply.content is None
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert isinstance(call, ToolCall)
    assert (call.id, call.name) == ("call_1", "compare_prices")
    assert call.arguments == {"query": "耳机"}      # 已解析成 dict（协议里是 JSON 字符串）
    assert reply.finish_reason == "tool_calls"


def test_chat_parses_parallel_tool_calls():
    """DeepSeek 支持一轮返回多个调用（实测：两个订单查询会合并到一个回合）。"""
    transport = ScriptedTransport(
        tool_call_response(("call_a", "compare_prices", {"query": "x"}), ("call_b", "scan_risk", {"product": "x"}))
    )
    reply = DeepSeekClient("k", transport=transport).chat([{"role": "user", "content": "hi"}])
    assert [call.name for call in reply.tool_calls] == ["compare_prices", "scan_risk"]
    assert [call.id for call in reply.tool_calls] == ["call_a", "call_b"]


def test_chat_survives_malformed_arguments_json():
    """模型偶尔给出坏 JSON——不能让整个循环崩掉；保留原文供排查。"""
    response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{不是JSON"}}],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    reply = DeepSeekClient("k", transport=ScriptedTransport(response)).chat([{"role": "user", "content": "hi"}])
    call = reply.tool_calls[0]
    assert call.arguments == {}          # 空参数 → 由工具的参数校验给出可读错误
    assert call.raw_arguments == "{不是JSON"


def test_chat_accepts_arguments_already_dict():
    """部分 OpenAI 兼容实现直接返回对象而不是 JSON 字符串——两种都要能吃。"""
    response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": {"a": 1}}}],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    reply = DeepSeekClient("k", transport=ScriptedTransport(response)).chat([{"role": "user", "content": "hi"}])
    assert reply.tool_calls[0].arguments == {"a": 1}


def test_chat_raises_on_malformed_response():
    transport = ScriptedTransport({"unexpected": True})
    with pytest.raises(LlmError, match="响应结构异常"):
        DeepSeekClient("k", transport=transport).chat([{"role": "user", "content": "hi"}])


# ---------- usage（token 计量的数据来源） ----------

def test_chat_reads_usage():
    response = with_usage(fake_response("x"), prompt_tokens=1200, completion_tokens=80, total_tokens=1280)
    reply = DeepSeekClient("k", transport=ScriptedTransport(response)).chat([{"role": "user", "content": "hi"}])
    assert reply.usage.prompt_tokens == 1200
    assert reply.usage.completion_tokens == 80
    assert reply.usage.total_tokens == 1280


def test_chat_reads_cache_hit_and_miss_tokens():
    """DeepSeek 把 prompt 分成缓存命中/未命中两档，单价差 30-50 倍——必须分开记。"""
    response = with_usage(
        fake_response("x"),
        prompt_tokens=1000,
        completion_tokens=10,
        prompt_cache_hit_tokens=900,
        prompt_cache_miss_tokens=100,
    )
    usage = DeepSeekClient("k", transport=ScriptedTransport(response)).chat([{"role": "user", "content": "hi"}]).usage
    assert usage.cache_hit_tokens == 900
    assert usage.cache_miss_tokens == 100
    assert usage.cache_hit_rate == pytest.approx(0.9)


def test_chat_usage_defaults_to_zero_when_absent():
    """没有 usage 字段时不能崩——如实记 0，不猜。"""
    reply = DeepSeekClient("k", transport=ScriptedTransport(fake_response("x"))).chat([{"role": "user", "content": "hi"}])
    assert reply.usage == Usage()
    assert reply.usage.cache_hit_rate == 0.0


def test_usage_cache_hit_rate_guards_against_zero_prompt():
    assert Usage(prompt_tokens=0, cache_hit_tokens=0).cache_hit_rate == 0.0


# ---------- complete() 不回归 + usage 出口 ----------

def test_complete_still_returns_plain_string():
    """`complete()` 的契约一字不改（既有节点与测试都依赖它）。"""
    transport = ScriptedTransport(fake_response("文本"))
    assert DeepSeekClient("k", transport=transport).complete("sys", "user") == "文本"


def test_complete_reports_usage_through_sink():
    response = with_usage(fake_response("文本"), prompt_tokens=50, completion_tokens=5)
    sink: list[Usage] = []
    DeepSeekClient("k", transport=ScriptedTransport(response)).complete("sys", "user", usage_sink=sink.append)
    assert sink and sink[0].prompt_tokens == 50


def test_complete_without_sink_is_unchanged():
    """不传 sink 时行为与从前完全一致（既有调用点零改动）。"""
    transport = ScriptedTransport(fake_response("文本"))
    assert DeepSeekClient("k", transport=transport).complete("sys", "user", temperature=0.1) == "文本"
    assert transport.calls[0]["temperature"] == 0.1


def test_complete_still_retries_on_schema_failure():
    """结构化输出校验失败重试一次的行为必须保住（既有护栏）。"""
    from pydantic import BaseModel

    class Shape(BaseModel):
        keyword: str

    transport = ScriptedTransport(fake_response('{"wrong": 1}'), fake_response('{"keyword": "耳机"}'))
    result = DeepSeekClient("k", transport=transport).complete("sys", "user", json_schema=Shape)
    assert result.keyword == "耳机"
    assert len(transport.calls) == 2
    assert "schema 校验" in transport.calls[1]["messages"][1]["content"]


def test_complete_schema_retry_does_not_crash_without_usage():
    """重试路径里没有 usage 时也不能崩（sink 只在有 usage 时调用）。"""
    from pydantic import BaseModel

    class Shape(BaseModel):
        keyword: str

    sink: list[Usage] = []
    transport = ScriptedTransport(fake_response('{"wrong": 1}'), fake_response('{"keyword": "x"}'))
    DeepSeekClient("k", transport=transport).complete("sys", "user", json_schema=Shape, usage_sink=sink.append)
    assert sink == []
