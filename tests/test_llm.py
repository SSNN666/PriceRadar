"""LLM 客户端与 LLM 节点测试（transport 注入，全离线）。"""

from decimal import Decimal

import pytest

from priceradar.agent.llm import DeepSeekClient, LlmError
from priceradar.agent.llm_nodes import (
    ExtractedQuery,
    compose_summary,
    extract_keyword,
    fallback_keyword,
    fallback_summary,
)
from priceradar.agent.pipeline import ConfirmedItem, OfferGroup, PipelineResult, render
from priceradar.domain.models import Offer, Platform
from priceradar.engine.pricing import compute_price

D = Decimal


def fake_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


# ---------- 模型名（会被平台下架，故走配置而非硬编码）----------

def test_default_model_is_deepseek_chat():
    """2026-10-10 实测：`deepseek-chat` 仍可用，但**已不在 `/models` 列表里**。

    默认值暂时不动（换模型要重测 prompt 缓存命中率 + 重录 agent_eval）。
    这条测试钉住现状——真要换，得连着改这里与 llm.py 的注释。
    """
    assert DeepSeekClient("k").model == "deepseek-chat"


def test_model_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setattr(
        "priceradar.agent.llm.load_env",
        lambda *names: {"DEEPSEEK_API_KEY": "k", "DEEPSEEK_MODEL": "deepseek-v4-pro"},
    )
    assert DeepSeekClient.from_env().model == "deepseek-v4-pro"


def test_blank_model_env_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(
        "priceradar.agent.llm.load_env",
        lambda *names: {"DEEPSEEK_API_KEY": "k", "DEEPSEEK_MODEL": ""},
    )
    assert DeepSeekClient.from_env().model == "deepseek-chat"


class ScriptedTransport:
    """按脚本依次返回响应；记录全部请求 body。"""

    def __init__(self, *responses: dict) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, body: dict) -> dict:
        self.calls.append(body)
        return self.responses.pop(0)


def make_result() -> PipelineResult:
    offer_a = Offer(platform=Platform.PDD, sku_id="a", title="QCY N20 降噪耳机", list_price=D("99"))
    offer_b = Offer(platform=Platform.PDD, sku_id="b", title="QCY N20 主动降噪 耳机", list_price=D("105"))
    confirmed = [
        ConfirmedItem(offer=offer_a, reason="检索结果（源商品）"),
        ConfirmedItem(offer=offer_b, reason="型号一致（N20）"),
    ]
    breakdowns = [(offer, compute_price(offer)) for offer in (offer_a, offer_b)]
    groups = [
        OfferGroup(representative=offer_a, members=[offer_a], final_price=D("99"), reason="检索结果（源商品）"),
        OfferGroup(representative=offer_b, members=[offer_b], final_price=D("105"), reason="型号一致（N20）"),
    ]
    return PipelineResult(
        keyword="降噪耳机",
        pdd_count=20,
        jd_count=50,
        source=offer_a,
        confirmed=confirmed,
        groups=groups,
        similar_count=3,
        excluded_summary={},
        breakdowns=breakdowns,
        notes=["京东侧未找到确切同款（候选 50 条均未达 A 档）"],
    )


# ---------- LLM 客户端 ----------

def test_complete_json_schema_ok():
    transport = ScriptedTransport(fake_response('{"keyword": "降噪耳机"}'))
    client = DeepSeekClient("test-key", transport=transport)
    parsed = client.complete("sys", "帮我看看降噪耳机哪个划算", json_schema=ExtractedQuery)
    assert isinstance(parsed, ExtractedQuery)
    assert parsed.keyword == "降噪耳机"
    assert transport.calls[0]["response_format"] == {"type": "json_object"}


def test_complete_retries_after_invalid_json():
    transport = ScriptedTransport(fake_response("这不是JSON"), fake_response('{"keyword": "耳机"}'))
    client = DeepSeekClient("test-key", transport=transport)
    parsed = client.complete("sys", "q", json_schema=ExtractedQuery)
    assert parsed.keyword == "耳机"
    assert len(transport.calls) == 2
    assert "校验" in transport.calls[1]["messages"][1]["content"]


def test_complete_raises_after_repeated_invalid():
    transport = ScriptedTransport(fake_response("x"), fake_response("y"))
    client = DeepSeekClient("test-key", transport=transport)
    with pytest.raises(LlmError):
        client.complete("sys", "q", json_schema=ExtractedQuery)


# ---------- 查询理解节点 ----------

def test_extract_keyword_uses_llm():
    transport = ScriptedTransport(fake_response('{"keyword": "QCY N20 降噪耳机"}'))
    client = DeepSeekClient("k", transport=transport)
    keyword, method = extract_keyword("帮我看看QCY N20降噪耳机哪个平台划算", client)
    assert (keyword, method) == ("QCY N20 降噪耳机", "llm")


def test_extract_keyword_falls_back_to_rule_without_client():
    keyword, method = extract_keyword("帮我看看 QCY N20 降噪耳机哪个平台划算！", None)
    assert method == "rule"
    assert "QCY N20" in keyword
    assert "划算" not in keyword and "帮我" not in keyword


def test_extract_keyword_falls_back_when_llm_fails():
    def boom(body: dict) -> dict:
        raise LlmError("网络错误")

    client = DeepSeekClient("k", transport=boom)
    keyword, method = extract_keyword("QCY N20 降噪耳机", client)
    assert method == "rule"
    assert "QCY N20" in keyword


def test_fallback_keyword_keeps_original_when_all_filler():
    assert fallback_keyword("看看") == "看看"


# ---------- 报告导语节点 ----------

def test_fallback_summary_is_deterministic_and_mentions_notes():
    text, method = compose_summary(make_result(), None)
    assert method == "rule"
    assert "￥99" in text
    assert "未找到确切同款" in text


def test_compose_summary_uses_llm():
    transport = ScriptedTransport(fake_response("已为你找到 2 个同款报价，最低到手价 ￥99。京东侧暂未命中。"))
    client = DeepSeekClient("k", transport=transport)
    text, method = compose_summary(make_result(), client)
    assert method == "llm"
    assert "￥99" in text
    # 事实清单必须完整进入 prompt（含降级说明）
    user_prompt = transport.calls[0]["messages"][1]["content"]
    assert "未找到确切同款" in user_prompt


def test_render_shows_summary_when_present():
    result = make_result()
    result.summary = "测试导语：最低 ￥99。"
    assert "测试导语：最低 ￥99。" in render(result)
