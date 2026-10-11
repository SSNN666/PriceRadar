"""埋点 / trace 测试。"""

import json
from decimal import Decimal

import pytest

from priceradar.trace import (
    API_UNIT_COST,
    LLM_PRICES,
    TokenAccounting,
    Trace,
    estimate_llm_cost,
)


def test_span_records_duration_and_detail():
    trace = Trace()
    with trace.span("step", "detail-x"):
        pass
    assert trace.spans[0].name == "step"
    assert trace.spans[0].detail == "detail-x"
    assert trace.spans[0].ms >= 0
    assert trace.total_ms == trace.spans[0].ms


def test_cost_estimation_accumulates():
    trace = Trace()
    trace.count("search_bocha", 2)
    trace.count("llm_deepseek", 3)
    trace.count("pdd")  # 免费但计数
    expected = API_UNIT_COST["search_bocha"] * 2 + API_UNIT_COST["llm_deepseek"] * 3
    assert trace.est_cost == expected
    assert "search_bocha×2" in trace.summary_line()


def test_append_jsonl(tmp_path):
    trace = Trace()
    trace.count("pdd")
    with trace.span("x"):
        pass
    path = tmp_path / "nested" / "traces.jsonl"
    trace.append_jsonl(path)
    trace.append_jsonl(path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["api_calls"]["pdd"] == 1
    assert "spans" in payload and payload["est_cost"] == "0"
    assert payload["query_id"] == trace.query_id


# ---------- 真实 token 计量 ----------

def test_record_usage_accumulates_across_calls():
    trace = Trace()
    trace.record_usage(prompt=1000, completion=50, cache_hit=800, cache_miss=200)
    trace.record_usage(prompt=500, completion=10, cache_hit=400, cache_miss=100)
    assert trace.tokens.prompt == 1500
    assert trace.tokens.completion == 60
    assert trace.tokens.cache_hit == 1200
    assert trace.tokens.calls == 2


def test_cache_hit_rate_uses_the_split_not_prompt_total():
    tokens = TokenAccounting(prompt=1000, cache_hit=900, cache_miss=100)
    assert tokens.cache_hit_rate == pytest.approx(0.9)
    assert tokens.input_tokens == 1000


def test_cache_hit_rate_guards_against_zero():
    assert TokenAccounting().cache_hit_rate == 0.0


def test_llm_cost_uses_real_token_pricing():
    """缓存命中与未命中单价差 30-50 倍——必须分开计价，不能只用一个总数。"""
    trace = Trace()
    trace.record_usage(prompt=1_000_000, completion=0, cache_hit=1_000_000, cache_miss=0)
    all_hit = trace.llm_cost

    other = Trace()
    other.record_usage(prompt=1_000_000, completion=0, cache_hit=0, cache_miss=1_000_000)
    all_miss = other.llm_cost

    assert all_miss > all_hit * 10, "全未命中必须显著贵于全命中"
    assert all_hit == LLM_PRICES["flash"]["cache_hit"]


def test_llm_cost_charges_all_input_as_miss_without_a_split():
    """拿不到缓存分项时，全按未命中价计——宁可高估，不假装省了钱。"""
    tokens = TokenAccounting(prompt=1_000_000)
    assert estimate_llm_cost(tokens) == LLM_PRICES["flash"]["cache_miss"]


def test_llm_cost_falls_back_to_per_call_when_no_tokens():
    """没有 token 数据时退回按次粗估，并**如实标注**这是估的。"""
    trace = Trace()
    trace.count("llm_deepseek", 3)
    assert trace.llm_cost == API_UNIT_COST["llm_deepseek"] * 3
    assert trace.llm_cost_is_estimate is True


def test_token_backed_cost_is_not_flagged_as_estimate():
    trace = Trace()
    trace.count("llm_deepseek")
    trace.record_usage(prompt=1000, completion=10, cache_hit=900, cache_miss=100)
    assert trace.llm_cost_is_estimate is False
    assert "按次粗估" not in trace.summary_line()
    assert "缓存命中 90%" in trace.summary_line()


def test_est_cost_is_api_plus_llm():
    trace = Trace()
    trace.count("search_bocha", 2)
    trace.record_usage(prompt=1_000_000, completion=0, cache_hit=0, cache_miss=1_000_000)
    assert trace.est_cost == trace.api_cost + trace.llm_cost
    assert trace.api_cost == API_UNIT_COST["search_bocha"] * 2


def test_summary_line_mentions_token_split():
    trace = Trace()
    trace.record_usage(prompt=1200, completion=80, cache_hit=1000, cache_miss=200)
    line = trace.summary_line()
    assert "token" in line and "缓存命中" in line and "￥" in line


def test_to_json_has_stable_cost_shape():
    trace = Trace()
    trace.count("pdd")
    payload = trace.to_json()
    for key in ("est_cost", "api_cost", "llm_cost", "cache_hit_rate", "llm_cost_is_estimate", "tokens"):
        assert key in payload
    assert payload["est_cost"] == "0"          # 不多带尾零
    assert payload["cache_hit_rate"] == 0.0


def test_to_json_serialises_nested_token_accounting():
    trace = Trace()
    trace.record_usage(prompt=100, completion=5, cache_hit=80, cache_miss=20)
    payload = json.loads(json.dumps(trace.to_json(), ensure_ascii=False))
    assert payload["tokens"]["cache_hit"] == 80
    assert payload["tokens"]["completion"] == 5
    # 一次调用只要 ￥0.0000416 量级——按小数位截断会显示成 0.0000，把成本抹掉
    assert payload["est_cost"] == "0.0000416"


def test_small_costs_are_not_flattened_to_zero():
    """单次 LLM 调用的成本极小，但**不能**因此显示成 0。"""
    trace = Trace()
    trace.record_usage(prompt=1263, completion=208, cache_hit=1024, cache_miss=239)
    text = trace.summary_line()
    assert "￥0.000" in text or "￥0.00" in text
    assert "估算成本 ￥0" not in text.replace("￥0.000", "").replace("￥0.00", "")


def test_prices_are_decimal_not_float():
    """金额一律 Decimal——浮点在成本累加时会飘。"""
    for tier in LLM_PRICES.values():
        for price in tier.values():
            assert isinstance(price, Decimal)
