"""埋点 / trace 测试。"""

import json

from priceradar.trace import API_UNIT_COST, Trace


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
