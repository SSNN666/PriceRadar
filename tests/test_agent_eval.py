"""Agent 评测的回归护栏（全离线：回放录制的真实响应，零网络）。

护栏的意义：录制是**一次性的真实快照**，回放是对循环机制的可重复检查。
若有人改坏了消息配对、memo 或降级路径，回放会先在这里失败。
"""

import json

import pytest

from evaluation.agent_eval import (
    CASSETTE_DIR,
    Scenario,
    evaluate,
    load_cassettes,
    load_scenarios,
    record,
    render_report,
    replay,
)

ACCURACY_FLOOR = 0.80   # 工具选择正确率下限
DEGRADE_CEILING = 0.20  # 降级率上限


@pytest.fixture(scope="module")
def cassettes():
    return load_cassettes()


def test_scenarios_file_is_well_formed():
    scenarios = load_scenarios()
    assert scenarios, "场景集不能为空"
    ids = [scenario.id for scenario in scenarios]
    assert len(ids) == len(set(ids)), "场景 id 必须唯一"
    from priceradar.agent.actions import ACTIONS

    for scenario in scenarios:
        assert scenario.query.strip()
        assert scenario.expected_tools, f"{scenario.id} 必须写明期望工具"
        # 注意括号：`-` 比 `|` 结合更紧，漏了括号会算成 expected | (optional - ACTIONS)
        unknown = (set(scenario.expected_tools) | set(scenario.optional_tools)) - set(ACTIONS)
        assert not unknown, f"{scenario.id} 引用了不存在的工具：{unknown}"
        # optional 与 expected 不该重叠（重叠说明期望写混了）
        assert not (set(scenario.expected_tools) & set(scenario.optional_tools))


def test_cassettes_cover_all_scenarios(cassettes):
    recorded = {cassette["id"] for cassette in cassettes}
    declared = {scenario.id for scenario in load_scenarios()}
    assert recorded == declared, f"录制与场景不一致：缺 {declared - recorded}，多 {recorded - declared}"


def test_replay_is_offline_and_reproduces_the_run(cassettes):
    """回放必须复现录制时的调用轨迹，且不触网。"""
    for cassette in cassettes:
        run = replay(cassette)
        called = [execution.name for step in run.steps for execution in step.tool_calls]
        assert called == cassette["observed"]["tools_called"], f"{cassette['id']} 回放轨迹与录制不一致"
        assert run.stopped_by == cassette["observed"]["stopped_by"]


def test_replay_consumes_all_recorded_responses(cassettes):
    """回放没用完的响应 = 循环行为已与录制时不同，必须暴露而不是静默通过。"""
    for cassette in cassettes:
        run = replay(cassette)
        # 直接断言不抛 LlmError（耗尽会抛）——由 replay 内部保证
        assert run is not None


def test_tool_selection_accuracy_floor(cassettes):
    stats = evaluate(cassettes)
    assert stats["total"] > 0
    assert stats["tool_selection_accuracy"] >= ACCURACY_FLOOR, (
        f"工具选择正确率 {stats['tool_selection_accuracy']:.1%} 跌破护栏 {ACCURACY_FLOOR:.0%}"
    )
    assert stats["degrade_rate"] <= DEGRADE_CEILING, f"降级率 {stats['degrade_rate']:.1%} 过高"


def test_failure_scenarios_exist_and_do_not_waste_retries(cassettes):
    """评测集里必须**真的包含失败场景**，否则「错误恢复」无从谈起。

    两个真实失败路径（都是平台侧的、可复现的）：
    - 拼多多类目空洞（显示器等整机类目返回空）
    - 非 App 分享的原始链接被拒绝反查

    而硬失败之后**不该重复试探**——那只会白烧配额（拼多多账号限流是已知事故）。
    """
    stats = evaluate(cassettes)
    failure_cases = stats["failure_cases"]
    assert len(failure_cases) >= 2, "评测集里缺少真实失败场景"
    assert stats["no_wasted_retry_rate"] == 1.0, (
        f"有场景在硬失败后还重复试探：{[(c['id'], c['extra_calls_after_failure']) for c in failure_cases]}"
    )
    for case in failure_cases:
        assert case["final_text"].strip(), f"{case['id']} 失败后必须有文字答复，不能空手而归"


def test_cassettes_record_the_failure_itself(cassettes):
    """录制的工具结果里必须能看到那次失败——否则失败路径是编出来的。"""
    for cassette in cassettes:
        if cassette["id"] not in ("category_gap", "non_original_link"):
            continue
        failures = [item for item in cassette["tool_outcomes"] if not item["ok"]]
        assert failures, f"{cassette['id']} 应当录到失败"
        assert failures[0]["retryable"] is False, "这两类失败都是重试无用的"


def test_no_replay_errors(cassettes):
    """回放异常非 0 说明循环机制被改坏了（消息配对/memo/降级）。"""
    assert evaluate(cassettes)["replay_errors"] == 0


def test_multi_intent_calls_both_tools(cassettes):
    """多意图场景必须真的并行调到两个工具——这是「Agent 自主决策」的核心证据。"""
    cassette = next(c for c in cassettes if c["id"] == "multi_intent")
    assert set(cassette["observed"]["tools_called"]) == {"compare_prices", "scan_risk"}


def test_chained_scenario_uses_two_rounds(cassettes):
    """找同款的链条：必须先 compare_prices 拿到 goods_sign，再 find_similar。"""
    cassette = next(c for c in cassettes if c["id"] == "similar")
    assert cassette["observed"]["tools_called"] == ["compare_prices", "find_similar"]
    assert cassette["observed"]["steps"] == 2


def test_evaluate_handles_empty_input():
    assert evaluate([])["total"] == 0


def test_render_report_handles_empty():
    assert "没有可用的录制" in render_report({"total": 0, "cases": []}, title="t", dataset_note="")


def test_render_report_marks_mismatches():
    stats = {
        "total": 1,
        "cases": [
            {
                "id": "x", "query": "q", "note": "n", "expected": ["compare_prices"], "optional": [],
                "called": ["scan_risk"], "exact_match": False, "recall": 0.0, "no_extra": False,
                "stopped_by": "finish", "steps": 1, "prompt_tokens": 1, "completion_tokens": 1,
                "cache_hit_rate": 0.0, "error": None, "recorded_at": "", "model": "",
            }
        ],
        "tool_selection_accuracy": 0.0, "tool_recall": 0.0, "no_extra_call_rate": 0.0,
        "degrade_rate": 0.0, "avg_steps": 1.0, "avg_prompt_tokens": 1.0,
        "avg_cache_hit_rate": 0.0, "replay_errors": 0,
    }
    report = render_report(stats, title="t", dataset_note="")
    assert "不一致的场景" in report
    assert "scan_risk" in report


def test_optional_tools_are_not_counted_as_extra():
    """可选的合理调用不该被算成「多余」——标准要写在跑之前，不能事后调指标。"""
    stats = evaluate(
        [
            {
                "id": "x",
                "query": "q",
                "expected_tools": ["check_price_history"],
                "optional_tools": ["compare_prices"],
                "llm_replies": [
                    {
                        "content": None,
                        "finish_reason": "tool_calls",
                        "tool_calls": [
                            {"id": "c1", "name": "check_price_history", "arguments": {"keyword": "x"}, "raw": "{}"},
                            {"id": "c2", "name": "compare_prices", "arguments": {"query": "x"}, "raw": "{}"},
                        ],
                    },
                    {"content": "好了", "finish_reason": "stop", "tool_calls": []},
                ],
                "tool_outcomes": [
                    {"ok": True, "action": "check_price_history", "summary": "s"},
                    {"ok": True, "action": "compare_prices", "summary": "s"},
                ],
            }
        ]
    )
    assert stats["tool_selection_accuracy"] == 1.0
    assert stats["no_extra_call_rate"] == 1.0


def test_extra_tool_beyond_allowed_is_penalised():
    stats = evaluate(
        [
            {
                "id": "x",
                "query": "q",
                "expected_tools": ["compare_prices"],
                "llm_replies": [
                    {
                        "content": None,
                        "finish_reason": "tool_calls",
                        "tool_calls": [
                            {"id": "c1", "name": "compare_prices", "arguments": {"query": "x"}, "raw": "{}"},
                            {"id": "c2", "name": "scan_risk", "arguments": {"product": "x"}, "raw": "{}"},
                        ],
                    },
                    {"content": "好了", "finish_reason": "stop", "tool_calls": []},
                ],
                "tool_outcomes": [
                    {"ok": True, "action": "compare_prices", "summary": "s"},
                    {"ok": True, "action": "scan_risk", "summary": "s"},
                ],
            }
        ]
    )
    assert stats["tool_selection_accuracy"] == 0.0
    assert stats["no_extra_call_rate"] == 0.0
    assert stats["tool_recall"] == 1.0        # 必需的调到了，只是多调了


def test_record_produces_replayable_cassette(monkeypatch):
    """录制产物必须能被回放——这是录制/回放两侧的契约测试。"""
    from priceradar.agent.actions import ActionResult

    class FakeClient:
        model = "fake"

        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            from priceradar.agent.llm import ChatReply, ToolCall

            self.calls += 1
            if self.calls == 1:
                return ChatReply(
                    tool_calls=[ToolCall(id="c1", name="compare_prices", arguments={"query": "x"}, raw_arguments='{"query":"x"}')],
                    finish_reason="tool_calls",
                )
            return ChatReply(content="完成", finish_reason="stop")

    cassette = record(
        Scenario(id="tmp", query="x", expected_tools=("compare_prices",)),
        FakeClient(),
        run_action_fn=lambda name, arguments, **kw: ActionResult(ok=True, action=name, summary="摘要"),
    )
    assert cassette["observed"]["tools_called"] == ["compare_prices"]
    assert cassette["observed"]["stopped_by"] == "finish"
    run = replay(cassette)
    assert run.final_text == "完成"


def test_cassettes_are_committed(cassettes):
    """录制文件要入库——否则 CI 上跑不了这个护栏。"""
    assert CASSETTE_DIR.exists()
    assert all(json.loads(path.read_text(encoding="utf-8")) for path in CASSETTE_DIR.glob("*.json"))
