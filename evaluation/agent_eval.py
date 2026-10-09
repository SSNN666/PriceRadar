"""Agent 行为评测：真实响应**录制** → 离线**回放** → 指标。

## 为什么必须录制，不能只写脚本

`tests/test_agent_loop.py` 用的 `ScriptedTransport` 是**自己写死的脚本**——
断言「工具选择正确率 100%」等于断言「我写的脚本等于我写的期望」，毫无信息量，
面试官一问就穿。

所以这里分两层：
- **机制正确性** → `tests/test_agent_loop.py`（脚本化，测轮次上限/memo/补齐/降级）
- **模型行为** → 本模块：跑真实 DeepSeek 把响应**录下来**，之后离线回放打分

## 关于「真数字」的诚实说明

回放出来的指标是**录制那一刻**的真实模型行为，不是当前模型的实时表现。
报告里会写明录制时间与模型名；要刷新就重跑 `--live`。

录制同时也会录下**工具结果**，因此回放全程零网络、零 API 花费。

用法：
    python -m evaluation.agent_eval --live      # 真实跑一遍并录制（会花钱与配额）
    python -m evaluation.agent_eval             # 用已有录制离线回放并出报告
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from priceradar.agent import loop
from priceradar.agent.actions import ActionResult
from priceradar.agent.llm import ChatReply, LlmError, ToolCall, Usage
from priceradar.trace import Trace

HERE = Path(__file__).parent
SCENARIOS = HERE / "agent_scenarios.jsonl"
CASSETTE_DIR = HERE / "cassettes"
RESULTS = HERE / "results_agent.md"


@dataclass(frozen=True)
class Scenario:
    id: str
    query: str
    expected_tools: tuple[str, ...]
    # 部分查询本来就存在**合理但非必需**的调用（例如问时机时顺手比个价）。
    # 把它们显式列出来，而不是事后把指标往下调——**标准要写在跑之前**。
    optional_tools: tuple[str, ...] = ()
    note: str = ""

    @property
    def allowed_tools(self) -> set[str]:
        return set(self.expected_tools) | set(self.optional_tools)


def load_scenarios(path: Path = SCENARIOS) -> list[Scenario]:
    if not path.exists():
        return []
    scenarios = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        record = json.loads(line)
        scenarios.append(
            Scenario(
                id=record["id"],
                query=record["query"],
                expected_tools=tuple(record["expected_tools"]),
                optional_tools=tuple(record.get("optional_tools", [])),
                note=record.get("note", ""),
            )
        )
    return scenarios


# ============================================================
# 录制
# ============================================================

class RecordingClient:
    """包住真实客户端，把每一次 chat 的响应录下来。"""

    def __init__(self, inner):
        self.inner = inner
        self.replies: list[dict] = []

    def chat(self, messages, **kwargs) -> ChatReply:
        reply = self.inner.chat(messages, **kwargs)
        self.replies.append(
            {
                "content": reply.content,
                "finish_reason": reply.finish_reason,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments, "raw": call.raw_arguments}
                    for call in reply.tool_calls
                ],
                "usage": {
                    "prompt_tokens": reply.usage.prompt_tokens,
                    "completion_tokens": reply.usage.completion_tokens,
                    "cache_hit_tokens": reply.usage.cache_hit_tokens,
                    "cache_miss_tokens": reply.usage.cache_miss_tokens,
                },
            }
        )
        return reply


def record(scenario: Scenario, client, *, max_steps: int = loop.MAX_STEPS, run_action_fn=None) -> dict:
    """跑一次真实 Agent 并录制成 cassette。

    工具结果一并录下——回放时才能全程零网络。
    """
    from priceradar.agent.actions import run_action as default_run_action

    real_run_action = run_action_fn or default_run_action
    recording_client = RecordingClient(client)
    outcomes: list[dict] = []

    def recording_run_action(name, arguments, **kwargs):
        outcome = real_run_action(name, arguments, **kwargs)
        outcomes.append(
            {
                "ok": outcome.ok,
                "action": outcome.action,
                "summary": outcome.summary,
                "error": outcome.error,
                "retryable": outcome.retryable,
                "hint": outcome.hint,
            }
        )
        return outcome

    with patch.object(loop, "run_action", recording_run_action):
        run = loop.run_agent(scenario.query, client=recording_client, max_steps=max_steps)

    return {
        "id": scenario.id,
        "query": scenario.query,
        "expected_tools": list(scenario.expected_tools),
        "optional_tools": list(scenario.optional_tools),
        "note": scenario.note,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "model": getattr(client, "model", "unknown"),
        "llm_replies": recording_client.replies,
        "tool_outcomes": outcomes,
        "observed": {
            "stopped_by": run.stopped_by,
            "final_text": run.final_text,
            "steps": len(run.steps),
            "tools_called": [execution.name for step in run.steps for execution in step.tool_calls],
        },
    }


def record_all(scenarios: list[Scenario], *, client=None) -> list[dict]:
    from priceradar.agent.actions import _llm_client

    client = client or _llm_client()
    if client is None:
        raise SystemExit("未配置 DEEPSEEK_API_KEY，无法录制真实响应")
    CASSETTE_DIR.mkdir(parents=True, exist_ok=True)
    recorded = []
    for scenario in scenarios:
        cassette = record(scenario, client)
        (CASSETTE_DIR / f"{scenario.id}.json").write_text(
            json.dumps(cassette, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        recorded.append(cassette)
        print(f"  录制 {scenario.id}：调用 {cassette['observed']['tools_called']} → {cassette['observed']['stopped_by']}")
    return recorded


# ============================================================
# 回放
# ============================================================

class ReplayDriftError(RuntimeError):
    """回放时循环行为与录制时不一致（响应没用完 / 结束原因变了）。"""


class ReplayClient:
    """按录制的顺序返回响应。"""

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.model = "replay"

    def chat(self, messages, **kwargs) -> ChatReply:
        if not self.replies:
            raise LlmError("回放耗尽：录制的响应已用完（循环行为与录制时不一致）")
        item = self.replies.pop(0)
        usage = item.get("usage") or {}
        return ChatReply(
            content=item.get("content"),
            finish_reason=item.get("finish_reason", ""),
            tool_calls=[
                ToolCall(id=call["id"], name=call["name"], arguments=call["arguments"], raw_arguments=call.get("raw", ""))
                for call in item.get("tool_calls", [])
            ],
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                cache_hit_tokens=usage.get("cache_hit_tokens", 0),
                cache_miss_tokens=usage.get("cache_miss_tokens", 0),
            ),
        )


def replay(cassette: dict, *, max_steps: int = loop.MAX_STEPS):
    """离线重跑一次循环，喂录制的 LLM 响应与工具结果。零网络。"""
    outcomes = list(cassette.get("tool_outcomes", []))

    def replayed_run_action(name, arguments, **kwargs):
        if not outcomes:
            return ActionResult(ok=False, action=name, error="回放耗尽：录制的工具结果已用完", hint="")
        return ActionResult(**outcomes.pop(0))

    stub_result = _stub_pipeline_result(cassette["query"])
    client = ReplayClient(cassette.get("llm_replies", []))
    with patch.object(loop, "run_action", replayed_run_action), patch.object(loop, "compare_prices", lambda *a, **kw: stub_result):
        run = loop.run_agent(cassette["query"], client=client, max_steps=max_steps)

    # ⚠️ **必须在这里显式检查漂移**：循环把 `chat` 的任何异常都降级处理（它的对外
    # 承诺是「永不抛异常」），所以「回放耗尽」会被吞成一次干干净净的 llm_error 降级，
    # 指标看起来完全正常——回放护栏就成了摆设。
    if client.replies:
        raise ReplayDriftError(f"还有 {len(client.replies)} 条录制响应未被消费——循环行为已与录制时不同")
    if outcomes:
        raise ReplayDriftError(f"还有 {len(outcomes)} 条录制的工具结果未被消费")
    recorded_stop = cassette.get("observed", {}).get("stopped_by")
    if recorded_stop and run.stopped_by != recorded_stop:
        raise ReplayDriftError(f"结束原因变了：录制时是 {recorded_stop}，回放得到 {run.stopped_by}")
    return run


def _stub_pipeline_result(keyword: str):
    """回放时的确定性结果占位（回放不重跑真实管线，因此不需要真数据）。"""
    from decimal import Decimal

    from priceradar.agent.pipeline import ConfirmedItem, OfferGroup, PipelineResult
    from priceradar.domain.models import Offer, Platform
    from priceradar.engine.pricing import compute_price

    offer = Offer(platform=Platform.PDD, sku_id="replay", title="回放占位商品", list_price=Decimal("99"))
    return PipelineResult(
        keyword=keyword,
        pdd_count=0,
        jd_count=0,
        source=offer,
        confirmed=[ConfirmedItem(offer=offer, reason="回放占位")],
        groups=[OfferGroup(representative=offer, members=[offer], final_price=offer.list_price, reason="回放占位")],
        similar_count=0,
        excluded_summary={},
        breakdowns=[(offer, compute_price(offer))],
    )


def load_cassettes(directory: Path = CASSETTE_DIR) -> list[dict]:
    if not directory.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]


# ============================================================
# 指标
# ============================================================

def evaluate(cassettes: list[dict]) -> dict:
    """跑回放并算指标。返回纯 dict（便于测试与复用渲染器）。"""
    per_case = []
    for cassette in cassettes:
        expected = tuple(cassette.get("expected_tools", []))
        optional = tuple(cassette.get("optional_tools", []))
        allowed = set(expected) | set(optional)
        try:
            run = replay(cassette)
            called = [execution.name for step in run.steps for execution in step.tool_calls]
            stopped_by = run.stopped_by
            steps = len(run.steps)
            usage = run.usage
            error = None
        except Exception as exc:  # 回放本身出问题（说明循环被改坏了）要如实暴露，不要静默
            called, stopped_by, steps, error = [], "replay_error", 0, f"{type(exc).__name__}: {exc}"
            usage = Usage()
        called_set = set(called)
        # 硬失败 = 重试也没用的失败（业务/参数类）。这类失败之后**不该再重复试探**
        # ——那只会白烧配额（拼多多账号限流是已知事故）。
        outcomes = cassette.get("tool_outcomes", [])
        hard_failures = [item for item in outcomes if not item.get("ok") and not item.get("retryable")]
        per_case.append(
            {
                "id": cassette["id"],
                "query": cassette["query"],
                "note": cassette.get("note", ""),
                "expected": list(expected),
                "optional": list(optional),
                "called": called,
                "hard_failures": len(hard_failures),
                "extra_calls_after_failure": max(0, len(called) - len(expected)) if hard_failures else 0,
                "final_text": cassette.get("observed", {}).get("final_text", ""),
                # 完全一致 = 必需的工具都调了，且没有调允许范围之外的
                "exact_match": set(expected) <= called_set <= allowed,
                "recall": (len(called_set & set(expected)) / len(expected)) if expected else 1.0,
                "no_extra": called_set <= allowed,
                "stopped_by": stopped_by,
                "steps": steps,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cache_hit_rate": usage.cache_hit_rate,
                "error": error,
                "recorded_at": cassette.get("recorded_at", ""),
                "model": cassette.get("model", ""),
            }
        )

    total = len(per_case)
    if not total:
        return {"total": 0, "cases": []}
    failure_cases = [case for case in per_case if case["hard_failures"]]
    return {
        "total": total,
        "cases": per_case,
        "failure_cases": failure_cases,
        "tool_selection_accuracy": sum(c["exact_match"] for c in per_case) / total,
        "tool_recall": sum(c["recall"] for c in per_case) / total,
        "no_extra_call_rate": sum(c["no_extra"] for c in per_case) / total,
        "degrade_rate": sum(c["stopped_by"] != "finish" for c in per_case) / total,
        "avg_steps": sum(c["steps"] for c in per_case) / total,
        "avg_prompt_tokens": sum(c["prompt_tokens"] for c in per_case) / total,
        "avg_cache_hit_rate": sum(c["cache_hit_rate"] for c in per_case) / total,
        "replay_errors": sum(c["error"] is not None for c in per_case),
        # 「硬失败之后不重复试探」——直接对应「别白烧配额」那条设计
        "no_wasted_retry_rate": (
            sum(case["extra_calls_after_failure"] == 0 for case in failure_cases) / len(failure_cases)
            if failure_cases
            else 1.0
        ),
    }


def render_report(stats: dict, *, title: str, dataset_note: str) -> str:
    if not stats["total"]:
        return f"# {title}\n\n没有可用的录制。先跑 `python -m evaluation.agent_eval --live`。\n"

    lines = [
        f"# {title}",
        "",
        dataset_note,
        "",
        "## 总体指标",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---|---|",
        f"| 工具选择正确率 | {stats['tool_selection_accuracy']:.1%} | 调用集合与期望**完全一致**的比例 |",
        f"| 工具召回率 | {stats['tool_recall']:.1%} | 期望的工具里被调到的比例（宽松） |",
        f"| 无多余调用率 | {stats['no_extra_call_rate']:.1%} | 没有调期望之外的工具 |",
        f"| 降级率 | {stats['degrade_rate']:.1%} | 未走到 `finish`（触顶/LLM 失败/未调工具） |",
        f"| 平均轮次 | {stats['avg_steps']:.2f} | 发生过工具调用的轮数 |",
        f"| 平均提示 token | {stats['avg_prompt_tokens']:.0f} | 多轮累计 |",
        f"| 平均缓存命中率 | {stats['avg_cache_hit_rate']:.1%} | 命中与未命中单价差 30-50 倍 |",
        f"| 回放异常 | {stats['replay_errors']} | 非 0 说明循环行为与录制时不一致 |",
        f"| 硬失败后不重复试探 | {stats.get('no_wasted_retry_rate', 1.0):.1%} | "
        f"{len(stats.get('failure_cases', []))} 个真实失败场景里，没有多余重试的比例 |",
        "",
        "## 逐条",
        "",
        "| 场景 | 期望工具 | 实际调用 | 完全一致 | 轮次 | 结束原因 |",
        "|---|---|---|---|---|---|",
    ]
    for case in stats["cases"]:
        mark = "✅" if case["exact_match"] else "❌"
        lines.append(
            f"| {case['id']} | {'、'.join(case['expected'])} | {'、'.join(case['called']) or '—'} "
            f"| {mark} | {case['steps']} | {case['stopped_by']} |"
        )
    failure_cases = stats.get("failure_cases", [])
    if failure_cases:
        lines += [
            "",
            "## 失败路径（这些场景里工具**真的失败了**，看模型怎么处理）",
            "",
            "| 场景 | 硬失败次数 | 多余重试 | 模型最终怎么说的 |",
            "|---|---|---|---|",
        ]
        for case in failure_cases:
            said = (case["final_text"] or "").replace("\n", " ").strip()[:90]
            lines.append(
                f"| {case['id']} | {case['hard_failures']} | {case['extra_calls_after_failure']} | {said}… |"
            )

    mismatches = [case for case in stats["cases"] if not case["exact_match"]]
    if mismatches:
        lines += ["", "## 不一致的场景（值得逐条看）", ""]
        for case in mismatches:
            lines.append(f"- **{case['id']}**（{case['note']}）")
            lines.append(f"  - 查询：{case['query']}")
            lines.append(f"  - 期望：{'、'.join(case['expected'])} ｜ 实际：{'、'.join(case['called']) or '（无）'}")
            if case["error"]:
                lines.append(f"  - ⚠️ 回放异常：{case['error']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Agent 行为评测（录制 / 回放）")
    parser.add_argument("--live", action="store_true", help="跑真实模型并录制（会花钱与配额）")
    parser.add_argument(
        "--only", nargs="*", metavar="ID",
        help="只录制这些场景（省花费；不加则录全部）。已有录制会被覆盖。",
    )
    args = parser.parse_args(argv)

    scenarios = load_scenarios()
    if not scenarios:
        raise SystemExit(f"没有场景集：{SCENARIOS}")

    if args.live:
        targets = scenarios
        if args.only:
            wanted = set(args.only)
            unknown = wanted - {scenario.id for scenario in scenarios}
            if unknown:
                raise SystemExit(f"场景集里没有：{'、'.join(sorted(unknown))}")
            # 只重录指定场景：既省花费，也避免把已核验过的录制一起换掉
            targets = [scenario for scenario in scenarios if scenario.id in wanted]
        print(f"真实录制 {len(targets)} 个场景……")
        record_all(targets)

    cassettes = load_cassettes()
    if not cassettes:
        raise SystemExit("没有录制可回放。先跑 --live。")

    stats = evaluate(cassettes)
    # 别只报第一条的时间：用 `--only` 分批补录过之后，各条 cassette 的录制时间不同，
    # 写成同一个时间点等于给数据编了个不存在的出处
    stamps = sorted({c.get("recorded_at", "") for c in cassettes if c.get("recorded_at")})
    models = sorted({c.get("model", "") for c in cassettes if c.get("model")})
    stamp = f"{stamps[0]} ~ {stamps[-1]}（分批录制）" if len(stamps) > 1 else (stamps[0] if stamps else "未记录")
    model = "、".join(models) if models else "未知"
    note = (
        f"> ⚠️ **这些指标来自真实录制，不是当前模型的实时表现。**\n"
        f"> 录制于 {stamp}，模型 `{model}`，共 {stats['total']} 个场景。\n"
        f"> 要刷新就重跑 `python -m evaluation.agent_eval --live`。\n"
        f"> 期望的工具集合由人工撰写（`evaluation/agent_scenarios.jsonl`），可随时改。\n"
        f"> 回放全程零网络、零 API 花费。"
    )
    RESULTS.write_text(render_report(stats, title="Agent 行为评测", dataset_note=note), encoding="utf-8")
    print(render_report(stats, title="Agent 行为评测", dataset_note=note))
    print(f"\n报告已写入 {RESULTS}")


if __name__ == "__main__":
    main()
