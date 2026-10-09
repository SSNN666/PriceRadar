"""轻量埋点 / trace（PRD §8 埋点清单的起点）。

记录每次查询的：节点耗时（span）、API 调用次数、**真实 token 计量与成本**；
以 JSONL 追加落盘（data/traces，gitignored），为"单次查询成本 / P90 拆解"提供数据。

## 成本为什么要分两块算

- 联盟/搜索类接口按**调用次数**计价（拼多多、京东在配额内免费；博查按次）→ `api_cost`
- LLM 按 **token** 计价，且**缓存命中与未命中单价差 30-50 倍**（远大于峰谷差价的 2 倍）
  → `llm_cost`

早期版本对 LLM 用「￥0.002/次」拍脑袋估算，2026-10-09 换成读响应里的 `usage`
按真实单价计算。**没有 token 数据时**（例如某些路径没埋到）才回退到按次粗估，
并在 `llm_cost_is_estimate` 里如实标注——宁可标注"这是估的"，也不假装精确。

## 关于单价

`LLM_PRICES` 抄自官方定价页，**官方保留调价权利**：
https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
实际花费请以账单为准；若计费档位变了，改这张表即可（其余代码不用动）。

峰谷差价（空闲时段为高峰的 50%）**没有计入**：它需要可靠的时区判断，而缓存命中
带来的 30-50 倍差异才是大头——把复杂度花在小头上不划算。真要算，接在
`estimate_llm_cost()` 的入口处乘一个系数即可。
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Callable, Iterator

# —— 按**调用次数**计价的接口（元/次）。LLM 见下面的 LLM_PRICES，不在这里 ——
API_UNIT_COST: dict[str, Decimal] = {
    "pdd": Decimal("0"),              # 配额内免费，仅计数
    "jd": Decimal("0"),
    "llm_deepseek": Decimal("0.002"),  # ⚠️ 仅作**回退估算**：拿不到 token 时才用
    "search_bocha": Decimal("0.05"),   # 博查按量价（元/次）
}

# —— LLM 单价（元/百万 token）——
# 键先按「输入·缓存命中 / 输入·缓存未命中 / 输出」三段。官方把模型分为 flash 与 pro
# 两档，deepseek-chat 未在定价页单独列出——这里按 flash 档计（偏乐观）；
# 若账单显示按 pro 档计费，把 deepseek-chat 换成 pro 那一行即可。
LLM_PRICES: dict[str, dict[str, Decimal]] = {
    "flash": {
        "cache_hit": Decimal("0.02"),
        "cache_miss": Decimal("1"),
        "output": Decimal("4"),
    },
    "pro": {
        "cache_hit": Decimal("0.15"),
        "cache_miss": Decimal("4.5"),
        "output": Decimal("13.5"),
    },
}
DEFAULT_PRICE_TIER = "flash"
PER_MILLION = Decimal("1000000")


@dataclass
class TokenAccounting:
    """一次查询内累计的 token 计量。缺字段记 0，**不猜**。"""

    prompt: int = 0
    completion: int = 0
    cache_hit: int = 0
    cache_miss: int = 0
    calls: int = 0

    def record(self, *, prompt: int, completion: int, cache_hit: int, cache_miss: int) -> None:
        self.prompt += prompt
        self.completion += completion
        self.cache_hit += cache_hit
        self.cache_miss += cache_miss
        self.calls += 1

    @property
    def is_empty(self) -> bool:
        return not (self.prompt or self.completion or self.cache_hit or self.cache_miss)

    @property
    def has_cache_split(self) -> bool:
        """是否拿到了缓存命中/未命中的分项（不是所有兼容实现都给）。"""
        return bool(self.cache_hit or self.cache_miss)

    @property
    def input_tokens(self) -> int:
        """输入侧 token。有分项时用「命中 + 未命中」，否则退回 prompt。

        两者通常相等，但分开记时更稳——某些实现的 `prompt_tokens` 与两项之和对不上。
        """
        if self.has_cache_split:
            return self.cache_hit + self.cache_miss
        return self.prompt

    @property
    def cache_hit_rate(self) -> float:
        total = self.input_tokens
        return (self.cache_hit / total) if total else 0.0


def estimate_llm_cost(tokens: TokenAccounting, tier: str = DEFAULT_PRICE_TIER) -> Decimal:
    """按真实 token 与单价算 LLM 花费（元）。

    没有缓存分项时，全部输入按**未命中**价计——宁可高估，不假装省了钱。
    """
    price = LLM_PRICES.get(tier, LLM_PRICES[DEFAULT_PRICE_TIER])
    hit = Decimal(tokens.cache_hit)
    miss = Decimal(tokens.cache_miss) if tokens.has_cache_split else Decimal(tokens.prompt)
    return (
        hit * price["cache_hit"] + miss * price["cache_miss"] + Decimal(tokens.completion) * price["output"]
    ) / PER_MILLION


@dataclass
class NodeSpan:
    name: str
    ms: int
    detail: str = ""


@dataclass
class Trace:
    query_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    spans: list[NodeSpan] = field(default_factory=list)
    api_calls: dict[str, int] = field(default_factory=dict)
    tokens: TokenAccounting = field(default_factory=TokenAccounting)

    @contextmanager
    def span(self, name: str, detail: str = "") -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.spans.append(NodeSpan(name=name, ms=int((time.perf_counter() - start) * 1000), detail=detail))

    def count(self, api: str, n: int = 1) -> None:
        self.api_calls[api] = self.api_calls.get(api, 0) + n

    def record_usage(self, *, prompt: int = 0, completion: int = 0, cache_hit: int = 0, cache_miss: int = 0) -> None:
        """记一次 LLM 调用的真实 token。由调用方从响应的 `usage` 里读出。"""
        self.tokens.record(prompt=prompt, completion=completion, cache_hit=cache_hit, cache_miss=cache_miss)

    @property
    def usage_sink(self) -> "Callable[[object], None]":
        """给 `DeepSeekClient.complete(usage_sink=...)` 用的适配器。

        工具内部的 LLM 调用（关键词提取 / 导语 / 风险聚类）也要计入——只统计循环
        自己的几轮会让成本严重少报：一次 `scan_risk` 的博查检索是 ￥0.2，
        比那几轮的 LLM 花费高两个数量级。
        """
        def sink(usage: object) -> None:
            self.record_usage(
                prompt=int(getattr(usage, "prompt_tokens", 0)),
                completion=int(getattr(usage, "completion_tokens", 0)),
                cache_hit=int(getattr(usage, "cache_hit_tokens", 0)),
                cache_miss=int(getattr(usage, "cache_miss_tokens", 0)),
            )

        return sink

    @property
    def total_ms(self) -> int:
        return sum(span.ms for span in self.spans)

    @property
    def api_cost(self) -> Decimal:
        """按调用次数计价的接口花费（不含 LLM）。"""
        return sum(
            (API_UNIT_COST.get(api, Decimal("0")) * calls for api, calls in self.api_calls.items() if api != "llm_deepseek"),
            Decimal("0"),
        )

    @property
    def llm_cost_is_estimate(self) -> bool:
        """LLM 花费是否只是按次粗估（拿不到 token 时）。"""
        return self.tokens.is_empty and bool(self.api_calls.get("llm_deepseek"))

    @property
    def llm_cost(self) -> Decimal:
        """LLM 花费：**有 token 就用真实单价算**，没有才回退到按次粗估。"""
        if not self.tokens.is_empty:
            return estimate_llm_cost(self.tokens)
        return API_UNIT_COST["llm_deepseek"] * self.api_calls.get("llm_deepseek", 0)

    @property
    def est_cost(self) -> Decimal:
        return self.api_cost + self.llm_cost

    def _cost_text(self, value: Decimal) -> str:
        """金额的最短**精确**表示。

        ⚠️ 不能按小数位截断：单次 LLM 调用只要 ￥0.00004 量级，
        截到 4 位小数会显示成 `0.0000`——把成本抹成零，比不显示更糟。
        `normalize()` 给的是精确的最短形式，正好合适。
        """
        normalized = value.normalize()
        text = format(normalized, "f")
        return text if len(text) <= 12 else format(normalized, ".6g")

    def summary_line(self) -> str:
        calls = "｜".join(f"{api}×{n}" for api, n in sorted(self.api_calls.items())) or "无"
        parts = [f"[trace {self.query_id}] 耗时 {self.total_ms} ms", f"API 调用 {calls}"]
        if not self.tokens.is_empty:
            parts.append(
                f"token 入 {self.tokens.input_tokens}/出 {self.tokens.completion}"
                f"（缓存命中 {self.tokens.cache_hit_rate:.0%}）"
            )
        cost = f"估算成本 ￥{self._cost_text(self.est_cost)}"
        if self.llm_cost_is_estimate:
            cost += "（其中 LLM 部分为按次粗估，未拿到 token）"
        parts.append(cost)
        return "｜".join(parts)

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["total_ms"] = self.total_ms
        # 用与 summary_line 相同的格式化：`str(Decimal)` 会带出多余尾零（0 → "0.000"），
        # 落盘的形状应当稳定
        payload["est_cost"] = self._cost_text(self.est_cost)
        payload["api_cost"] = self._cost_text(self.api_cost)
        payload["llm_cost"] = self._cost_text(self.llm_cost)
        payload["llm_cost_is_estimate"] = self.llm_cost_is_estimate
        payload["cache_hit_rate"] = round(self.tokens.cache_hit_rate, 4)
        return payload

    def append_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.to_json(), ensure_ascii=False) + "\n")
