"""轻量埋点 / trace（PRD §8 埋点清单的起点）。

记录每次查询的：节点耗时（span）、API 调用次数、估算成本；
以 JSONL 追加落盘（data/traces，gitignored），为"单次查询成本 / P90 拆解"提供数据。

单价为估算值，随账单校准（拼多多/京东联盟在配额内免费；DeepSeek 按当前 token 量级粗估）。
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Iterator

API_UNIT_COST: dict[str, Decimal] = {
    "pdd": Decimal("0"),              # 配额内免费，仅计数
    "jd": Decimal("0"),
    "llm_deepseek": Decimal("0.002"),  # 估算：单次查询量级（元/次）
    "search_bocha": Decimal("0.05"),   # 博查按量价（元/次）
}


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

    @contextmanager
    def span(self, name: str, detail: str = "") -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.spans.append(NodeSpan(name=name, ms=int((time.perf_counter() - start) * 1000), detail=detail))

    def count(self, api: str, n: int = 1) -> None:
        self.api_calls[api] = self.api_calls.get(api, 0) + n

    @property
    def total_ms(self) -> int:
        return sum(span.ms for span in self.spans)

    @property
    def est_cost(self) -> Decimal:
        return sum((API_UNIT_COST.get(api, Decimal("0")) * calls for api, calls in self.api_calls.items()), Decimal("0"))

    def summary_line(self) -> str:
        calls = "｜".join(f"{api}×{n}" for api, n in sorted(self.api_calls.items())) or "无"
        cost_text = format(self.est_cost.normalize(), "f")  # 保留 3 位精度（如 0.004）
        return f"[trace {self.query_id}] 耗时 {self.total_ms} ms ｜ API 调用 {calls} ｜ 估算成本 ￥{cost_text}"

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["total_ms"] = self.total_ms
        payload["est_cost"] = str(self.est_cost)
        return payload

    def append_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.to_json(), ensure_ascii=False) + "\n")
