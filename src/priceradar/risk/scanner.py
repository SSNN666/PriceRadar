"""L2 风险检索管线 v1（PRD §5.8）。

流程：查询扩展（风险词模板）→ 多源检索（博查）→ 按 URL 去重 → 问题聚类（LLM）。

边界（与 PRD 一致）：
- 检索与去重是确定性的；问题聚类由 LLM 完成，失败降级为"原始检索清单"，绝不编造结论
- 跨查询词交叉验证体现在聚类规则：同一问题在多个查询词下重复出现 → 提及计数聚合
- 输出固定附覆盖声明（"不代表全部用户反馈"）
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from priceradar.agent.llm import DeepSeekClient, LlmError
from priceradar.sources.bocha import BochaClient

RISK_QUERY_TEMPLATES = ("{name} 避雷", "{name} 翻车", "{name} 踩坑", "{name} 问题")
COVERAGE_NOTE = "部分平台内容未被搜索引擎完整收录，检索结果不代表全部用户反馈"

_DEGRADED_NOTE = "未做问题归类（LLM 未启用/失败）；以下为检索结果原始清单"


@dataclass
class RiskItem:
    query: str
    title: str
    url: str
    site_name: str
    snippet: str


@dataclass
class RiskSummary:
    product: str
    queries: list[str]
    items: list[RiskItem]
    issues: list[dict] = field(default_factory=list)  # {"topic", "mention_count", "urls"}
    note: str = _DEGRADED_NOTE
    degraded: bool = True  # True = 未做问题聚类


def build_risk_queries(product_name: str) -> list[str]:
    """查询扩展：商品名 × 风险词模板（确定性）。"""
    return [template.format(name=product_name) for template in RISK_QUERY_TEMPLATES]


def scan_risk(product_name: str, client: BochaClient, *, per_query: int = 6) -> RiskSummary:
    """执行多源风险检索（按 URL 去重），返回原始清单（聚类由 cluster_issues 完成）。"""
    queries = build_risk_queries(product_name)
    items: list[RiskItem] = []
    seen: set[str] = set()
    for query in queries:
        for result in client.web_search(query, count=per_query):
            if result.url in seen:
                continue
            seen.add(result.url)
            items.append(
                RiskItem(query=query, title=result.title, url=result.url, site_name=result.site_name, snippet=result.snippet)
            )
    return RiskSummary(product=product_name, queries=queries, items=items)


def scan_with_env(product_name: str) -> RiskSummary:
    """便捷入口：从 .env 读取博查凭证执行扫描。"""
    return scan_risk(product_name, BochaClient.from_env())


class RiskIssue(BaseModel):
    topic: str = Field(description="风险主题，如：续航缩水 / 翻新机风险 / 尺码偏小")
    mention_count: int = Field(ge=1, description="跨查询词聚合后的提及数")
    urls: list[str] = Field(default_factory=list)


class RiskIssues(BaseModel):
    issues: list[RiskIssue]
    note: str = ""


_CLUSTER_SYSTEM = (
    "你是购物风险分析员。根据商品的风险检索结果（每条含查询词/标题/摘要/链接）提取风险主题。"
    "严格规则：① 只依据给定材料，不得编造；② 同一问题在不同查询词下重复出现要合并计数"
    "（这是跨查询词交叉验证）；③ 材料不足以判断就少列，宁缺毋滥；④ 最多 6 条，按提及数降序。"
    '只输出 JSON：{"issues":[{"topic":"...","mention_count":1,"urls":["..."]}],"note":""}'
)


def cluster_issues(
    summary: RiskSummary,
    client: DeepSeekClient | None = None,
    *,
    max_items: int = 40,
    usage_sink=None,
) -> RiskSummary:
    """LLM 把检索结果聚类为风险主题；未启用/失败时保持降级状态（不编造）。"""
    if client is None or not summary.items:
        return summary

    lines = [f"商品：{summary.product}"]
    for item in summary.items[:max_items]:
        snippet = f"｜{item.snippet[:60]}" if item.snippet else ""
        lines.append(f"[{item.query}] {item.title[:50]}{snippet}（{item.url}）")

    try:
        parsed = client.complete(_CLUSTER_SYSTEM, "\n".join(lines), json_schema=RiskIssues)
    except LlmError:
        return summary
    if not isinstance(parsed, RiskIssues) or not parsed.issues:
        return summary

    issues = sorted(
        (
            {"topic": issue.topic, "mention_count": issue.mention_count, "urls": issue.urls[:3]}
            for issue in parsed.issues
        ),
        key=lambda entry: -entry["mention_count"],
    )
    return RiskSummary(
        product=summary.product,
        queries=summary.queries,
        items=summary.items,
        issues=issues,
        note=parsed.note or "由 LLM 基于检索摘要聚合（跨查询词交叉验证）",
        degraded=False,
    )
