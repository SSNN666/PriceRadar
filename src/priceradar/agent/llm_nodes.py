"""LLM 节点（PRD §5.2 中的两个 LLM 边界）：查询理解 与 报告导语。

共同约束：
- LLM 只做"理解"与"表达"，不产生任何数字结论（数字全部来自确定性管线）
- 每个节点都有确定性兜底：LLM 不可用 / 失败 / 超时 → 自动降级，不阻塞主流程
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from priceradar.agent.llm import DeepSeekClient, LlmError
from priceradar.agent.pipeline import PipelineResult
from priceradar.formatting import fmt_amount
from priceradar.report import PLATFORM_LABELS
from priceradar.sources.links import extract_pdd_link


class ExtractedQuery(BaseModel):
    keyword: str = Field(description="用于商品检索的核心关键词")


_EXTRACT_SYSTEM = (
    "你是电商比价助手的信息抽取器。从用户输入中提取用于商品检索的核心关键词，"
    '去掉"帮我看看/哪个平台买/划算/值得买吗"等口语成分，保留品牌、型号、品类。'
    '只输出 JSON，格式：{"keyword": "..."}'
)

_FILLERS = (
    "帮我看看", "帮我查查", "帮我查", "帮我比价", "帮我比", "对比一下", "对比下", "比一下",
    "看一下", "看看", "查一下", "哪个平台买", "哪个平台", "哪家", "哪个划算", "划算", "值得买吗",
    "值得买么", "该不该买", "哪里买", "怎么买", "多少钱", "贵不贵", "求推荐", "推荐一下", "推荐",
)


def fallback_keyword(text: str) -> str:
    """规则兜底：剥离常见口语填充词。"""
    result = text.strip()
    for filler in _FILLERS:
        result = result.replace(filler, " ")
    result = re.sub(r"[?？!！,，。.;；:：]", " ", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result or text.strip()


def extract_keyword(
    text: str, client: DeepSeekClient | None = None, *, usage_sink=None
) -> tuple[str, str]:
    """从自然语言中提取检索关键词。返回 (keyword, method)，method ∈ {"link", "llm", "rule"}。

    **链接短路优先**：用户粘的是拼多多商品链接时，**原样透传**给检索层
    （平台支持用链接精确反查商品）。必须走这条短路——否则链接会被当成自然语言
    交给 LLM 提取"关键词"，URL 直接被毁掉，反查也就无从谈起。
    """
    link = extract_pdd_link(text)
    if link:
        return link, "link"
    if client is not None:
        try:
            parsed = client.complete(_EXTRACT_SYSTEM, text, json_schema=ExtractedQuery, usage_sink=usage_sink)
            keyword = parsed.keyword.strip() if isinstance(parsed, ExtractedQuery) else ""
            if keyword:
                return keyword, "llm"
        except LlmError:
            pass
    return fallback_keyword(text), "rule"


_SUMMARY_SYSTEM = (
    "你是比价助手。用 2-3 句中文总结下面的比价结果。"
    "严格规则：只能使用给定事实；不得改动或编造任何数字与优惠；"
    "如果结果里说明京东侧未命中或降级，必须如实提及。输出纯文本，不要 markdown。"
)


def _fact_sheet(result: PipelineResult) -> str:
    lines = [
        f"关键词：{result.keyword}",
        f"检索范围：拼多多 {result.pdd_count} 条，京东京粉频道 {result.jd_count} 条",
    ]
    lines.append(
        f"源商品：{result.source.title}（{PLATFORM_LABELS[result.source.platform]}，标价￥{fmt_amount(result.source.list_price)}）"
    )
    for item in result.confirmed[1:]:
        lines.append(
            f"同款：{item.offer.title}（{PLATFORM_LABELS[item.offer.platform]}，标价￥{fmt_amount(item.offer.list_price)}；{item.reason}）"
        )
    for offer, breakdown in result.breakdowns:
        lines.append(
            f"到手价：{PLATFORM_LABELS[offer.platform]} ￥{fmt_amount(breakdown.final_price)}"
            f"（共省￥{fmt_amount(breakdown.total_discount)}）"
        )
    lines.extend(f"备注：{note}" for note in result.notes)
    return "\n".join(lines)


def fallback_summary(result: PipelineResult) -> str:
    """模板兜底：只复述确定性结果，不含任何生成性内容。"""
    cheapest_offer, cheapest = min(result.breakdowns, key=lambda pair: pair[1].final_price)
    text = (
        f"已为「{result.keyword}」匹配到 {len(result.confirmed)} 个同款报价，"
        f"最低到手价 ￥{fmt_amount(cheapest.final_price)}（{PLATFORM_LABELS[cheapest_offer.platform]}）。"
    )
    if result.notes:
        text += " " + "；".join(result.notes) + "。"
    return text


def compose_summary(
    result: PipelineResult, client: DeepSeekClient | None = None, *, usage_sink=None
) -> tuple[str, str]:
    """生成报告导语。返回 (text, method)，method ∈ {"llm", "rule"}。"""
    if client is not None:
        try:
            text = client.complete(_SUMMARY_SYSTEM, _fact_sheet(result), usage_sink=usage_sink)
            cleaned = str(text).strip()
            if cleaned:
                return cleaned, "llm"
        except LlmError:
            pass
    return fallback_summary(result), "rule"
