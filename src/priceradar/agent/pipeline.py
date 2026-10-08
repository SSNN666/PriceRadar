"""单轮管线（Agent 骨架的确定性版本，PRD §4 核心流程的最小实现）。

关键词/自然语言 → 拼多多检索（源池）→ 京东频道候选 → 同款匹配（A 档成组）
→ 同款组去重折叠 → 到手价引擎 → 对比卡（附轻量 trace）。

设计约束（PRD §5.2 / §4 / §8）：
- 管线核心不含 LLM：LLM 只在 CLI 边缘做"查询理解/报告导语"（llm_nodes），且全部可兜底
- 京东候选池失败视为降级：继续出拼多多对比卡并明确标注，不阻塞主流程
- 同款命中不足时如实标注（"未找到确切同款"），不假装精确
- 每次运行记录 trace（节点耗时 / API 调用数 / 估算成本），CLI 下追加落盘 data/traces
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable

from priceradar.agent.tools import (
    compute_best_price,
    match_same_product,
    search_jd_channel,
    search_jd_goods,
    search_pdd_goods,
)
from priceradar.config import project_root
from priceradar.domain.models import Offer
from priceradar.engine.pricing import PriceBreakdown
from priceradar.formatting import fmt_amount
from priceradar.insights import attach_history_insights, campaign_context
from priceradar.matching import MatchTier, extract_specs, similarity
from priceradar.report import PLATFORM_LABELS, render_price_card
from priceradar.risk import RiskSummary
from priceradar.sources.jd import JdApiError
from priceradar.sources.pdd import PddApiError
from priceradar.trace import Trace

_GROUP_SIMILARITY = 0.5  # 同款折叠的标题相似度下限（同平台 + 同到手价 + 同规格 + 相似标题）


@dataclass
class ConfirmedItem:
    offer: Offer
    reason: str


@dataclass
class OfferGroup:
    """同款组：把"不同店铺铺的同一个商品、同价"折叠成一条。"""

    representative: Offer
    members: list[Offer]
    final_price: Decimal
    reason: str

    @property
    def count(self) -> int:
        return len(self.members)


@dataclass
class PipelineResult:
    keyword: str
    pdd_count: int
    jd_count: int
    source: Offer
    confirmed: list[ConfirmedItem]      # 同款组全量（含源商品，首项为源）
    groups: list[OfferGroup]            # 折叠后的同款组（源所在组排第一）
    similar_count: int
    excluded_summary: dict[str, int]
    breakdowns: list[tuple[Offer, PriceBreakdown]]  # 每组一个代表
    jd_error: str | None = None
    notes: list[str] = field(default_factory=list)
    summary: str | None = None          # 报告导语（LLM 生成或模板兜底，由 CLI 层填入）
    trace: Trace | None = None
    risk: RiskSummary | None = None     # L2 风险扫描（可选增强）
    history: dict[str, dict] = field(default_factory=dict)  # 价格观测（键：platform:sku_id）
    similar_items: list[ConfirmedItem] = field(default_factory=list)  # 近似款（未参与比价，供参考）

    @property
    def cross_platform_hit(self) -> bool:
        return any(item.offer.platform is not self.source.platform for item in self.confirmed)


def _pick_source(pool: list[Offer], probe: int) -> Offer:
    """在检索结果中挑"同款重复出现最多"的商品作为源（确定性启发式）。

    好商品往往被多个店铺重复铺货——出现次数最多的同款组即本轮的"目标商品"。
    """
    best_source = pool[0]
    best_count = -1
    for candidate_source in pool[:probe]:
        report = match_same_product(candidate_source, pool)
        count = sum(1 for result in report.results if result.tier is MatchTier.CONFIRMED)
        if count > best_count:
            best_source, best_count = candidate_source, count
    return best_source


def _group_confirmed(confirmed: list[ConfirmedItem], price_by_sku: dict[str, Decimal], source: Offer) -> list[OfferGroup]:
    """同款去重折叠：同平台 + 同到手价 + 同规格 + 标题高度相似 → 并为一组。"""
    groups: list[OfferGroup] = []
    for item in confirmed:
        price = price_by_sku[item.offer.sku_id]
        matched: OfferGroup | None = None
        for group in groups:
            representative = group.representative
            if item.offer.platform is not representative.platform:
                continue
            if price != group.final_price:
                continue
            if extract_specs(item.offer.title).models != extract_specs(representative.title).models:
                continue
            if similarity(representative.title, item.offer.title) < _GROUP_SIMILARITY:
                continue
            matched = group
            break
        if matched is None:
            groups.append(OfferGroup(representative=item.offer, members=[item.offer], final_price=price, reason=item.reason))
        else:
            matched.members.append(item.offer)
            if similarity(source.title, item.offer.title) > similarity(source.title, matched.representative.title):
                matched.representative = item.offer  # 更接近源商品者作为代表（源自身的相似度为 1，不会被替换）
                matched.reason = item.reason

    def sort_key(group: OfferGroup) -> tuple:
        is_source_group = any(member.platform is source.platform and member.sku_id == source.sku_id for member in group.members)
        return (0 if is_source_group else 1, group.final_price, -group.count, group.representative.sku_id)

    groups.sort(key=sort_key)
    return groups


def run(
    keyword: str,
    *,
    pdd_searcher: Callable[[str, int], list[Offer]] | None = None,
    jd_searcher: Callable[[str], list[Offer]] | None = None,
    pdd_page_size: int = 30,
    jd_enabled: bool = True,
    jd_elite_id: int | None = None,
    jd_pool_size: int = 50,
    probe: int = 5,
    trace: Trace | None = None,
    with_risk: bool = False,
    risk_scanner: Callable[[str], RiskSummary] | None = None,
    history_enricher: Callable[[PipelineResult], None] | None = None,
) -> PipelineResult:
    """执行单轮管线（searcher 可注入，测试离线化；默认走官方 API）。"""
    pdd_searcher = pdd_searcher or (lambda kw, size: search_pdd_goods(kw, size))
    if jd_searcher is None:
        # 京东无关键词搜索接口 → 默认走品类路由（查询 → 频道 → 拉池 → 过滤）；
        # 显式给了 eliteId 则定点拉该频道（排查/复现用）
        jd_searcher = (
            (lambda _kw: search_jd_channel(jd_elite_id, jd_pool_size))
            if jd_elite_id is not None
            else search_jd_goods
        )
    trace = trace or Trace()

    notes: list[str] = []
    with trace.span("pdd_search", f"关键词={keyword}"):
        pool = pdd_searcher(keyword, pdd_page_size)
    trace.count("pdd")
    if not pool:
        raise ValueError(f"关键词 {keyword!r} 无检索结果")

    jd_pool: list[Offer] = []
    jd_error: str | None = None
    if jd_enabled:
        jd_label = f"eliteId={jd_elite_id}" if jd_elite_id is not None else "品类路由"
        with trace.span("jd_search", jd_label):
            try:
                jd_pool = jd_searcher(keyword)
                trace.count("jd")
            except (JdApiError, OSError) as exc:  # 京东侧失败 → 降级不阻塞
                jd_error = str(exc)
                notes.append(f"京东侧检索失败（{jd_error}），本次仅输出拼多多对比")

    with trace.span("match"):
        source = _pick_source(pool, probe)
        candidates = [
            offer for offer in pool if not (offer.platform is source.platform and offer.sku_id == source.sku_id)
        ] + jd_pool
        report = match_same_product(source, candidates)

    confirmed = [ConfirmedItem(offer=source, reason="检索结果（源商品）")]
    confirmed += [
        ConfirmedItem(offer=result.candidate, reason="；".join(result.reasons))
        for result in report.results
        if result.tier is MatchTier.CONFIRMED
    ]
    similar_items = [
        ConfirmedItem(offer=result.candidate, reason="；".join(result.reasons))
        for result in report.results
        if result.tier is MatchTier.SIMILAR
    ]

    if len(confirmed) == 1:
        notes.append("未找到确切同款（A 档为空），以下仅列出源商品到手价")
    elif not any(item.offer.platform is not source.platform for item in confirmed):
        if not jd_error:
            if jd_pool:
                notes.append(f"京东侧未找到确切同款（候选 {len(jd_pool)} 条均未达 A 档）")
            else:
                notes.append("京东候选池为空，未参与匹配")

    similar_count = sum(1 for result in report.results if result.tier is MatchTier.SIMILAR)
    excluded_summary: dict[str, int] = {}
    for excluded in report.excluded:
        key = excluded.reason.split("（")[0]
        excluded_summary[key] = excluded_summary.get(key, 0) + 1

    with trace.span("pricing", f"{len(confirmed)} 个报价"):
        breakdown_by_sku = {item.offer.sku_id: compute_best_price(item.offer) for item in confirmed}

    price_by_sku = {sku: breakdown.final_price for sku, breakdown in breakdown_by_sku.items()}
    groups = _group_confirmed(confirmed, price_by_sku, source)
    breakdowns = [(group.representative, breakdown_by_sku[group.representative.sku_id]) for group in groups]

    risk: RiskSummary | None = None
    if with_risk:
        with trace.span("risk_search", f"对象={keyword}"):
            try:
                if risk_scanner is not None:
                    risk = risk_scanner(keyword)
                else:
                    from priceradar.risk import scan_with_env

                    risk = scan_with_env(keyword)
                trace.count("search_bocha", len(risk.queries))
            except Exception as exc:  # 风险扫描为可选增强：失败不阻塞主流程
                notes.append(f"风险扫描未执行（{exc}）")

    result = PipelineResult(
        keyword=keyword,
        pdd_count=len(pool),
        jd_count=len(jd_pool),
        source=source,
        confirmed=confirmed,
        groups=groups,
        similar_count=similar_count,
        excluded_summary=excluded_summary,
        breakdowns=breakdowns,
        jd_error=jd_error,
        notes=notes,
        trace=trace,
        risk=risk,
        similar_items=similar_items,
    )
    (history_enricher or attach_history_insights)(result)
    return result


def render(result: PipelineResult) -> str:
    """把管线结果渲染成完整输出（同款组 + 到手价对比卡 + 候选统计 + trace）。"""
    lines: list[str] = []
    lines.append("==== 真价雷达 · 单轮管线（真实数据）====")
    lines.append(f"关键词：{result.keyword}")
    search_line = f"检索：拼多多 {result.pdd_count} 条"
    if result.jd_count:
        search_line += f" ｜ 京东京粉频道 {result.jd_count} 条"
    lines.append(search_line)
    if result.summary:
        lines.append("")
        lines.append(result.summary)
    lines.append("")
    lines.append("【同款组】")

    source_group = result.groups[0] if result.groups else None
    source_note = f"｜所在同款组 ×{source_group.count} 家店铺" if source_group and source_group.count > 1 else ""
    lines.append(
        f"源商品：[{PLATFORM_LABELS[result.source.platform]}] {result.source.title[:40]}"
        f"（￥{fmt_amount(result.source.list_price)}）{source_note}"
    )
    for group in result.groups[1:]:
        count_note = f"同款 ×{group.count} 家｜" if group.count > 1 else ""
        lines.append(
            f"  + [{PLATFORM_LABELS[group.representative.platform]}] 到手 ￥{fmt_amount(group.final_price)} "
            f"{count_note}{group.representative.title[:32]}（{group.reason}）"
        )
    for note in result.notes:
        lines.append(f"  △ {note}")
    lines.append("")
    lines.append("【到手价明细】")
    count_by_sku = {group.representative.sku_id: group.count for group in result.groups if group.count > 1}
    lines.append(render_price_card(result.breakdowns, count_by_sku=count_by_sku))
    if result.history:
        lines.append("")
        lines.append("【价格观测】（自采快照 · 诚实标注观测天数）")
        context = campaign_context()
        if context:
            lines.append(f"  大促日历（参考往年规律）：{context}")
        for offer, _ in result.breakdowns:
            view = result.history.get(f"{offer.platform.value}:{offer.sku_id}")
            if not view:
                continue
            lines.append(f"  - [{PLATFORM_LABELS[offer.platform]}] {offer.title[:30]}：已观测 {view['observed_days']} 天 ｜ {view['verdict']}")
            if view.get("lowest"):
                lines.append(f"    观测最低：￥{view['lowest']}")
            if view.get("fake_discount"):
                lines.append(f"    △ {view['fake_discount']}")
        lines.append("  ※ 基于自观测数据（非平台全历史）；样本不足时不出时机结论")
    if result.risk is not None:
        lines.append("")
        lines.append(f"【风险扫描】{len(result.risk.queries)} 组关键词 · 命中 {len(result.risk.items)} 条讨论")
        if result.risk.issues:
            for issue in result.risk.issues:
                lines.append(f"  - {issue['topic']}：{issue['mention_count']} 条提及")
            lines.append(f"  ※ {result.risk.note}")
        else:
            lines.append(f"  △ {result.risk.note}")
            for item in result.risk.items[:5]:
                lines.append(f"    · [{item.site_name}] {item.title[:34]}")
        lines.append("  ※ 部分平台内容未被搜索引擎完整收录，检索结果不代表全部用户反馈")
    lines.append("")
    other = f"【其他候选】近似款 {result.similar_count} 条（按设计折叠，未参与比价）"
    if result.excluded_summary:
        detail = "｜".join(f"{key} {value}" for key, value in result.excluded_summary.items())
        other += f"；排除 {sum(result.excluded_summary.values())} 条（{detail}）"
    lines.append(other)
    if result.trace is not None:
        lines.append(result.trace.summary_line())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="真价雷达 · 单轮管线（自然语言/关键词 → 对比卡）")
    parser.add_argument("query", help="自然语言或关键词，如：帮我看看QCY N20哪里买划算")
    parser.add_argument("--jd-channel", type=int, default=0, help="京东京粉频道 eliteId（0=自动品类路由，-1=关闭京东侧）")
    parser.add_argument("--pdd-page-size", type=int, default=30, help="拼多多检索条数")
    parser.add_argument("--no-llm", action="store_true", help="禁用 LLM（纯规则与模板兜底）")
    parser.add_argument("--risk", action="store_true", help="附加 L2 风险扫描（博查检索 + LLM 聚类）")
    args = parser.parse_args(argv)

    from priceradar.agent.llm import DeepSeekClient, LlmError
    from priceradar.agent.llm_nodes import compose_summary, extract_keyword
    from priceradar.risk import cluster_issues

    llm_client: DeepSeekClient | None = None
    if not args.no_llm:
        try:
            llm_client = DeepSeekClient.from_env()
        except LlmError:
            llm_client = None  # 未配置 key → 自动走确定性兜底

    trace = Trace()
    with trace.span("llm_extract"):
        keyword, extract_method = extract_keyword(args.query, llm_client)

    try:
        result = run(
            keyword,
            pdd_page_size=args.pdd_page_size,
            jd_enabled=args.jd_channel >= 0,
            jd_elite_id=args.jd_channel if args.jd_channel > 0 else None,
            trace=trace,
            with_risk=args.risk,
        )
    except PddApiError as exc:
        print(f"拼多多检索失败：{exc}")
        sys.exit(1)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    with trace.span("llm_summary"):
        result.summary, summary_method = compose_summary(result, llm_client)
    if result.risk is not None:
        with trace.span("risk_cluster"):
            result.risk = cluster_issues(result.risk, llm_client)

    llm_calls = int(extract_method == "llm") + int(summary_method == "llm")
    if result.risk is not None and not result.risk.degraded:
        llm_calls += 1
    if llm_calls:
        trace.count("llm_deepseek", llm_calls)

    print(render(result))
    try:
        trace.append_jsonl(project_root() / "data" / "traces" / "pipeline.jsonl")
    except OSError:
        pass


if __name__ == "__main__":
    main()
