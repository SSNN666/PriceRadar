"""工具层：把确定性能力封装成统一签名的工具。

设计原则：
- 每个工具只做一件事；参数/返回一律用领域模型，不用裸 dict
- 工具失败抛各自领域异常（PddApiError / JdApiError），由编排层决定降级策略
- 该注册表是未来 LangGraph 节点与 MCP Server 的工具来源（写一次，两处消费）
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from priceradar.domain.models import Offer
from priceradar.engine.pricing import PriceBreakdown, compute_price
from priceradar.matching import MatchReport, match_offers
from priceradar.sources.jd import JdApiError, JdClient
from priceradar.jobs.jd_pool import load_pool
from priceradar.sources.jd_channels import CHANNEL_LABELS, relax_tokens, route_channels
from priceradar.sources.links import extract_pdd_link
from priceradar.sources.pdd import PddApiError, PddClient

# 京东侧检索参数（实测见 docs/jd-api-findings.md §9.5）
JD_PAGES_PER_CHANNEL = 4
JD_PAGE_SIZE = 50
JD_MAX_RESULTS = 60            # 单次查询返回上限（避免匹配池过大）
JD_MIN_HITS_BEFORE_RELAX = 3   # 精确命中不足此数才放宽（放宽会引入噪声，不轻易用）


def search_pdd_goods(keyword: str, page_size: int = 30, *, expand_link: bool = True) -> list[Offer]:
    """拼多多检索。

    传**商品链接**时：先按链接精确反查到那件商品，再自动用平台相似推荐
    （`channel_type=3`）扩展候选池——否则"粘链接"只会得到孤零零一个报价，
    没有比价对象，而用户粘链接恰恰是想知道"这个值不值"。

    扩展失败不影响主结果（主商品照常返回）。
    """
    client = PddClient.from_env()
    offers = client.search_goods(keyword, page_size=page_size)
    if not (expand_link and extract_pdd_link(keyword) and offers):
        return offers

    sign = offers[0].source_extra.get("goods_sign")
    if not sign:
        return offers
    try:
        similar = client.recommend_similar(sign, limit=20)
    except PddApiError:      # 扩展是增值，失败不阻塞
        return offers
    seen = {offer.sku_id for offer in offers}
    return offers + [offer for offer in similar if offer.sku_id not in seen]


def search_jd_channel(elite_id: int = 24, page_size: int = 50) -> list[Offer]:
    """京东京粉精选频道商品（按频道 ID；eliteId 地图见 docs/jd-api-findings.md §9.2）。"""
    return JdClient.from_env().jingfen_goods(elite_id=elite_id, page_size=page_size)


def _fetch_jd_channel(client: JdClient, elite_id: int, pages: int, page_size: int) -> list[Offer]:
    """拉取单个频道的多页（翻到最后一页会抛错，视为正常终止，不计为失败）。"""
    offers: list[Offer] = []
    for page in range(1, pages + 1):
        try:
            batch = client.jingfen_goods(elite_id=elite_id, page_index=page, page_size=page_size)
        except JdApiError:
            break
        if not batch:
            break
        offers.extend(batch)
    return offers


def _filter_jd(offers: list[Offer], tokens: list[str]) -> list[Offer]:
    return [offer for offer in offers if any(token in offer.title for token in tokens)]


def _best_hits(offers: list[Offer], keyword: str) -> list[Offer]:
    """精确子串优先；命中不足时放宽（实测「降噪耳机」0 条但「蓝牙耳机」23 条）。"""
    exact = _filter_jd(offers, [keyword])
    if len(exact) >= JD_MIN_HITS_BEFORE_RELAX:
        return exact
    relaxed = _filter_jd(offers, relax_tokens(keyword))
    return relaxed if len(relaxed) > len(exact) else exact


def search_jd_goods(
    keyword: str,
    *,
    channels: list[int] | None = None,
    pages_per_channel: int = JD_PAGES_PER_CHANNEL,
    page_size: int = JD_PAGE_SIZE,
    max_results: int = JD_MAX_RESULTS,
    max_workers: int = 6,
    use_cache: bool = True,
) -> list[Offer]:
    """按关键词检索京东商品。

    **首选路径：读本地频道池缓存**（`jobs/jd_pool.py` 刷新）。现场拉取需 3 频道 × 4 页
    = 12 次调用、耗时 2.4 秒；走缓存则是毫秒级、零 API 调用，且**全池都在本地，
    命中不足时直接全池搜索，不必再付"扩大范围"的额外开销**（实测「运动鞋」只存在于
    路由外的频道，现场版要多花 4 秒）。

    **回退路径：现场按品类路由拉取**（缓存缺失时）。京东没有关键词搜索接口
    （`goods.query` 403），只能按频道拉池——"搜京东搜不到"的根因是**默认频道选错**
    （搜保温杯却拉数码频道），而非京东没货。
    """
    explicit = channels is not None
    if use_cache and not explicit:
        cached = load_pool()
        if cached:
            return _best_hits(cached, keyword)[:max_results]

    client = JdClient.from_env()
    targets = channels or route_channels(keyword)

    def fetch(target_list: list[int]) -> dict[str, Offer]:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batches = list(
                executor.map(lambda elite: _fetch_jd_channel(client, elite, pages_per_channel, page_size), target_list)
            )
        pool: dict[str, Offer] = {}      # 按 sku 去重（同一商品可能出现在多个频道）
        for batch in batches:
            for offer in batch:
                pool.setdefault(offer.sku_id, offer)
        return pool

    pool = fetch(targets)
    hits = _best_hits(list(pool.values()), keyword)
    if not explicit and len(hits) < JD_MIN_HITS_BEFORE_RELAX:
        # 频道标签是按各频道头几条标题推断的，**实际内容比标签杂**（实测"运动鞋"
        # 出现在家居/百货/母婴/美妆频道，无一在服饰路由内）——命中不足时扩大范围
        extra = [c for c in CHANNEL_LABELS if c not in targets]
        pool.update(fetch(extra))
        hits = _best_hits(list(pool.values()), keyword)
    return hits[:max_results]


def compute_best_price(offer: Offer) -> PriceBreakdown:
    """到手价引擎（纯函数——"钱不由 LLM 算"原则的落点）。"""
    return compute_price(offer)


def match_same_product(source: Offer, candidates: list[Offer]) -> MatchReport:
    """同款匹配：规格硬规则 + 相似度 sanity + A/B/C 分档（含排除原因）。"""
    return match_offers(source, candidates)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    func: Callable[..., Any]


TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        Tool("search_pdd_goods", "拼多多关键词搜索商品", search_pdd_goods),
        Tool("search_jd_goods", "京东按关键词检索商品（品类路由 + 关键词过滤）", search_jd_goods),
        Tool("search_jd_channel", "京东京粉精选频道商品（按频道 ID，定点排查用）", search_jd_channel),
        Tool("compute_best_price", "计算单个商品报价的公开优惠口径到手价", compute_best_price),
        Tool("match_same_product", "在候选池中找出同款（输出 A/B/C 分档）", match_same_product),
    )
}
