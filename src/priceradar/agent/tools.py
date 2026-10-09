"""底层原语：把确定性能力封装成统一签名的函数。

设计原则：
- 每个函数只做一件事；参数/返回一律用领域模型，不用裸 dict
- 失败抛各自领域异常（PddApiError / JdApiError），由编排层决定降级策略

**这里不是 Agent 的工具注册表**——消费者是：`agent/pipeline.py`（管线内部）、
`agent/actions.py`（意图级能力）、`jobs/snapshot.py`（批量采集）。
面向 LLM 的工具注册表在 `agent/actions.py`（带参数 schema，可导出 OpenAI tools）。

（历史上这里曾有一个 `TOOLS` 注册表，注释声称要当"LangGraph 节点与 MCP Server 的
工具来源"，但从未被任何代码消费、也没有参数 schema，已于 2026-10-09 删除——
它的继任者是 `actions.ACTIONS`。）
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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
