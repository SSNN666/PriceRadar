"""决策融合：信号 → 结论（PRD §5.8，确定性可审计）。

**定位**：把"数据展示"变成"决策建议"。此前卡片只列价格与信号，用户得自己权衡；
本模块按**可展示、可单测的规则表**给出档位与结论。

⚠️ **仍然不做"综合分"**（PRD §5.8 输出形态明令禁止伪科学评分）。信号不合成一个数字，
而是进入规则表产出档位；LLM 只负责文案表达，不参与判定。

规则（PRD §5.8 决策融合规则）：
1. 候选按到手价升序
2. 分层：**T1** = 平台认证资质店铺 且 信号达标；**T2** = 其余
3. T1 最低价 → **推荐**；若 T2 更便宜 → 输出"最低价 X（⚠️ 原因）/ 推荐 X（贵 ¥N）"
4. **观望**：到手价处于观测高分位且距大促临近（样本不足时不出此档）
5. **不建议**：命中高危风险且无 T1 候选
6. **降级**：无 T1 候选时退为"纯价格对比 + 信号不可得标注"，不强行出档位
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field

from priceradar.domain.models import CERTIFIED_SHOP_KINDS, SHOP_KIND_LABELS, Offer, ShopKind
from priceradar.engine.pricing import PriceBreakdown
from priceradar.formatting import fmt_amount
from priceradar.insights.quality import QualityAssessment, QualityTier, assess_quality


class DecisionTier(str, Enum):
    RECOMMEND = "recommend"    # 推荐：T1 中的最低价
    TRADEOFF = "tradeoff"      # 有取舍：更便宜的 T2 存在，差价换可信度
    WATCH = "watch"            # 观望：价格处于高位且大促临近
    DISCOURAGE = "discourage"  # 不建议：命中高危风险且无可靠候选
    FALLBACK = "fallback"      # 降级：信号普遍缺失，只做纯价格对比

    @property
    def label(self) -> str:
        return _TIER_LABELS[self]


_TIER_LABELS = {
    DecisionTier.RECOMMEND: "推荐",
    DecisionTier.TRADEOFF: "有取舍",
    DecisionTier.WATCH: "观望",
    DecisionTier.DISCOURAGE: "不建议",
    DecisionTier.FALLBACK: "仅价格对比",
}

# 高危风险词表（PRD §5.8 举的例：翻新/假货/退货纠纷类）
DEFAULT_RISK_TERMS = (
    "翻新", "假货", "假一赔", "仿品", "山寨", "盗版", "以次充好",
    "退货纠纷", "质量差", "货不对板", "虚假宣传", "欺诈",
)


class DecisionPolicy(BaseModel):
    """决策阈值——PRD §5.8 要求**全部配置化**，随评测集与用户反馈调优。"""

    qualifying_kinds: frozenset[ShopKind] = CERTIFIED_SHOP_KINDS
    qualifying_quality: frozenset[QualityTier] = frozenset({QualityTier.STRONG, QualityTier.FAIR})
    risk_terms: tuple[str, ...] = DEFAULT_RISK_TERMS
    risk_min_mentions: int = 2          # 跨查询词验证 ≥ 2 处（聚类已跨词合并计数）
    watch_percentile: float = 0.8       # 到手价处于观测 80 分位以上
    watch_days_to_campaign: int = 14    # 距大促 < 14 天


DEFAULT_POLICY = DecisionPolicy()


class Decision(BaseModel):
    """一次查询的结论（含依据与降级说明，无综合分）。"""

    tier: DecisionTier
    label: str = Field(description="展示用中文档位")
    headline: str = Field(description="一句话结论")
    recommended: Offer | None = Field(default=None, description="推荐项（T1 最低价）")
    recommended_price: Decimal | None = None
    cheapest: Offer | None = Field(default=None, description="全量最低价（可能不建议买）")
    cheapest_price: Decimal | None = None
    tradeoff: str | None = Field(default=None, description="差价换可信度陈述")
    high_risk_topics: list[str] = Field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class _Candidate:
    offer: Offer
    breakdown: PriceBreakdown
    quality: QualityAssessment
    tier1: bool
    why_not_tier1: str


def _is_high_risk(topic: str, policy: DecisionPolicy) -> bool:
    return any(term in topic for term in policy.risk_terms)


def _qualify(offer: Offer, quality: QualityAssessment, policy: DecisionPolicy) -> tuple[bool, str]:
    """判定是否 T1（资质店铺 + 信号达标），并给出未达标的原因（供差价陈述用）。"""
    if offer.shop_kind is None:
        return False, "店铺资质未知"
    if offer.shop_kind not in policy.qualifying_kinds:
        label = SHOP_KIND_LABELS.get(offer.shop_kind)
        return False, f"{label}非平台认证资质" if label else "非平台认证资质"
    if quality.tier not in policy.qualifying_quality:
        return False, f"质量信号未达标（{quality.label}）"
    return True, ""


def _watch_hit(candidate: _Candidate, history_by_sku: dict[str, dict], days_to_campaign: int | None, policy: DecisionPolicy) -> bool:
    """观望：样本足够 + 当前价处于观测高分位 + 大促临近（三者缺一不出此档）。"""
    if days_to_campaign is None or days_to_campaign > policy.watch_days_to_campaign:
        return False
    history = history_by_sku.get(f"{candidate.offer.platform.value}:{candidate.offer.sku_id}")
    if not history or history.get("status") != "ok":
        return False  # 样本不足（< 14 天）→ 按 PRD 规定不出观望结论
    percentile = history.get("percentile")
    return percentile is not None and float(percentile) >= policy.watch_percentile


def decide(
    breakdowns: list[tuple[Offer, PriceBreakdown]],
    *,
    quality_of: Callable[[Offer], QualityAssessment] = assess_quality,
    history_by_sku: dict[str, dict] | None = None,
    days_to_campaign: int | None = None,
    risk_topics: list[tuple[str, int]] | None = None,
    policy: DecisionPolicy = DEFAULT_POLICY,
) -> Decision:
    """按规则表产出决策（确定性、可单测；LLM 不参与判定）。

    risk_topics 为**商品级**信号（L2 风险扫描按查询商品检索，非按报价），
    故"不建议"是商品级结论而非针对某个报价——输出中不把它算到具体店铺头上。
    """
    if not breakdowns:
        return Decision(
            tier=DecisionTier.FALLBACK,
            label=DecisionTier.FALLBACK.label,
            headline="无候选可比",
            note="没有可比较的报价。",
        )

    ordered = sorted(breakdowns, key=lambda pair: pair[1].final_price)
    candidates: list[_Candidate] = []
    for offer, breakdown in ordered:
        quality = quality_of(offer)                       # 只算一次，判定与展示共用同一结果
        qualifies, reason = _qualify(offer, quality, policy)
        candidates.append(_Candidate(offer, breakdown, quality, qualifies, reason))

    cheapest = candidates[0]
    tier1 = [c for c in candidates if c.tier1]
    high_risk = [
        topic for topic, mentions in (risk_topics or []) if _is_high_risk(topic, policy) and mentions >= policy.risk_min_mentions
    ]

    # 规则 5：命中高危且无 T1 候选 → 不建议（商品级结论）
    if high_risk and not tier1:
        return Decision(
            tier=DecisionTier.DISCOURAGE,
            label=DecisionTier.DISCOURAGE.label,
            headline=f"暂不建议购买：检索到 {len(high_risk)} 类集中风险，且没有资质与信号都达标的候选",
            cheapest=cheapest.offer,
            cheapest_price=cheapest.breakdown.final_price,
            high_risk_topics=high_risk,
            note="风险为商品级信号（跨查询词验证），未指向具体店铺；请结合风险卡自行判断。",
        )

    # 规则 6：无 T1 → 降级为纯价格对比，不强行出档位
    if not tier1:
        return Decision(
            tier=DecisionTier.FALLBACK,
            label=DecisionTier.FALLBACK.label,
            headline=f"暂无可信度达标的候选，仅按价格对比：最低 ￥{fmt_amount(cheapest.breakdown.final_price)}",
            cheapest=cheapest.offer,
            cheapest_price=cheapest.breakdown.final_price,
            high_risk_topics=high_risk,
            note="当前候选的店铺资质或质量信号均未达标（缺口已在每条报价上标注），故不给出推荐结论。",
        )

    recommended = tier1[0]

    # 规则 4：观望（优先于"推荐"——价格在高位且大促临近时，现在买不划算）
    if _watch_hit(recommended, history_by_sku or {}, days_to_campaign, policy):
        return Decision(
            tier=DecisionTier.WATCH,
            label=DecisionTier.WATCH.label,
            headline=f"建议观望：当前价 ￥{fmt_amount(recommended.breakdown.final_price)} 处于观测高位，且距大促不足 {policy.watch_days_to_campaign} 天",
            recommended=recommended.offer,
            recommended_price=recommended.breakdown.final_price,
            cheapest=cheapest.offer,
            cheapest_price=cheapest.breakdown.final_price,
            note="基于自观测数据（非平台全历史）的分位判断。",
        )

    # 规则 3：推荐 T1 最低价；若更便宜的是 T2/T3 → 差价换可信度
    if cheapest.offer.sku_id != recommended.offer.sku_id:
        gap = recommended.breakdown.final_price - cheapest.breakdown.final_price
        tradeoff = (
            f"最低价 ￥{fmt_amount(cheapest.breakdown.final_price)}（{cheapest.why_not_tier1}）"
            f" / 推荐 ￥{fmt_amount(recommended.breakdown.final_price)}（贵 ￥{fmt_amount(gap)}）"
        )
        return Decision(
            tier=DecisionTier.TRADEOFF,
            label=DecisionTier.TRADEOFF.label,
            headline=tradeoff,
            recommended=recommended.offer,
            recommended_price=recommended.breakdown.final_price,
            cheapest=cheapest.offer,
            cheapest_price=cheapest.breakdown.final_price,
            tradeoff=tradeoff,
            high_risk_topics=high_risk,
        )

    return Decision(
        tier=DecisionTier.RECOMMEND,
        label=DecisionTier.RECOMMEND.label,
        headline=f"推荐 ￥{fmt_amount(recommended.breakdown.final_price)}：资质与质量信号均达标，且是其中最低价",
        recommended=recommended.offer,
        recommended_price=recommended.breakdown.final_price,
        cheapest=cheapest.offer,
        cheapest_price=cheapest.breakdown.final_price,
        high_risk_topics=high_risk,
    )
