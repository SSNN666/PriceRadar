"""决策融合测试（PRD §5.8：信号 → 结论，确定性可审计）。"""

from decimal import Decimal

from priceradar.domain.models import Offer, Platform, ShopKind, Signals
from priceradar.engine.pricing import compute_price
from priceradar.insights.decision import (
    DEFAULT_POLICY,
    DecisionPolicy,
    DecisionTier,
    decide,
)
from priceradar.insights.quality import QualityTier

D = Decimal


def candidate(sku: str, price: str, kind: ShopKind | None, rate=None, comments=None):
    offer = Offer(
        platform=Platform.JD,
        sku_id=sku,
        title="测试商品",
        list_price=D(price),
        shop_kind=kind,
        signals=Signals(good_comments_share=rate, comments=comments),
    )
    return (offer, compute_price(offer))


TRUSTED = dict(kind=ShopKind.FLAGSHIP, rate=98.0, comments=2000)   # 有资质 + 信号达标


# —— 规则 3：推荐 ——


def test_recommends_cheapest_tier1_when_it_is_also_cheapest():
    decision = decide([candidate("a", "100", **TRUSTED), candidate("b", "120", kind=ShopKind.POP)])
    assert decision.tier is DecisionTier.RECOMMEND
    assert decision.recommended.sku_id == "a"
    assert D("100") == decision.recommended_price


def test_tradeoff_when_cheaper_option_is_not_tier1():
    """差价换可信度：更便宜的没资质 → 给出"最低价 X / 推荐 Y（贵 ¥N）"。"""
    decision = decide([candidate("a", "100", **TRUSTED), candidate("b", "80", kind=ShopKind.POP)])
    assert decision.tier is DecisionTier.TRADEOFF
    assert decision.recommended.sku_id == "a"      # 推荐贵的那个
    assert decision.cheapest.sku_id == "b"         # 但如实告知更便宜的在哪
    assert "80" in decision.tradeoff and "20" in decision.tradeoff  # 差价 ¥20


def test_tradeoff_states_the_reason_not_just_the_price():
    """陈述必须说明"为什么便宜的那个不推荐"，否则用户无法判断取舍。"""
    decision = decide([candidate("a", "100", **TRUSTED), candidate("b", "80", kind=ShopKind.PERSONAL)])
    assert "个人店" in decision.tradeoff


def test_signal_failure_also_blocks_tier1():
    """有资质但质量信号未达标（样本不足）→ 同样不算 T1，且原因要说清。"""
    decision = decide([candidate("a", "100", **TRUSTED), candidate("b", "90", kind=ShopKind.FLAGSHIP, rate=100.0, comments=3)])
    assert decision.tier is DecisionTier.TRADEOFF
    assert "质量信号未达标" in decision.tradeoff


# —— 规则 6：降级 ——


def test_falls_back_to_price_only_when_nothing_qualifies():
    """全部候选都不达标 → 降级为纯价格对比，**不强行出推荐**（PRD §5.8 边界）。"""
    decision = decide([candidate("a", "100", kind=ShopKind.POP), candidate("b", "80", kind=ShopKind.PERSONAL)])
    assert decision.tier is DecisionTier.FALLBACK
    assert decision.recommended is None
    assert decision.note                                  # 必须说明为什么不给结论


def test_unknown_shop_kind_does_not_qualify():
    """资质未知（None）不能当作达标——缺失就是缺失，不猜。"""
    decision = decide([candidate("a", "100", kind=None, rate=98.0, comments=2000)])
    assert decision.tier is DecisionTier.FALLBACK


# —— 规则 5：不建议 ——


def test_discourages_on_high_risk_without_tier1():
    decision = decide([candidate("a", "100", kind=ShopKind.POP)], risk_topics=[("翻新机风险", 3)])
    assert decision.tier is DecisionTier.DISCOURAGE
    assert decision.high_risk_topics == ["翻新机风险"]


def test_risk_below_cross_validation_threshold_is_ignored():
    """跨查询词验证 ≥ 2 处才算——单处提及不出"不建议"（宁缺毋滥）。"""
    decision = decide([candidate("a", "100", kind=ShopKind.POP)], risk_topics=[("翻新机风险", 1)])
    assert decision.tier is DecisionTier.FALLBACK


def test_risk_topic_must_be_high_risk_kind():
    """非高危主题（如"续航缩水"）不触发不建议。"""
    decision = decide([candidate("a", "100", kind=ShopKind.POP)], risk_topics=[("续航缩水", 5)])
    assert decision.tier is DecisionTier.FALLBACK


def test_risk_does_not_override_available_tier1():
    """有可靠候选时，风险不阻断推荐——PRD 规则 5 限定的是"无 T1/T2 候选"。"""
    decision = decide([candidate("a", "100", **TRUSTED)], risk_topics=[("翻新机风险", 3)])
    assert decision.tier is DecisionTier.RECOMMEND
    assert decision.high_risk_topics == ["翻新机风险"]   # 但风险仍如实带出


# —— 规则 4：观望 ——


def test_watch_when_price_at_high_percentile_and_campaign_near():
    decision = decide(
        [candidate("a", "100", **TRUSTED)],
        history_by_sku={"jd:a": {"status": "ok", "percentile": 0.92}},
        days_to_campaign=7,
    )
    assert decision.tier is DecisionTier.WATCH
    assert "观望" in decision.headline


def test_watch_requires_enough_samples():
    """样本不足（< 14 天）不出观望结论——PRD §5.8 的样本门槛。"""
    decision = decide(
        [candidate("a", "100", **TRUSTED)],
        history_by_sku={"jd:a": {"status": "insufficient", "percentile": 0.92}},
        days_to_campaign=7,
    )
    assert decision.tier is DecisionTier.RECOMMEND


def test_watch_requires_campaign_to_be_near():
    """离大促还远时，价格高位不构成"等一等"的理由。"""
    decision = decide(
        [candidate("a", "100", **TRUSTED)],
        history_by_sku={"jd:a": {"status": "ok", "percentile": 0.92}},
        days_to_campaign=60,
    )
    assert decision.tier is DecisionTier.RECOMMEND


def test_watch_requires_high_percentile():
    decision = decide(
        [candidate("a", "100", **TRUSTED)],
        history_by_sku={"jd:a": {"status": "ok", "percentile": 0.3}},
        days_to_campaign=7,
    )
    assert decision.tier is DecisionTier.RECOMMEND


# —— 配置化（PRD 要求阈值全部可配）——


def test_policy_thresholds_are_configurable():
    """放宽店铺资质后，POP 店也能成为 T1（信号仍须达标）。"""
    lenient = DecisionPolicy(qualifying_kinds=frozenset({ShopKind.POP}))
    decision = decide([candidate("a", "100", kind=ShopKind.POP, rate=98.0, comments=2000)], policy=lenient)
    assert decision.tier is DecisionTier.RECOMMEND

    # 对照：默认策略下同一候选不达标
    assert decide([candidate("a", "100", kind=ShopKind.POP, rate=98.0, comments=2000)]).tier is DecisionTier.FALLBACK


def test_default_policy_excludes_personal_and_pop():
    from priceradar.domain.models import CERTIFIED_SHOP_KINDS

    assert DEFAULT_POLICY.qualifying_kinds == CERTIFIED_SHOP_KINDS
    assert ShopKind.PERSONAL not in DEFAULT_POLICY.qualifying_kinds
    assert QualityTier.STRONG in DEFAULT_POLICY.qualifying_quality


def test_empty_candidates_returns_fallback_not_crash():
    decision = decide([])
    assert decision.tier is DecisionTier.FALLBACK
    assert decision.recommended is None
