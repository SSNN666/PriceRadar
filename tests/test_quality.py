"""质量可信度评估测试（PRD §5.8 信号卡三铁律第 3 条：好评率必须配评论数一起看）。"""

from decimal import Decimal

import pytest

from priceradar.domain.models import Offer, Platform, Signals
from priceradar.insights.quality import (
    MIN_SAMPLES,
    QualityTier,
    assess_quality,
    wilson_lower_bound,
)

D = Decimal


def jd_offer(rate=None, comments=None, **kwargs) -> Offer:
    signals = Signals(good_comments_share=rate, comments=comments, **kwargs)
    return Offer(platform=Platform.JD, sku_id="j", title="t", list_price=D("100"), signals=signals)


def pdd_offer(tier=None, sales=None, sales_tip=None) -> Offer:
    signals = Signals(shop_score_tier=tier, sales_estimate=sales, sales_tip=sales_tip)
    return Offer(platform=Platform.PDD, sku_id="p", title="t", list_price=D("100"), signals=signals)


# —— Wilson 下界本身 ——


def test_wilson_pulls_small_samples_down():
    """核心行为：样本越小，下界被拉得越低。"""
    assert wilson_lower_bound(1.00, 3) < 0.5        # 满分但只有 3 条 → 惨
    assert wilson_lower_bound(1.00, 10) < 0.8
    assert wilson_lower_bound(0.98, 2000) > 0.97    # 大样本 → 接近点估计


def test_wilson_orders_by_credibility_not_by_raw_rate():
    """用户的例子：**高好评率 + 小样本** 的可信度低于 **略低好评率 + 大样本**。

    95% / 50 条  vs  90% / 500 条 —— 前者好评率更高，可信度却更低。
    """
    assert wilson_lower_bound(0.95, 50) < wilson_lower_bound(0.90, 500)
    # 100% / 3 条 排在 97% / 2 万条 之后
    assert wilson_lower_bound(1.00, 3) < wilson_lower_bound(0.97, 20000)


def test_wilson_converges_to_rate_with_large_sample():
    assert wilson_lower_bound(0.98, 2000) == pytest.approx(0.973, abs=0.002)
    assert wilson_lower_bound(0.50, 100000) == pytest.approx(0.50, abs=0.005)


def test_wilson_edge_cases():
    assert wilson_lower_bound(0.99, 0) == 0.0        # 无样本
    assert wilson_lower_bound(1.5, 100) <= 1.0       # 越界输入被夹住
    assert wilson_lower_bound(-0.5, 100) >= 0.0


# —— 京东路径：好评率 + 评论数 ——


def test_jd_small_sample_is_flagged_not_trusted():
    """99% 好评但只有 3 条评论 → 必须标"样本不足"，不能当高分。"""
    a = assess_quality(jd_offer(rate=100.0, comments=3))
    assert a.tier is QualityTier.THIN_SAMPLE
    assert a.label == "样本不足"
    assert "3 条评论" in a.basis


def test_jd_large_sample_high_rate_is_strong():
    a = assess_quality(jd_offer(rate=98.0, comments=2000))
    assert a.tier is QualityTier.STRONG
    assert "97.3%" in a.basis                        # 依据里给出下界数值，可核对


def test_jd_same_rate_different_sample_size_differ():
    """同样的 98% 好评，样本量不同 → 档位不同（这正是"必须配评论数一起看"）。"""
    assert assess_quality(jd_offer(rate=98.0, comments=2000)).tier is QualityTier.STRONG
    assert assess_quality(jd_offer(rate=98.0, comments=30)).tier is QualityTier.THIN_SAMPLE


def test_jd_genuinely_low_rate_with_big_sample_is_caution():
    """大样本 + 确实低的好评 → "存疑"，这与"样本不足"是**两种不同的结论**。"""
    a = assess_quality(jd_offer(rate=80.0, comments=5000))
    assert a.tier is QualityTier.CAUTION
    assert assess_quality(jd_offer(rate=100.0, comments=3)).tier is QualityTier.THIN_SAMPLE


def test_jd_rate_without_comments_is_unknown():
    a = assess_quality(jd_offer(rate=98.0, comments=None))
    assert a.tier is QualityTier.UNKNOWN
    assert "评论数" in a.missing


def test_min_samples_boundary():
    """样本门槛的作用是"能否出统计结论"，不是"直接给高分"。

    注意 MIN_SAMPLES=100、98% 好评时下界只有 93.0%（→「一般」）——
    跨过门槛只是**有资格被评估**，要拿「可信」仍需更大样本（如 2000 条 → 97.3%）。
    """
    assert MIN_SAMPLES == 100
    assert assess_quality(jd_offer(rate=98.0, comments=MIN_SAMPLES - 1)).tier is QualityTier.THIN_SAMPLE
    assert assess_quality(jd_offer(rate=98.0, comments=MIN_SAMPLES)).tier is QualityTier.FAIR
    assert assess_quality(jd_offer(rate=98.0, comments=2000)).tier is QualityTier.STRONG


# —— 拼多多路径：两档评分 + 销量 ——


def test_pdd_high_tier_with_sales_is_capped_at_fair():
    """平台只给两档评分，做不出统计推断——上限只能是「一般」，不能宣称"可信"。"""
    a = assess_quality(pdd_offer(tier="高", sales=12000, sales_tip="1.2万+"))
    assert a.tier is QualityTier.FAIR
    assert "两档" in a.basis                         # 诚实说明为什么封顶
    assert a.tier is not QualityTier.STRONG


def test_pdd_high_tier_without_sales_is_thin():
    a = assess_quality(pdd_offer(tier="高", sales=500, sales_tip="500"))
    assert a.tier is QualityTier.THIN_SAMPLE


def test_pdd_low_tier_with_big_sales_is_caution():
    a = assess_quality(pdd_offer(tier="低", sales=50000, sales_tip="5万+"))
    assert a.tier is QualityTier.CAUTION


def test_pdd_missing_sales_is_listed_as_missing():
    a = assess_quality(pdd_offer(tier="高", sales=None, sales_tip=None))
    assert "销量" in a.missing


# —— 缺失一律如实标注，不估算 ——


def test_no_signals_at_all_is_unknown():
    offer = Offer(platform=Platform.PDD, sku_id="x", title="t", list_price=D("10"))
    a = assess_quality(offer)
    assert a.tier is QualityTier.UNKNOWN
    assert a.missing                                 # 缺失清单非空


def test_basis_always_carries_numbers_or_reason():
    """每条结论都要能回溯（PRD 三铁律第 1 条：每个信号有出处）。"""
    for offer in (jd_offer(rate=98.0, comments=2000), pdd_offer(tier="高", sales=12000, sales_tip="1.2万+")):
        assert assess_quality(offer).basis.strip()
