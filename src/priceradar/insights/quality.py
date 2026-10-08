"""质量信号的可信度评估（PRD §5.8 信号卡三铁律第 3 条）。

**要解决的问题**：「100% 好评 / 3 条评论」在朴素比较里强于「97% 好评 / 2 万条评论」，
但统计上前者的"满分"几乎没有信息量——小样本的高分不可信。
用 **Wilson 置信区间下界** 衡量：样本越少，下界被拉得越低。

    Wilson 下界(100%, n=3)    ≈ 0.44
    Wilson 下界( 98%, n=2000) ≈ 0.973
    Wilson 下界( 97%, n=20000)≈ 0.968

⚠️ 刻意**不做"综合 X 分"**——PRD §5.8 输出形态明令禁止伪科学评分。
本模块只产出「档位 + 依据数字 + 缺失清单」，每条结论都能回溯到具体信号。

⚠️ 平台差异是真实的、且不隐藏：
- 京东给好评率 + 评论数 → 可做统计推断
- 拼多多只给**两档**店铺评分（高/低）+ 模糊销量 → 上限只能到「一般」
  （不是拼多多商品差，是**我们能拿到的信号粗**，必须如实说明）
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field

from priceradar.domain.models import Offer

# —— 阈值（可配置、可解释）——

MIN_SAMPLES = 100          # 评论数低于此视为"样本不足"，不出统计结论
WILSON_Z = 1.96            # 95% 置信度
STRONG_FLOOR = 0.95        # 可信下界 ≥ 此值 → 可信
FAIR_FLOOR = 0.90          # 可信下界 ≥ 此值 → 一般；低于 → 存疑

PDD_SALES_FLOOR = 1000     # 拼多多侧"样本量"下限（累计销量估算）


class QualityTier(str, Enum):
    STRONG = "strong"            # 可信：大样本 + 高评价
    FAIR = "fair"                # 一般
    CAUTION = "caution"          # 存疑：有大样本证据提示评价偏低
    THIN_SAMPLE = "thin_sample"  # 样本不足：有数据但太少，无法判断
    UNKNOWN = "unknown"          # 无可用信号

    @property
    def label(self) -> str:
        return _TIER_LABELS[self]


_TIER_LABELS = {
    QualityTier.STRONG: "可信",
    QualityTier.FAIR: "一般",
    QualityTier.CAUTION: "存疑",
    QualityTier.THIN_SAMPLE: "样本不足",
    QualityTier.UNKNOWN: "数据不足",
}

# 需要提示用户的档位（前端据此上色）；THIN/UNKNOWN 用中性样式
ALERT_TIERS = frozenset({QualityTier.CAUTION, QualityTier.THIN_SAMPLE})


class QualityAssessment(BaseModel):
    """单个报价的质量可信度评估（结论 + 依据 + 缺失，不含综合分）。"""

    tier: QualityTier
    label: str = Field(description="展示用中文档位")
    basis: str = Field(description="依据说明，含具体数字，可核对")
    missing: list[str] = Field(default_factory=list, description="缺失的信号清单")


def wilson_lower_bound(rate: float, n: int, z: float = WILSON_Z) -> float:
    """好评率的 Wilson 置信区间下界（默认 95%）。

    ⚠️ `rate` 是**好评率（0~1）**，不是成功次数——签名刻意收"率"，
    因为误传次数不会报错、只会静默算出错误结果（本函数初版就栽在这上面，
    把 0.98 当成 0.98 次成功除以 2000 条 → 下界算成 0）。

    比"直接看好评率"更诚实的点：小样本时下界显著低于点估计，
    从而避免「3 条评论 100% 好评」被当成满分。
    """
    if n <= 0:
        return 0.0
    phat = min(max(rate, 0.0), 1.0)
    z2 = z * z
    denominator = 1 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    return max(0.0, (center - margin) / denominator)


def _assess_jd(offer: Offer) -> QualityAssessment | None:
    """京东：好评率 + 评论数 → Wilson 下界。"""
    signals = offer.signals
    if signals is None or signals.good_comments_share is None:
        return None
    rate = float(signals.good_comments_share)
    n = int(signals.comments or 0)
    if n <= 0:
        return QualityAssessment(
            tier=QualityTier.UNKNOWN,
            label=QualityTier.UNKNOWN.label,
            basis=f"好评率 {rate:g}% 但无评论数，无法判断可信度",
            missing=["评论数"],
        )
    lower = wilson_lower_bound(rate / 100, n)
    if n < MIN_SAMPLES:
        return QualityAssessment(
            tier=QualityTier.THIN_SAMPLE,
            label=QualityTier.THIN_SAMPLE.label,
            basis=f"好评率 {rate:g}% · 仅 {n} 条评论 → 样本不足，无法判断（下界 {lower:.0%}）",
            missing=[],
        )
    if lower >= STRONG_FLOOR:
        tier = QualityTier.STRONG
    elif lower >= FAIR_FLOOR:
        tier = QualityTier.FAIR
    else:
        tier = QualityTier.CAUTION
    return QualityAssessment(
        tier=tier,
        label=tier.label,
        basis=f"好评率 {rate:g}% · {n} 条评论 → 可信下界 {lower:.1%}",
        missing=[],
    )


def _assess_pdd(offer: Offer) -> QualityAssessment | None:
    """拼多多：店铺评分档位（两档）+ 销量估算。

    平台只给两档评分，**做不出统计推断**——因此上限只能是「一般」，
    不能因为一个"高"字就宣称"可信"。
    """
    signals = offer.signals
    if signals is None or signals.shop_score_tier is None:
        return None
    tier_text = signals.shop_score_tier
    sales = signals.sales_estimate
    sales_desc = signals.sales_tip or (f"{sales}" if sales else None)
    missing = [] if sales is not None else ["销量"]

    if tier_text == "高":
        if sales is not None and sales >= PDD_SALES_FLOOR:
            return QualityAssessment(
                tier=QualityTier.FAIR,
                label=QualityTier.FAIR.label,
                basis=f"店铺评分「高」· 已售 {sales_desc} → 平台仅提供两档评分，上限为「一般」",
                missing=missing,
            )
        return QualityAssessment(
            tier=QualityTier.THIN_SAMPLE,
            label=QualityTier.THIN_SAMPLE.label,
            basis=f"店铺评分「高」但销量证据不足（已售 {sales_desc or '缺失'}）→ 无法判断",
            missing=missing,
        )
    if sales is not None and sales >= PDD_SALES_FLOOR:
        return QualityAssessment(
            tier=QualityTier.CAUTION,
            label=QualityTier.CAUTION.label,
            basis=f"店铺评分「低」· 已售 {sales_desc} → 大样本下评价偏低",
            missing=missing,
        )
    return QualityAssessment(
        tier=QualityTier.THIN_SAMPLE,
        label=QualityTier.THIN_SAMPLE.label,
        basis=f"店铺评分「低」且销量少（已售 {sales_desc or '缺失'}）→ 证据不足",
        missing=missing,
    )


def assess_quality(offer: Offer) -> QualityAssessment:
    """评估一个报价的质量可信度（缺失一律如实标注，不估算）。"""
    assessment = _assess_jd(offer) or _assess_pdd(offer)
    if assessment is not None:
        return assessment
    return QualityAssessment(
        tier=QualityTier.UNKNOWN,
        label=QualityTier.UNKNOWN.label,
        basis="无可用的质量信号（该平台不提供评价类字段）",
        missing=["好评率", "评论数", "店铺评分"],
    )
