"""价格对比卡渲染（确定性部分）。

卡片结构与数字全部确定性生成；LLM 只负责最终文案润色（PRD §5.2）。
"""

from __future__ import annotations

from typing import Sequence

from priceradar.domain.models import Offer, Platform
from priceradar.engine.pricing import PriceBreakdown, RejectReason
from priceradar.formatting import fmt_amount

PLATFORM_LABELS = {
    Platform.JD: "京东",
    Platform.PDD: "拼多多",
    Platform.TAOBAO: "淘宝",
}

_REASON_LABELS = {
    RejectReason.THRESHOLD_NOT_MET: "未达门槛",
    RejectReason.EXPIRED: "已过期",
    RejectReason.CONFLICT_SAME_TYPE: "同类型仅取最优一张",
    RejectReason.CONFLICT_EXCLUSION_GROUP: "与已用券互斥",
    RejectReason.SUPERSEDED: "存在更优组合",
}

REASON_LABELS = _REASON_LABELS  # 对外公开（服务层 / 前端复用）

_DISCLAIMER = "※ 基于平台公开优惠测算，未含个性化券；价格实时变动，以结算页为准。"


def render_price_card(
    items: Sequence[tuple[Offer, PriceBreakdown]],
    *,
    count_by_sku: dict[str, int] | None = None,
) -> str:
    """把各平台报价与明细渲染成可读的对比卡（终端/演示用）。

    count_by_sku：同款折叠组内店铺数（>1 时在标题上标注"同款 ×N 家"）。
    """
    ranked = sorted(items, key=lambda item: (item[1].final_price, item[0].platform.value))
    lines: list[str] = ["【到手价对比 · 公开优惠口径】", "─" * 46]
    for rank, (offer, breakdown) in enumerate(ranked, start=1):
        headline = f"{rank}. {PLATFORM_LABELS[offer.platform]}  ￥{fmt_amount(breakdown.final_price)}"
        if breakdown.total_discount > 0:
            headline += f"（已省 ￥{fmt_amount(breakdown.total_discount)}）"
        if count_by_sku and count_by_sku.get(offer.sku_id, 0) > 1:
            headline += f"（同款 ×{count_by_sku[offer.sku_id]} 家）"
        lines.append(headline)
        lines.append(f"   {offer.shop_name or offer.title}")
        lines.append(f"   {breakdown.formula()}")
        for rejected in breakdown.rejected:
            lines.append(f"   △ 未计入：{rejected.label}（{_REASON_LABELS[rejected.reason]}）")
        for note in breakdown.notes:
            lines.append(f"   ※ {note}")
    lines.append("─" * 46)
    lines.append(_DISCLAIMER)
    return "\n".join(lines)
