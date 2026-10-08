"""到手价引擎 —— 确定性、纯函数、无 I/O（PRD §5.2「钱不由 LLM 算」）。

口径（PRD §5.1 / §6.1）：
- 仅计算"公开优惠"：优惠券（店铺/平台/品类/会员）+ 满减促销；返佣不计入（合规决策）
- 平行优惠口径：所有门槛按标价（list_price）判定，各项优惠加性叠加
- 金额全程 Decimal，最终价 = max(标价 - 总优惠, 0.01)

叠加规则（StackingPolicy 可配置，随平台规则实测校准）：
- 同类型券互斥（店铺券/平台券各取最优一张）
- exclusion_group 相同的券互斥（跨类型强制二选一）
- 促销：group 相同视为同一活动的多个档位，只取最优一档；不同 group 默认可叠加
- 券组合求解：在互斥约束下穷举子集取总优惠最大（券数量少，精确解可控且可复现）
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Sequence

from pydantic import BaseModel, Field

from priceradar.domain.models import Coupon, Offer, Platform, Promotion, PromotionKind
from priceradar.formatting import fmt_amount

FEN = Decimal("0.01")
MIN_PRICE = Decimal("0.01")
MAX_CANDIDATES = 20  # 参与组合求解的券上限（防御性截断，正常场景远小于此）


class RejectReason(str, Enum):
    """某张券/某个促销未计入到手价的原因（透明展示用）。"""

    THRESHOLD_NOT_MET = "threshold_not_met"                # 未达门槛
    EXPIRED = "expired"                                    # 已过期
    CONFLICT_SAME_TYPE = "conflict_same_type"              # 同类型互斥，已取更优一张
    CONFLICT_EXCLUSION_GROUP = "conflict_exclusion_group"  # 互斥组冲突
    SUPERSEDED = "superseded"                              # 被更优档位/组合取代


class StackingPolicy(BaseModel):
    """叠加策略（可配置、可单测）。"""

    one_per_type: bool = True


DEFAULT_POLICY = StackingPolicy()


class AppliedDiscount(BaseModel):
    kind: str  # "coupon" | "promotion"
    label: str
    amount: Decimal
    source_id: str


class RejectedDiscount(BaseModel):
    label: str
    source_id: str
    reason: RejectReason


class PriceBreakdown(BaseModel):
    """到手价明细：逐项列公式、可核对、可溯源（PRD §4.2 价格区）。"""

    platform: Platform | None = None
    list_price: Decimal
    applied: list[AppliedDiscount] = Field(default_factory=list)
    rejected: list[RejectedDiscount] = Field(default_factory=list)
    total_discount: Decimal = Decimal("0")
    final_price: Decimal = Decimal("0")
    clamped: bool = False
    notes: list[str] = Field(default_factory=list)

    def formula(self) -> str:
        """逐项列公式（人类可读、可核对）。"""
        if not self.applied:
            return f"{fmt_amount(self.list_price)} = {fmt_amount(self.final_price)}"
        parts = " - ".join(f"{a.label} {fmt_amount(a.amount)}" for a in self.applied)
        return f"{fmt_amount(self.list_price)} - {parts} = {fmt_amount(self.final_price)}"


def _coupon_reject(coupon: Coupon, price: Decimal, as_of: date) -> RejectReason | None:
    if coupon.valid_until is not None and coupon.valid_until < as_of:
        return RejectReason.EXPIRED
    if coupon.threshold > price:
        return RejectReason.THRESHOLD_NOT_MET
    return None


def _promotion_amount(promotion: Promotion, price: Decimal) -> Decimal:
    if promotion.kind is PromotionKind.EVERY_N:
        times = int(price // promotion.threshold)
        return (promotion.reduction * times).quantize(FEN, ROUND_HALF_UP)
    # STEP：达标只减一次；多档位由 group 机制选出最优一档
    if price >= promotion.threshold:
        return promotion.reduction.quantize(FEN, ROUND_HALF_UP)
    return Decimal("0")


def _conflicts(a: Coupon, b: Coupon, policy: StackingPolicy) -> bool:
    if a.exclusion_group is not None and a.exclusion_group == b.exclusion_group:
        return True
    return policy.one_per_type and a.type == b.type


def _solve_coupons(
    candidates: list[Coupon],
    pre_rejected: list[RejectedDiscount],
    policy: StackingPolicy,
) -> tuple[Decimal, list[AppliedDiscount], list[RejectedDiscount], list[str]]:
    notes: list[str] = []
    if len(candidates) > MAX_CANDIDATES:
        candidates = sorted(candidates, key=lambda c: (-c.face_value, c.coupon_id))[:MAX_CANDIDATES]
        notes.append(f"候选券过多，已按面额保留前 {MAX_CANDIDATES} 张参与组合求解")

    ordered = sorted(candidates, key=lambda c: (-c.face_value, c.coupon_id))
    best: list[Coupon] = []
    best_sum = Decimal("-1")
    chosen: list[Coupon] = []

    def dfs(index: int, total: Decimal) -> None:
        nonlocal best, best_sum
        if index == len(ordered):
            if total > best_sum or (total == best_sum and len(chosen) < len(best)):
                best = list(chosen)
                best_sum = total
            return
        candidate = ordered[index]
        if not any(_conflicts(candidate, x, policy) for x in chosen):
            chosen.append(candidate)
            dfs(index + 1, total + candidate.face_value)
            chosen.pop()
        dfs(index + 1, total)

    dfs(0, Decimal("0"))

    selected_ids = {c.coupon_id for c in best}
    applied = [
        AppliedDiscount(
            kind="coupon",
            label=c.label(),
            amount=c.face_value.quantize(FEN, ROUND_HALF_UP),
            source_id=c.coupon_id,
        )
        for c in sorted(best, key=lambda c: (-c.face_value, c.coupon_id))
    ]

    rejected = list(pre_rejected)
    for coupon in ordered:
        if coupon.coupon_id in selected_ids:
            continue
        reason = RejectReason.SUPERSEDED
        for selected in best:
            if coupon.exclusion_group is not None and coupon.exclusion_group == selected.exclusion_group:
                reason = RejectReason.CONFLICT_EXCLUSION_GROUP
                break
            if policy.one_per_type and coupon.type == selected.type:
                reason = RejectReason.CONFLICT_SAME_TYPE
                break
        rejected.append(RejectedDiscount(label=coupon.label(), source_id=coupon.coupon_id, reason=reason))

    total = max(best_sum, Decimal("0")).quantize(FEN, ROUND_HALF_UP)
    return total, applied, rejected, notes


def _solve_promotions(
    promotions: Sequence[Promotion],
    price: Decimal,
    as_of: date,
) -> tuple[Decimal, list[AppliedDiscount], list[RejectedDiscount]]:
    groups: dict[str, list[Promotion]] = {}
    for promotion in promotions:
        key = promotion.group or f"__solo__{promotion.promotion_id}"
        groups.setdefault(key, []).append(promotion)

    applied: list[AppliedDiscount] = []
    rejected: list[RejectedDiscount] = []
    total = Decimal("0")

    for key in sorted(groups):
        scored: list[tuple[Decimal, Promotion]] = []
        for promotion in groups[key]:
            if promotion.valid_until is not None and promotion.valid_until < as_of:
                rejected.append(RejectedDiscount(label=promotion.label(), source_id=promotion.promotion_id, reason=RejectReason.EXPIRED))
                continue
            scored.append((_promotion_amount(promotion, price), promotion))
        if not scored:
            continue
        scored.sort(key=lambda item: (-item[0], item[1].promotion_id))
        best_amount, best_promotion = scored[0]
        if best_amount <= 0:
            for _, promotion in scored:
                rejected.append(RejectedDiscount(label=promotion.label(), source_id=promotion.promotion_id, reason=RejectReason.THRESHOLD_NOT_MET))
            continue
        total += best_amount
        applied.append(AppliedDiscount(kind="promotion", label=best_promotion.label(), amount=best_amount, source_id=best_promotion.promotion_id))
        for amount, promotion in scored[1:]:
            reason = RejectReason.THRESHOLD_NOT_MET if amount <= 0 else RejectReason.SUPERSEDED
            rejected.append(RejectedDiscount(label=promotion.label(), source_id=promotion.promotion_id, reason=reason))

    return total.quantize(FEN, ROUND_HALF_UP), applied, rejected


def compute_price(
    offer: Offer,
    as_of: date | None = None,
    policy: StackingPolicy = DEFAULT_POLICY,
) -> PriceBreakdown:
    """计算单个报价的公开优惠口径到手价（纯函数，无 I/O）。"""
    as_of = as_of or date.today()
    price = offer.list_price

    promotion_total, promotion_applied, promotion_rejected = _solve_promotions(offer.promotions, price, as_of)

    candidates: list[Coupon] = []
    pre_rejected: list[RejectedDiscount] = []
    for coupon in offer.coupons:
        reason = _coupon_reject(coupon, price, as_of)
        if reason is None:
            candidates.append(coupon)
        else:
            pre_rejected.append(RejectedDiscount(label=coupon.label(), source_id=coupon.coupon_id, reason=reason))

    coupon_total, coupon_applied, coupon_rejected, notes = _solve_coupons(candidates, pre_rejected, policy)

    total_discount = (coupon_total + promotion_total).quantize(FEN, ROUND_HALF_UP)
    raw_final = price - total_discount
    clamped = raw_final < MIN_PRICE
    final_price = max(raw_final, MIN_PRICE).quantize(FEN, ROUND_HALF_UP)

    return PriceBreakdown(
        platform=offer.platform,
        list_price=price,
        applied=coupon_applied + promotion_applied,
        rejected=coupon_rejected + promotion_rejected,
        total_discount=total_discount,
        final_price=final_price,
        clamped=clamped,
        notes=notes,
    )
