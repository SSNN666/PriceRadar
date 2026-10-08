"""到手价引擎测试：门槛 / 最优券组合 / 互斥 / 满减 / 边界 / 精度 / 确定性。"""

from datetime import date
from decimal import Decimal

from priceradar.domain.models import (
    Coupon,
    CouponType,
    Offer,
    Platform,
    Promotion,
    PromotionKind,
)
from priceradar.engine.pricing import RejectReason, compute_price

D = Decimal
AS_OF = date(2026, 10, 5)


def make_coupon(coupon_id="c1", type_=CouponType.SHOP, face="30", threshold="300", **kwargs) -> Coupon:
    return Coupon(coupon_id=coupon_id, type=type_, face_value=D(face), threshold=D(threshold), **kwargs)


def make_promotion(promotion_id="p1", kind=PromotionKind.EVERY_N, threshold="300", reduction="50", **kwargs) -> Promotion:
    return Promotion(promotion_id=promotion_id, kind=kind, threshold=D(threshold), reduction=D(reduction), **kwargs)


def make_offer(price="399", coupons=(), promotions=(), platform=Platform.JD) -> Offer:
    return Offer(
        platform=platform,
        sku_id="sku-1",
        title="测试商品",
        list_price=D(price),
        coupons=list(coupons),
        promotions=list(promotions),
    )


# ---------- 门槛判定 ----------

def test_no_discounts_keeps_price():
    breakdown = compute_price(make_offer("399"), as_of=AS_OF)
    assert breakdown.final_price == D("399")
    assert breakdown.total_discount == D("0")
    assert breakdown.clamped is False
    assert breakdown.applied == [] and breakdown.rejected == []


def test_coupon_applies_at_exact_threshold():
    breakdown = compute_price(make_offer("300", [make_coupon(threshold="300", face="30")]), as_of=AS_OF)
    assert breakdown.final_price == D("270")


def test_coupon_below_threshold_is_rejected():
    breakdown = compute_price(make_offer("299", [make_coupon(threshold="300")]), as_of=AS_OF)
    assert breakdown.final_price == D("299")
    assert [r.reason for r in breakdown.rejected] == [RejectReason.THRESHOLD_NOT_MET]


def test_zero_threshold_coupon_always_applies():
    breakdown = compute_price(make_offer("50", [make_coupon(threshold="0", face="10")]), as_of=AS_OF)
    assert breakdown.final_price == D("40")


def test_coupon_threshold_is_judged_on_list_price():
    # 促销后为 500，但券门槛 550 按标价 600 判定 → 仍可使用（平行优惠口径）
    offer = make_offer("600", [make_coupon(threshold="550", face="20")], [make_promotion(threshold="300", reduction="50")])
    breakdown = compute_price(offer, as_of=AS_OF)
    assert breakdown.total_discount == D("120")  # 促销 100 + 券 20


# ---------- 最优券组合与互斥 ----------

def test_same_type_picks_largest():
    offer = make_offer("400", [make_coupon("c1", face="30"), make_coupon("c2", face="50")])
    breakdown = compute_price(offer, as_of=AS_OF)
    assert breakdown.total_discount == D("50")
    assert breakdown.applied[0].source_id == "c2"
    assert [r.reason for r in breakdown.rejected] == [RejectReason.CONFLICT_SAME_TYPE]


def test_different_types_stack():
    offer = make_offer("400", [
        make_coupon("c1", type_=CouponType.SHOP, face="30"),
        make_coupon("c2", type_=CouponType.PLATFORM, face="40"),
    ])
    breakdown = compute_price(offer, as_of=AS_OF)
    assert breakdown.total_discount == D("70")


def test_exclusion_group_forbids_stacking():
    offer = make_offer("400", [
        make_coupon("c1", type_=CouponType.SHOP, face="30", exclusion_group="g1"),
        make_coupon("c2", type_=CouponType.PLATFORM, face="40", exclusion_group="g1"),
    ])
    breakdown = compute_price(offer, as_of=AS_OF)
    assert breakdown.total_discount == D("40")
    assert any(r.reason == RejectReason.CONFLICT_EXCLUSION_GROUP for r in breakdown.rejected)


def test_mixed_constraints_pick_global_optimum():
    # 店铺 30（可叠）+ 平台 40（互斥组 g1）+ 品类 25（互斥组 g1）→ 最优 30+40
    offer = make_offer("500", [
        make_coupon("c1", type_=CouponType.SHOP, face="30"),
        make_coupon("c2", type_=CouponType.PLATFORM, face="40", exclusion_group="g1"),
        make_coupon("c3", type_=CouponType.CATEGORY, face="25", exclusion_group="g1"),
    ])
    breakdown = compute_price(offer, as_of=AS_OF)
    assert breakdown.total_discount == D("70")


def test_tie_between_equal_coupons_is_stable():
    offer = make_offer("400", [make_coupon("c1", face="30"), make_coupon("c2", face="30")])
    breakdown = compute_price(offer, as_of=AS_OF)
    assert breakdown.applied[0].source_id == "c1"


# ---------- 每满减 ----------

def test_every_n_exact_multiple():
    breakdown = compute_price(make_offer("600", promotions=[make_promotion(threshold="300", reduction="50")]), as_of=AS_OF)
    assert breakdown.total_discount == D("100")
    assert breakdown.final_price == D("500")


def test_every_n_remainder_counts_floor():
    breakdown = compute_price(make_offer("599", promotions=[make_promotion(threshold="300", reduction="50")]), as_of=AS_OF)
    assert breakdown.total_discount == D("50")


def test_every_n_below_step_is_rejected():
    breakdown = compute_price(make_offer("299", promotions=[make_promotion(threshold="300", reduction="50")]), as_of=AS_OF)
    assert breakdown.total_discount == D("0")
    assert [r.reason for r in breakdown.rejected] == [RejectReason.THRESHOLD_NOT_MET]


# ---------- 阶梯满减（同活动多档位） ----------

def test_step_promotion_picks_highest_met_tier():
    promotions = [
        make_promotion("t1", PromotionKind.STEP, "300", "50", group="act"),
        make_promotion("t2", PromotionKind.STEP, "600", "120", group="act"),
    ]
    breakdown = compute_price(make_offer("650", promotions=promotions), as_of=AS_OF)
    assert breakdown.total_discount == D("120")
    assert any(r.reason == RejectReason.SUPERSEDED for r in breakdown.rejected)


def test_step_promotion_low_tier_only():
    promotions = [
        make_promotion("t1", PromotionKind.STEP, "300", "50", group="act"),
        make_promotion("t2", PromotionKind.STEP, "600", "120", group="act"),
    ]
    breakdown = compute_price(make_offer("300", promotions=promotions), as_of=AS_OF)
    assert breakdown.total_discount == D("50")


def test_step_promotion_below_all_tiers():
    promotions = [make_promotion("t1", PromotionKind.STEP, "300", "50", group="act")]
    breakdown = compute_price(make_offer("299", promotions=promotions), as_of=AS_OF)
    assert breakdown.total_discount == D("0")
    assert breakdown.rejected[0].reason == RejectReason.THRESHOLD_NOT_MET


def test_promotions_in_different_groups_stack():
    promotions = [
        make_promotion("p1", PromotionKind.STEP, "300", "50", group="a"),
        make_promotion("p2", PromotionKind.STEP, "100", "10", group="b"),
    ]
    breakdown = compute_price(make_offer("400", promotions=promotions), as_of=AS_OF)
    assert breakdown.total_discount == D("60")


# ---------- 券 + 满减组合 ----------

def test_coupon_and_promotion_combine():
    offer = make_offer("400", [make_coupon(threshold="300", face="30")], [make_promotion(threshold="300", reduction="20")])
    breakdown = compute_price(offer, as_of=AS_OF)
    assert breakdown.total_discount == D("50")
    assert breakdown.final_price == D("350")


# ---------- 有效期 ----------

def test_expired_coupon_is_rejected():
    coupon = make_coupon(valid_until=date(2026, 10, 4))
    breakdown = compute_price(make_offer("400", [coupon]), as_of=date(2026, 10, 5))
    assert breakdown.rejected[0].reason == RejectReason.EXPIRED


def test_valid_until_is_inclusive():
    coupon = make_coupon(valid_until=date(2026, 10, 5))
    breakdown = compute_price(make_offer("400", [coupon]), as_of=date(2026, 10, 5))
    assert breakdown.total_discount == D("30")


def test_expired_promotion_is_rejected():
    promotion = make_promotion(valid_until=date(2026, 10, 1))
    breakdown = compute_price(make_offer("400", promotions=[promotion]), as_of=date(2026, 10, 5))
    assert breakdown.rejected[0].reason == RejectReason.EXPIRED


# ---------- 边界与精度 ----------

def test_final_price_never_below_one_fen():
    breakdown = compute_price(make_offer("10", [make_coupon(threshold="0", face="50")]), as_of=AS_OF)
    assert breakdown.clamped is True
    assert breakdown.final_price == D("0.01")


def test_decimal_precision_is_exact():
    breakdown = compute_price(make_offer("10.10", [make_coupon(threshold="0", face="0.20")]), as_of=AS_OF)
    assert breakdown.final_price == D("9.90")


def test_too_many_coupons_is_truncated_with_note():
    coupons = [make_coupon(f"c{i}", face=str(100 + i)) for i in range(30)]
    breakdown = compute_price(make_offer("5000", coupons), as_of=AS_OF)
    assert breakdown.total_discount == D("129")  # 面额最大的 c29 仍在保留集合内
    assert breakdown.notes and "保留前" in breakdown.notes[0]
    assert len(breakdown.applied) == 1


def test_repeated_compute_is_deterministic():
    offer = make_offer("400", [make_coupon("c1"), make_coupon("c2", type_=CouponType.PLATFORM)], [make_promotion()])
    assert compute_price(offer, as_of=AS_OF) == compute_price(offer, as_of=AS_OF)


def test_formula_lists_each_discount():
    breakdown = compute_price(make_offer("300", [make_coupon(threshold="300", face="30")]), as_of=AS_OF)
    formula = breakdown.formula()
    assert "300" in formula and "减30" in formula and "270" in formula
