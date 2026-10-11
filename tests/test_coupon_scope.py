"""券的适用范围判定（纯函数，全离线）。

这个模块的存在理由是一条实测 bug：补录券原先只按平台过滤，一张「A 店铺」的券
会把 B 店铺的商品也打折。判定表里每一行都对应一种"没有依据时该怎么办"。
"""

from decimal import Decimal

import pytest

from priceradar.coupon_scope import (
    ScopeReject,
    coupon_scope_reject,
    coupons_for_offer,
    normalize_shop_name,
)
from priceradar.domain.models import Coupon, CouponType, Offer, Platform

D = Decimal


def coupon(kind: CouponType, **kwargs) -> Coupon:
    return Coupon(coupon_id="c1", type=kind, face_value=D("10"), threshold=D("50"), source="screenshot", **kwargs)


def offer(**kwargs) -> Offer:
    return Offer(platform=Platform.PDD, sku_id="a", title="商品", list_price=D("100"), **kwargs)


# ---------- 归一化 ----------

def test_normalize_handles_fullwidth_and_whitespace():
    """京东店名常带全角括号（如 `苏泊尔（SUPOR）热水器…`），全角半角要能对上。"""
    assert normalize_shop_name("京造（SUPOR）旗舰店") == normalize_shop_name("京造(SUPOR)旗舰店")
    assert normalize_shop_name(" A B 店 ") == normalize_shop_name("ab店")
    assert normalize_shop_name("QCY官方店") == normalize_shop_name("qcy官方店")
    assert normalize_shop_name(None) == ""


def test_normalize_does_not_strip_prefixes():
    """刻意不剥离「京东自营」这类前缀——剥离属于猜测，会把两家不同的店并成一家。"""
    assert normalize_shop_name("苏泊尔热水器京东自营旗舰店") != normalize_shop_name("苏泊尔旗舰店")


# ---------- 平台券 / 会员券：全平台适用 ----------

@pytest.mark.parametrize("kind", [CouponType.PLATFORM, CouponType.MEMBER])
def test_platform_wide_coupons_ignore_shop_and_category(kind):
    assert coupon_scope_reject(coupon(kind), offer()) is None
    assert coupon_scope_reject(coupon(kind), offer(shop_id="9", shop_name="别家店")) is None


# ---------- 店铺券 ----------

def test_shop_coupon_applies_when_ids_match():
    assert coupon_scope_reject(coupon(CouponType.SHOP, shop_id="888"), offer(shop_id="888")) is None


def test_shop_coupon_rejected_when_ids_differ():
    """这条就是被修掉的那个 bug：A 店铺的券不能打到 B 店铺。"""
    assert (
        coupon_scope_reject(coupon(CouponType.SHOP, shop_id="888"), offer(shop_id="999"))
        is ScopeReject.SHOP_MISMATCH
    )


def test_shop_id_takes_precedence_over_name():
    """两边都有 id 时只看 id——店名可能被平台改写，id 才是权威。"""
    assert coupon_scope_reject(
        coupon(CouponType.SHOP, shop_id="888", shop_name="旧店名"), offer(shop_id="888", shop_name="新店名")
    ) is None


def test_shop_coupon_falls_back_to_name_when_ids_missing():
    """拼多多没有店铺 id（实测 `mall_id` 连键都不存在），只能靠店名。"""
    assert coupon_scope_reject(
        coupon(CouponType.SHOP, shop_name="A店铺"), offer(shop_name="A店铺")
    ) is None
    assert (
        coupon_scope_reject(coupon(CouponType.SHOP, shop_name="A店铺"), offer(shop_name="B店铺"))
        is ScopeReject.SHOP_MISMATCH
    )


def test_name_match_is_exact_not_substring():
    """**绝不能用包含匹配**——`A店铺` 是 `A店铺旗舰店` 的子串，但它们是两家店。"""
    assert (
        coupon_scope_reject(coupon(CouponType.SHOP, shop_name="A店铺"), offer(shop_name="A店铺旗舰店"))
        is ScopeReject.SHOP_MISMATCH
    )


def test_unbound_shop_coupon_is_rejected():
    assert coupon_scope_reject(coupon(CouponType.SHOP), offer(shop_id="888")) is ScopeReject.SHOP_NOT_BOUND


def test_offer_without_any_shop_identity_is_rejected():
    """拼多多 87% 的商品没有店铺名，且没有店铺 id —— 券一律不套用（不是默认全平台）。"""
    assert coupon_scope_reject(coupon(CouponType.SHOP, shop_id="888"), offer()) is ScopeReject.SHOP_UNKNOWN


def test_mismatched_key_types_are_rejected_not_guessed():
    """券只有 id、商品只有名字——无从比对，拒绝而不是猜。"""
    assert (
        coupon_scope_reject(coupon(CouponType.SHOP, shop_id="888"), offer(shop_name="A店铺"))
        is ScopeReject.SHOP_KEY_UNMATCHED
    )


def test_collect_coupon_follows_shop_rules():
    """店铺收藏券同样是店铺范围。"""
    assert coupon_scope_reject(coupon(CouponType.COLLECT, shop_id="888"), offer(shop_id="888")) is None
    assert (
        coupon_scope_reject(coupon(CouponType.COLLECT, shop_id="888"), offer(shop_id="999"))
        is ScopeReject.SHOP_MISMATCH
    )


# ---------- 品类券 ----------

def test_category_coupon_matches_on_leaf():
    assert coupon_scope_reject(
        coupon(CouponType.CATEGORY, category_ids=["5752", "5826", "19146"]),
        offer(category_ids=["5752", "5826", "19146"]),
    ) is None


def test_category_coupon_does_not_match_on_shared_ancestor():
    """只比链尾：共享「手机」这一级**不算**匹配。

    用整条链相交会让「耳机」的券匹配到所有手机——那是假阳性，正是要避免的方向。
    """
    assert (
        coupon_scope_reject(
            coupon(CouponType.CATEGORY, category_ids=["5752", "5826", "19146"]),
            offer(category_ids=["5752", "5826", "99999"]),
        )
        is ScopeReject.CATEGORY_MISMATCH
    )


def test_unbound_category_coupon_is_rejected():
    assert coupon_scope_reject(coupon(CouponType.CATEGORY), offer(category_ids=["1"])) is ScopeReject.CATEGORY_NOT_BOUND


def test_category_coupon_on_offer_without_category_is_rejected():
    assert (
        coupon_scope_reject(coupon(CouponType.CATEGORY, category_ids=["1"]), offer())
        is ScopeReject.CATEGORY_UNKNOWN
    )


# ---------- 分批 ----------

def test_coupons_for_offer_splits_and_reports_reasons():
    """不适用的券不能被静默丢掉——"我补录了券怎么没生效"要能在界面上找到答案。"""
    coupons = [
        Coupon(coupon_id="p1", type=CouponType.PLATFORM, face_value=D("5")),
        Coupon(coupon_id="s1", type=CouponType.SHOP, face_value=D("10"), shop_id="888", source="screenshot"),
        Coupon(coupon_id="s2", type=CouponType.SHOP, face_value=D("10"), shop_id="999", source="screenshot"),
        Coupon(coupon_id="s3", type=CouponType.SHOP, face_value=D("10"), source="screenshot"),
    ]
    applicable, rejected = coupons_for_offer(coupons, offer(shop_id="888"))
    assert [c.coupon_id for c in applicable] == ["p1", "s1"]
    assert [(c.coupon_id, reason) for c, reason in rejected] == [
        ("s2", ScopeReject.SHOP_MISMATCH),
        ("s3", ScopeReject.SHOP_NOT_BOUND),
    ]


def test_every_reject_reason_has_a_human_label():
    """拒绝原因要能对用户说人话——漏一个标签就会在界面上露出英文枚举。"""
    from priceradar.coupon_scope import SCOPE_REJECT_LABELS

    assert set(SCOPE_REJECT_LABELS) == set(ScopeReject)
