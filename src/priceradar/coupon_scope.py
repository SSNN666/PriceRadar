"""券的适用范围判定（纯函数，零 I/O）。

## 为什么需要这个模块

补录券原先只按**平台**过滤就并入到手价——于是一张「A 店铺」的券会把 B 店铺的商品
也打折（2026-10-10 实测复现）。`Coupon.scope_note` 记了适用范围，但注释写着
「展示用」，从来没被用来判定。**记了范围却不判范围**，就是价格算错。

## 边界（很重要）

本模块**只判用户补录券**（`user_coupons` 表里的）。平台 API 直接返回、挂在
`Offer.coupons` 上的券（如拼多多的 `pdd-{goods_id}-coupon`）**本来就适用于该商品**，
不要拿它们过这道过滤——它们的 `source == "api"`，归属已经由来源保证。

## 判定规则

| 券型 | 规则 |
|---|---|
| `PLATFORM` / `MEMBER` | 全平台，直接适用（不看店铺/品类） |
| `SHOP` / `COLLECT` | 两边都有 `shop_id` → **全等**才适用；否则两边都有 `shop_name` → 归一化全等 |
| `CATEGORY` | 两边都有品类链 → **最细一级**（链尾）全等才适用 |

**没有匹配依据时一律不适用**（并给出拒绝码）。这是本模块存在的意义：
从「默认全平台套用」改成「缺席」。宁可少算一张券，不能算错一笔价。
"""

from __future__ import annotations

import unicodedata
from enum import Enum

from priceradar.domain.models import Coupon, CouponType, Offer

# 全平台券：本就无店铺/品类限制
_PLATFORM_WIDE = frozenset({CouponType.PLATFORM, CouponType.MEMBER})
# 店铺范围券：需要店铺标识才能判定
_SHOP_SCOPED = frozenset({CouponType.SHOP, CouponType.COLLECT})


class ScopeReject(str, Enum):
    """券**不适用**于该商品的原因（每一种都要能对用户说清楚）。"""

    SHOP_MISMATCH = "shop_mismatch"            # 双方都有店铺信息，但对不上
    SHOP_NOT_BOUND = "shop_not_bound"          # 券没绑定店铺
    SHOP_UNKNOWN = "shop_unknown"              # 商品没提供任何店铺标识
    SHOP_KEY_UNMATCHED = "shop_key_unmatched"  # 一边只有 id、另一边只有名字，无从比对
    CATEGORY_NOT_BOUND = "category_not_bound"  # 券没绑定品类
    CATEGORY_UNKNOWN = "category_unknown"      # 商品没有品类信息
    CATEGORY_MISMATCH = "category_mismatch"    # 品类对不上


SCOPE_REJECT_LABELS: dict[ScopeReject, str] = {
    ScopeReject.SHOP_MISMATCH: "不是这家店的券",
    ScopeReject.SHOP_NOT_BOUND: "这张券还没绑定店铺，未计入",
    ScopeReject.SHOP_UNKNOWN: "该商品没有店铺信息，无法判断券是否适用，未计入",
    ScopeReject.SHOP_KEY_UNMATCHED: "券与商品的店铺信息对不上（一边只有 id），未计入",
    ScopeReject.CATEGORY_NOT_BOUND: "这张品类券还没绑定品类，未计入",
    ScopeReject.CATEGORY_UNKNOWN: "该商品没有品类信息，无法判断券是否适用，未计入",
    ScopeReject.CATEGORY_MISMATCH: "不是这个品类的券",
}


def normalize_shop_name(value: str | None) -> str:
    """店铺名归一化：NFKC（统一全角括号/数字/字母）→ 去所有空白 → casefold。

    **只做无损处理**，比较时用全等——绝不用子串/包含匹配，那正是"店铺券打到别家店"的
    事故形态。京东店名常带前缀（如 `苏泊尔（SUPOR）热水器京东自营旗舰店`），
    **刻意不剥离**：剥离属于猜测，`XX官方旗舰店` 与 `XX家电旗舰店` 不能合并。

    唯一接受的副作用：`"A B店"` 与 `"AB店"` 视为同一家。
    """
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(normalized.split()).casefold()


def _shop_reject(coupon: Coupon, offer: Offer) -> ScopeReject | None:
    """店铺范围券的判定。返回 None 表示适用。"""
    coupon_id, coupon_name = coupon.shop_id, normalize_shop_name(coupon.shop_name)
    offer_id, offer_name = offer.shop_id, normalize_shop_name(offer.shop_name)

    if coupon_id and offer_id:
        # id 权威：两边都有 id 时只看 id（不再回看名字，店名可能被平台改写）
        return None if coupon_id == offer_id else ScopeReject.SHOP_MISMATCH
    if coupon_name and offer_name:
        return None if coupon_name == offer_name else ScopeReject.SHOP_MISMATCH
    if not coupon_id and not coupon_name:
        return ScopeReject.SHOP_NOT_BOUND
    if not offer_id and not offer_name:
        return ScopeReject.SHOP_UNKNOWN
    return ScopeReject.SHOP_KEY_UNMATCHED


def _category_reject(coupon: Coupon, offer: Offer) -> ScopeReject | None:
    """品类范围券的判定：比**最细一级**（链尾）。返回 None 表示适用。

    只比链尾是刻意收窄的：券挂在哪个层级，从截图上看不出来，用整条链相交会让
    「耳机」的券匹配到所有「手机」——那是假阳性，正是要避免的方向。
    """
    if not coupon.category_ids:
        return ScopeReject.CATEGORY_NOT_BOUND
    if not offer.category_ids:
        return ScopeReject.CATEGORY_UNKNOWN
    return None if coupon.category_ids[-1] == offer.category_ids[-1] else ScopeReject.CATEGORY_MISMATCH


def coupon_scope_reject(coupon: Coupon, offer: Offer) -> ScopeReject | None:
    """这张券适用于这个商品吗？返回 None = 适用，否则给出拒绝原因。"""
    if coupon.type in _PLATFORM_WIDE:
        return None
    if coupon.type in _SHOP_SCOPED:
        return _shop_reject(coupon, offer)
    return _category_reject(coupon, offer)  # CATEGORY（CouponType 目前仅此一个剩余取值）


def coupons_for_offer(
    coupons: list[Coupon], offer: Offer
) -> tuple[list[Coupon], list[tuple[Coupon, ScopeReject]]]:
    """把一批用户券分成「适用于该商品」与「不适用（附原因）」两组。

    调用方拿到适用的那组后才并入到手价；不适用的一组应当透出给用户看，
    而不是静默丢掉——"我补录了券怎么没生效"必须能在界面上找到答案。
    """
    applicable: list[Coupon] = []
    rejected: list[tuple[Coupon, ScopeReject]] = []
    for coupon in coupons:
        reason = coupon_scope_reject(coupon, offer)
        if reason is None:
            applicable.append(coupon)
        else:
            rejected.append((coupon, reason))
    return applicable, rejected
