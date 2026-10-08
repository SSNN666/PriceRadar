"""M0 命令行 Demo：用 mock 数据演示到手价引擎（真实 API 接入见 PRD §7 M0）。

运行：python -m priceradar.demo
"""

from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal

from priceradar.domain.models import (
    Coupon,
    CouponType,
    Offer,
    Platform,
    Promotion,
    PromotionKind,
    ShopKind,
)
from priceradar.engine.pricing import compute_price
from priceradar.report import render_price_card

AS_OF = date(2026, 10, 5)


def _d(value: str) -> Decimal:
    return Decimal(value)


def build_mock_offers() -> list[Offer]:
    """mock 场景：同一款降噪耳机在三个平台的报价（数据仅用于演示引擎）。"""
    return [
        Offer(
            platform=Platform.JD,
            sku_id="jd-10001",
            title="索尼 WH-1000XM6 头戴式降噪耳机",
            list_price=_d("2499"),
            shop_name="索尼自营旗舰店",
            shop_kind=ShopKind.SELF_RUN,
            coupons=[
                Coupon(coupon_id="jd-c1", type=CouponType.SHOP, face_value=_d("100"), threshold=_d("2000")),
                Coupon(coupon_id="jd-c2", type=CouponType.PLATFORM, face_value=_d("200"), threshold=_d("3000")),
            ],
            promotions=[
                Promotion(promotion_id="jd-p1", kind=PromotionKind.EVERY_N, threshold=_d("1000"), reduction=_d("80")),
            ],
        ),
        Offer(
            platform=Platform.PDD,
            sku_id="pdd-20002",
            title="索尼 WH-1000XM6 国行 降噪耳机",
            list_price=_d("2399"),
            shop_name="索尼影音专营店",
            shop_kind=ShopKind.POP,
            coupons=[
                Coupon(coupon_id="pdd-c1", type=CouponType.SHOP, face_value=_d("150"), threshold=_d("2000")),
            ],
        ),
        Offer(
            platform=Platform.TAOBAO,
            sku_id="tb-30003",
            title="SONY WH-1000XM6 头戴降噪蓝牙耳机",
            list_price=_d("2450"),
            shop_name="天猫索尼官方旗舰店",
            shop_kind=ShopKind.FLAGSHIP,
            coupons=[
                Coupon(coupon_id="tb-c1", type=CouponType.CATEGORY, face_value=_d("50"), threshold=_d("2000")),
            ],
            promotions=[
                Promotion(promotion_id="tb-p1", kind=PromotionKind.STEP, threshold=_d("1500"), reduction=_d("100"), group="tb-tmall-act"),
                Promotion(promotion_id="tb-p2", kind=PromotionKind.STEP, threshold=_d("3000"), reduction=_d("250"), group="tb-tmall-act"),
            ],
        ),
    ]


def main() -> None:
    # Windows 管道/终端编码差异防护：优先 UTF-8 输出，无法编码的字符降级替换
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    print("PriceRadar Demo（mock 数据 · 到手价引擎 v0.1）\n")
    offers = build_mock_offers()
    items = [(offer, compute_price(offer, as_of=AS_OF)) for offer in offers]
    print(render_price_card(items))


if __name__ == "__main__":
    main()
