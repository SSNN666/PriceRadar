"""报告卡渲染测试。"""

from priceradar.demo import build_mock_offers
from priceradar.engine.pricing import compute_price
from priceradar.report import render_price_card


def test_card_orders_platforms_by_final_price():
    offers = build_mock_offers()
    items = [(offer, compute_price(offer)) for offer in offers]
    card = render_price_card(items)
    assert "到手价对比" in card
    assert "以结算页为准" in card
    # 京东（¥2239）应排在拼多多（¥2249）与淘宝（¥2300）之前
    assert card.index("京东") < card.index("拼多多") < card.index("淘宝")


def test_card_lists_rejected_coupons_with_reason():
    offers = build_mock_offers()
    jd = offers[0]
    card = render_price_card([(jd, compute_price(jd))])
    assert "未计入" in card
    assert "未达门槛" in card


def test_card_annotation_shows_group_count():
    offers = build_mock_offers()
    jd = offers[0]
    card = render_price_card([(jd, compute_price(jd))], count_by_sku={jd.sku_id: 3})
    assert "同款 ×3 家" in card
