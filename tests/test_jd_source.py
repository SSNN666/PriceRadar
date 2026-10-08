"""京东联盟数据源客户端测试（全部离线：注入 transport + 真实响应结构样本）。"""

import json
import re
from datetime import date
from decimal import Decimal

import pytest

from priceradar.domain.models import CouponType, Platform, ShopKind
from priceradar.engine.pricing import compute_price
from priceradar.sources.jd import JdApiError, JdClient, jingfen_item_to_offer

ITEM = {
    "itemId": "JBQ6AYcAxzmqMF2EV6mLqMF2EV6mLQ_364ty2sPqrYQyxvOld",
    "skuName": "挂钩强力粘胶墙壁贴承重无痕挂勾贴钩免打孔钩子粘贴粘钩厨房收纳",
    "owner": "g",
    "goodCommentsShare": 98.0,
    "comments": 200000,
    "inOrderCount30Days": 100000,
    "priceInfo": {"price": 4.1, "lowestPrice": 4.1, "lowestCouponPrice": 0.1},
    "couponInfo": {
        "couponList": [
            {"discount": 4.0, "quota": 4.01, "isBest": 1, "platformType": 0, "useEndTime": 1792252799000},
            {"discount": 2.0, "quota": 2.1, "isBest": 0, "platformType": 0, "useEndTime": 1798387199000},
        ]
    },
    "shopInfo": {"shopName": "京造旗舰店"},
    "materialUrl": "jingfen.jd.com/detail/JBQ6AYcAxzmqMF2EV6mLqMF2EV6mLQ_364ty2sPqrYQyxvOld.html",
    "spuid": "364ty2sPqrYQyxvOld",
    "imageInfo": {"imageList": [{"url": "https://img14.360buyimg.com/n1/test.jpg"}]},
}


class FakeTransport:
    """记录请求 payload，返回预置响应（模拟京东 routerjson 外层包裹）。"""

    def __init__(
        self,
        inner: dict,
        key: str = "jd_union_open_goods_jingfen_query_responce",
        field: str = "result",
    ) -> None:
        self.inner = inner
        self.key = key
        self.field = field
        self.payloads: list[dict[str, str]] = []

    def __call__(self, payload: dict[str, str]) -> dict:
        self.payloads.append(payload)
        return {self.key: {"code": "0", self.field: json.dumps(self.inner)}}


def make_client(inner: dict) -> tuple[JdClient, FakeTransport]:
    transport = FakeTransport(inner)
    client = JdClient("test-app-key", "test-app-secret", transport=transport)
    return client, transport


def test_jingfen_item_maps_offer_with_signals():
    offer = jingfen_item_to_offer(ITEM)
    assert offer.platform is Platform.JD
    assert offer.sku_id == ITEM["itemId"]
    assert offer.title.startswith("挂钩强力粘胶")
    assert offer.list_price == Decimal("4.1")
    assert offer.shop_name == "京造旗舰店"
    assert offer.shop_kind is ShopKind.SELF_RUN  # owner=g -> 自营
    assert offer.signals is not None
    assert offer.signals.good_comments_share == 98.0
    assert offer.signals.comments == 200000
    assert offer.signals.in_order_count_30d == 100000
    assert offer.url == "https://jingfen.jd.com/detail/JBQ6AYcAxzmqMF2EV6mLqMF2EV6mLQ_364ty2sPqrYQyxvOld.html"
    assert offer.image_url == "https://img14.360buyimg.com/n1/test.jpg"


def test_jingfen_coupons_map_with_best_flag_and_expiry():
    offer = jingfen_item_to_offer(ITEM)
    assert len(offer.coupons) == 2
    best = offer.coupons[0]
    assert best.type is CouponType.SHOP
    assert best.face_value == Decimal("4.0")
    assert best.threshold == Decimal("4.01")
    assert best.scope_note == "最优券"
    assert best.valid_until is not None and best.valid_until.year == 2026
    assert offer.source_extra["material_url"].startswith("jingfen.jd.com/detail/")
    assert offer.source_extra["lowest_coupon_price"] == "0.1"  # 官方券后价参考（供引擎交叉校验）


def test_jingfen_offer_feeds_pricing_engine():
    """端到端小闭环：京粉响应 → Offer → 到手价引擎（同类型券取最优）。"""
    offer = jingfen_item_to_offer(ITEM)
    breakdown = compute_price(offer, as_of=date(2026, 10, 5))
    # 4.1 - max(满4.01减4, 满2.1减2) = 0.1
    assert breakdown.total_discount == Decimal("4.0")
    assert breakdown.final_price == Decimal("0.1")


def test_call_signs_and_unwraps():
    client, transport = make_client({"code": 200, "data": [ITEM]})
    offers = client.jingfen_goods(elite_id=1, page_size=10)
    assert len(offers) == 1
    payload = transport.payloads[0]
    assert payload["method"] == "jd.union.open.goods.jingfen.query"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", payload["timestamp"])
    assert re.fullmatch(r"[0-9A-F]{32}", payload["sign"])
    assert "access_token" not in payload  # 未配置 token 时不携带
    business = json.loads(payload["360buy_param_json"])
    assert business["goodsReq"]["eliteId"] == 1


def test_inner_business_error_raises():
    client, _ = make_client({"code": 403, "message": "无访问权限"})
    with pytest.raises(JdApiError) as excinfo:
        client.jingfen_goods()
    assert excinfo.value.code == 403
    assert "无访问权限" in str(excinfo.value)


PROMO_ITEM = {
    "skuId": 61861866,
    "goodsName": "童装女童连衣裙儿童裙子",
    "unitPrice": "89",
    "materialUrl": "http://item.jd.com/10000000.html",
    "imgUrl": "http://img.jd.com/x.jpg",
    "isJdSale": "1",
    "inOrderCount": "703",
    "shopId": "168168",
    "cidName": "母婴",
    "commisionRatioWl": "25",
}


def make_sku_client(inner: dict) -> tuple[JdClient, FakeTransport]:
    transport = FakeTransport(
        inner,
        key="jd_union_open_goods_promotiongoodsinfo_query_responce",
        field="queryResult",
    )
    return JdClient("test-app-key", "test-app-secret", transport=transport), transport


def test_goods_by_sku_maps_and_sends_comma_string():
    client, transport = make_sku_client({"code": 200, "data": {"promotionGoodsResp": [PROMO_ITEM]}})
    offers = client.goods_by_sku(["61861866", 7275691])
    assert len(offers) == 1
    offer = offers[0]
    assert offer.sku_id == "61861866"
    assert offer.list_price == Decimal("89")
    assert offer.shop_kind is ShopKind.SELF_RUN
    assert offer.signals is not None and offer.signals.in_order_count_30d == 703
    assert offer.source_extra["cid_name"] == "母婴"
    business = json.loads(transport.payloads[0]["360buy_param_json"])
    assert business["skuIds"] == "61861866,7275691"


def test_goods_by_sku_single_dict_is_normalized():
    client, _ = make_sku_client({"code": 200, "data": {"promotionGoodsResp": PROMO_ITEM}})
    offers = client.goods_by_sku(["61861866"])
    assert len(offers) == 1


def test_goods_by_sku_rejects_empty_and_oversized_lists():
    client, _ = make_sku_client({})
    with pytest.raises(ValueError):
        client.goods_by_sku([])
    with pytest.raises(ValueError):
        client.goods_by_sku([str(i) for i in range(101)])


# —— 店铺类型（owner 字段 + 店名兜底）——


def test_flagship_shop_kind_falls_back_to_shop_name():
    """实测 owner 只有 g/p 两种，官方旗舰店常落到 p——按店名补旗舰档。"""
    item = dict(ITEM, owner="p", shopInfo={"shopName": "某品牌旗舰店"})
    assert jingfen_item_to_offer(item).shop_kind is ShopKind.FLAGSHIP


def test_self_run_owner_wins_over_shop_name():
    """owner 信号更硬：自营店名里含"旗舰店"仍算自营。"""
    item = dict(ITEM, owner="g", shopInfo={"shopName": "京造旗舰店"})
    assert jingfen_item_to_offer(item).shop_kind is ShopKind.SELF_RUN


def test_unknown_owner_and_plain_shop_name_yields_no_kind():
    """专营店/无 owner：不猜，留 None（前端不显示徽标）。"""
    item = dict(ITEM, owner="", shopInfo={"shopName": "某某数码专营店"})
    assert jingfen_item_to_offer(item).shop_kind is None
