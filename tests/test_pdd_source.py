"""拼多多数据源客户端测试（全部离线：签名黄金值 + 注入 transport）。"""

import json
import re
from datetime import date
from decimal import Decimal

import pytest

from priceradar.domain.models import (
    CERTIFIED_SHOP_KINDS,
    SHOP_KIND_LABELS,
    CouponType,
    Platform,
    ShopKind,
)
from priceradar.engine.pricing import compute_price
from priceradar.sources.pdd import PddApiError, PddClient, goods_to_offer, parse_sales_tip, sign_params

GOODS = {
    "goods_id": 976939198068,
    "goods_name": "【海底捞】虾滑100g8袋虾球挤袋大颗粒速食成品火锅食材冒菜",
    "min_group_price": 15500,
    "coupon_discount": 9000,
    "coupon_min_order_amount": 9000,
    "mall_name": "海底捞旗舰店",
    "goods_sign": "E9v2QRxT2OBtSPqVwfvAtEyEKikY0Ixa_JreXo3rcy",
    "goods_thumbnail_url": "https://img.pddpic.com/gaudit-image/x.jpeg",
}


class FakeTransport:
    """记录请求 payload，返回预置响应。"""

    def __init__(self, response: dict) -> None:
        self.response = response
        self.payloads: list[dict[str, str]] = []

    def __call__(self, payload: dict[str, str]) -> dict:
        self.payloads.append(payload)
        return self.response


def make_client(response: dict) -> tuple[PddClient, FakeTransport]:
    transport = FakeTransport(response)
    client = PddClient("test-client", "test-secret", pid="44845383_318359410", transport=transport)
    return client, transport


def test_sign_matches_known_vector():
    params = {
        "type": "pdd.ddk.goods.search",
        "client_id": "test-client",
        "timestamp": "1700000000",
        "data_type": "JSON",
        "keyword": "test",
        "page": "1",
        "page_size": "10",
    }
    assert sign_params(params, "test-secret") == "6339FD86D73A2D822F19779A4C204BF8"


def test_search_goods_maps_offer():
    client, transport = make_client({"goods_search_response": {"goods_list": [GOODS]}})
    offers = client.search_goods("虾滑")
    assert len(offers) == 1
    offer = offers[0]
    assert offer.platform is Platform.PDD
    assert offer.sku_id == "976939198068"
    assert offer.list_price == Decimal("155")
    assert offer.shop_name == "海底捞旗舰店"
    assert offer.coupons[0].type is CouponType.SHOP
    assert offer.coupons[0].face_value == Decimal("90")
    assert offer.coupons[0].threshold == Decimal("90")
    assert offer.source_extra["goods_sign"] == GOODS["goods_sign"]
    assert offer.url == "https://mobile.yangkeduo.com/goods.html?goods_id=976939198068"
    assert offer.image_url == "https://img.pddpic.com/gaudit-image/x.jpeg"

    payload = transport.payloads[0]
    assert payload["type"] == "pdd.ddk.goods.search"
    assert payload["pid"] == "44845383_318359410"
    assert re.fullmatch(r"[0-9A-F]{32}", payload["sign"])


def test_search_result_without_coupon():
    offer = goods_to_offer(dict(GOODS, coupon_discount=0))
    assert offer.coupons == []


def test_search_with_coupon_param_passthrough():
    client, transport = make_client({"goods_search_response": {"goods_list": []}})
    client.search_goods("x", with_coupon=True)
    assert transport.payloads[0]["with_coupon"] == "true"


def test_recommend_goods_maps():
    client, transport = make_client({"goods_basic_detail_response": {"list": [GOODS]}})
    offers = client.recommend_goods(limit=3)
    assert len(offers) == 1
    payload = transport.payloads[0]
    assert payload["type"] == "pdd.ddk.goods.recommend.get"
    assert "pid" not in payload  # 推荐接口不携带 pid（实测无需备案）


def test_error_response_raises_api_error():
    client, _ = make_client(
        {
            "error_response": {
                "error_code": 50001,
                "sub_code": "60001",
                "sub_msg": "未传入已经授权备案过的相关参数(pid/custom_parameters)",
            }
        }
    )
    with pytest.raises(PddApiError) as excinfo:
        client.search_goods("x")
    assert excinfo.value.sub_code == "60001"
    assert "备案" in str(excinfo.value)


def test_real_shape_offer_feeds_pricing_engine():
    """端到端小闭环：拼多多真实响应结构 → Offer → 到手价引擎。"""
    offer = goods_to_offer(GOODS)
    breakdown = compute_price(offer, as_of=date(2026, 10, 5))
    # 拼团价 155 − 满90减90 = 65
    assert breakdown.final_price == Decimal("65")
    assert breakdown.total_discount == Decimal("90")


# —— 实测挖出的字段：结构化品牌 / 保障标签 / 销量 ——


def test_parse_sales_tip_handles_mixed_platform_formats():
    """实测：拼多多销量是混合格式，精确数字与"1.2万+"模糊值并存（模糊取下界）。"""
    assert parse_sales_tip("6861") == 6861
    assert parse_sales_tip("1.2万+") == 12000
    assert parse_sales_tip("15万+") == 150000


def test_parse_sales_tip_returns_none_when_unparseable():
    """解析不出来就给 None——不编造销量（诚实口径）。"""
    assert parse_sales_tip(None) is None
    assert parse_sales_tip("") is None
    assert parse_sales_tip("暂无销量") is None


def test_goods_to_offer_maps_brand_tags_and_sales():
    item = dict(
        GOODS,
        brand_name="Edifier/漫步者",
        unified_tags=["正品险", "退货包运费", "假一赔十"],
        sales_tip="1.2万+",
    )
    offer = goods_to_offer(item)
    assert offer.brand == "Edifier/漫步者"
    assert offer.tags == ["正品险", "退货包运费", "假一赔十"]
    assert offer.signals.sales_tip == "1.2万+"   # 原文保留（展示用）
    assert offer.signals.sales_estimate == 12000  # 解析值（筛选用）


def test_collect_coupon_is_mapped_into_final_price():
    """店铺收藏券：实测约 5% 商品有，此前完全未映射 → 到手价偏高。

    样本形态：面额 500 分、门槛 500 分（门槛=面额 → 等效无条件立减），与商品券互斥出现。
    """
    item = dict(
        GOODS,
        min_group_price=766,        # 7.66 元
        coupon_discount=0,          # 与收藏券互斥：此时没有商品券
        coupon_min_order_amount=0,
        clt_cpn_discount=300,       # 收藏券 3 元
        clt_cpn_min_amt=300,
        clt_cpn_end_time=1804867199,
    )
    offer = goods_to_offer(item)
    collect = [c for c in offer.coupons if c.type is CouponType.COLLECT]
    assert len(collect) == 1
    assert collect[0].face_value == Decimal("3")
    assert collect[0].valid_until == date(2027, 3, 12)

    breakdown = compute_price(offer, as_of=date(2026, 10, 7))
    assert breakdown.final_price == Decimal("4.66")   # 7.66 − 3
    assert breakdown.total_discount == Decimal("3")


def test_collect_coupon_label_states_the_condition():
    """标签必须写明"需收藏店铺"——用户看到低价时要知道怎么才能拿到。"""
    offer = goods_to_offer(dict(GOODS, min_group_price=766, coupon_discount=0, clt_cpn_discount=300, clt_cpn_min_amt=300))
    label = offer.coupons[0].label()
    assert "收藏" in label
    assert "满3减3" in label


def test_extra_coupon_amount_is_not_double_counted():
    """⚠️ extra_coupon_amount 是 coupon_discount 的重复字段，绝不可相加。"""
    base = dict(GOODS, coupon_discount=900, coupon_min_order_amount=900)
    with_extra = dict(base, extra_coupon_amount=900)
    a = compute_price(goods_to_offer(base), as_of=date(2026, 10, 7))
    b = compute_price(goods_to_offer(with_extra), as_of=date(2026, 10, 7))
    assert a.final_price == b.final_price          # 结果不受该字段影响
    assert len(goods_to_offer(with_extra).coupons) == 1


def test_expired_collect_coupon_is_rejected_not_counted():
    """过期券必须走既有拒因逻辑，不能被当成有效优惠。"""
    item = dict(
        GOODS,
        min_group_price=766,
        coupon_discount=0,
        clt_cpn_discount=300,
        clt_cpn_min_amt=300,
        clt_cpn_end_time=946684800,   # 2000-01-01，早已过期
    )
    breakdown = compute_price(goods_to_offer(item), as_of=date(2026, 10, 7))
    assert breakdown.final_price == Decimal("7.66")
    assert any(r.reason.value == "expired" for r in breakdown.rejected)


def test_merchant_type_maps_to_shop_kind():
    """店铺类型是拼多多侧唯一的结构化"店铺资质"信号（实测分布：旗舰店 64%、个人店 14%）。"""
    assert goods_to_offer(dict(GOODS, merchant_type=3)).shop_kind is ShopKind.FLAGSHIP
    assert goods_to_offer(dict(GOODS, merchant_type=4)).shop_kind is ShopKind.EXCLUSIVE
    assert goods_to_offer(dict(GOODS, merchant_type=5)).shop_kind is ShopKind.FRANCHISE
    assert goods_to_offer(dict(GOODS, merchant_type=1)).shop_kind is ShopKind.PERSONAL
    assert goods_to_offer(dict(GOODS, merchant_type=6)).shop_kind is ShopKind.POP


def test_merchant_type_accepts_string_form():
    """⚠️ recommend.get 返回 string、search 返回 int——两种都要能解析。"""
    assert goods_to_offer(dict(GOODS, merchant_type="3")).shop_kind is ShopKind.FLAGSHIP


def test_unknown_merchant_type_is_not_guessed():
    """认不出来就给 None（前端不显示徽标），不硬塞成一个默认值。"""
    for bad in (None, "", 0, 99, "abc", "旗舰店"):
        assert goods_to_offer(dict(GOODS, merchant_type=bad)).shop_kind is None


def test_shop_score_tier_is_mapped_from_desc_txt():
    """店铺评分档位（拼多多三档文字，实测三维度恒同）。"""
    offer = goods_to_offer(dict(GOODS, desc_txt="高", lgst_txt="高", serv_txt="高"))
    assert offer.signals.shop_score_tier == "高"
    assert goods_to_offer(dict(GOODS, desc_txt="低")).signals.shop_score_tier == "低"
    assert goods_to_offer(dict(GOODS, desc_txt="")).signals.shop_score_tier is None


def test_shop_kind_labels_omit_pop_on_purpose():
    """普通第三方店是"没有信号"，不给徽标（避免噪声）。"""
    assert SHOP_KIND_LABELS[ShopKind.FLAGSHIP] == "旗舰店"
    assert ShopKind.POP not in SHOP_KIND_LABELS
    assert ShopKind.PERSONAL in SHOP_KIND_LABELS


def test_certified_shop_kinds_exclude_personal_and_pop():
    assert ShopKind.FLAGSHIP in CERTIFIED_SHOP_KINDS
    assert ShopKind.SELF_RUN in CERTIFIED_SHOP_KINDS
    assert ShopKind.PERSONAL not in CERTIFIED_SHOP_KINDS
    assert ShopKind.POP not in CERTIFIED_SHOP_KINDS


def test_recommend_similar_requests_channel_type_3():
    """⚠️ channel_type 默认值是 5（实时热销榜）＝全局榜单，与参考商品无关。

    要"找相似"必须显式传 3——忘传就静默返回一堆无关商品（不报错，最难查）。
    """
    client, transport = make_client({"goods_basic_detail_response": {"list": [GOODS]}})
    offers = client.recommend_similar("SIGN123", limit=20)

    payload = transport.payloads[0]
    assert payload["type"] == "pdd.ddk.goods.recommend.get"
    assert payload["channel_type"] == "3"
    assert json.loads(payload["goods_sign_list"]) == ["SIGN123"]   # 约定取数组第一位
    assert len(offers) == 1


def test_recommend_similar_handles_empty_response():
    client, _ = make_client({"goods_basic_detail_response": {"list": []}})
    assert client.recommend_similar("SIGN") == []


def test_recommend_similar_reuses_goods_to_offer():
    """推荐接口字段比搜索少（无 mall_coupon_*/service_tags），但映射器必须兼容。"""
    item = {k: v for k, v in GOODS.items() if not k.startswith("mall_coupon")}
    item.pop("service_tags", None)
    client, _ = make_client({"goods_basic_detail_response": {"list": [item]}})
    offer = client.recommend_similar("SIGN")[0]
    assert offer.title == GOODS["goods_name"]
    assert offer.list_price == Decimal("155")


def test_goods_to_offer_without_new_fields_stays_clean():
    """字段缺失时不猜：品牌 None、标签空、销量 None。"""
    offer = goods_to_offer(GOODS)
    assert offer.brand is None
    assert offer.tags == []
    assert offer.signals.sales_tip is None
    assert offer.signals.sales_estimate is None
