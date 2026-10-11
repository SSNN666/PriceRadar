"""Web 服务层测试：序列化（离线）+ 路由（TestClient + 打桩）。"""

import urllib.error
from decimal import Decimal

from fastapi.testclient import TestClient

import priceradar.service.api as api_module
from priceradar.agent.pipeline import ConfirmedItem, OfferGroup, PipelineResult
from priceradar.domain.models import Coupon, CouponType, Offer, Platform, ShopKind, Signals
from priceradar.engine.pricing import compute_price
from priceradar.trace import Trace

D = Decimal


def make_result() -> PipelineResult:
    offer_a = Offer(platform=Platform.PDD, sku_id="a", title="QCY N20 降噪耳机", list_price=D("99"), shop_name="QCY官方")
    offer_b = Offer(platform=Platform.PDD, sku_id="b", title="QCY N20 主动降噪 耳机", list_price=D("105"))
    confirmed = [
        ConfirmedItem(offer=offer_a, reason="检索结果（源商品）"),
        ConfirmedItem(offer=offer_b, reason="型号一致（N20）"),
    ]
    groups = [
        OfferGroup(representative=offer_a, members=[offer_a, offer_b], final_price=D("99"), reason="检索结果（源商品）"),
    ]
    breakdowns = [(offer_a, compute_price(offer_a))]
    result = PipelineResult(
        keyword="降噪耳机",
        pdd_count=20,
        jd_count=50,
        source=offer_a,
        confirmed=confirmed,
        groups=groups,
        similar_count=3,
        excluded_summary={"相似度过低": 10},
        breakdowns=breakdowns,
        notes=["京东侧未找到确切同款（候选 50 条均未达 A 档）"],
        summary="已找到 2 个同款报价，最低 ￥99。",
        trace=Trace(),
    )
    return result


def test_to_payload_shape():
    payload = api_module.to_payload(make_result())
    assert payload["keyword"] == "降噪耳机"
    assert payload["source"]["platform"] == "拼多多"
    assert payload["groups"][0]["count"] == 2 and payload["groups"][0]["is_source_group"] is True
    assert "url" in payload["offers"][0] and "image_url" in payload["offers"][0]
    assert "url" in payload["groups"][0] and "image_url" in payload["groups"][0]
    assert "shop_kind" in payload["offers"][0] and "signals" in payload["offers"][0]
    assert "similar_items" in payload
    assert payload["offers"][0]["final_price"] == "99"
    assert payload["offers"][0]["group_count"] == 2
    assert payload["candidates"]["similar"] == 3
    assert payload["trace"].startswith("[trace ")
    assert payload["cross_platform_hit"] is False


def test_payload_coupon_list_says_which_ones_counted_and_why_not():
    """券面板下半部分靠这个：每张券都要能回答"算没算进去、没算的话为什么"。"""
    counted = Coupon(coupon_id="c1", type=CouponType.PLATFORM, face_value=D("10"), threshold=D("50"))
    too_high = Coupon(coupon_id="c2", type=CouponType.PLATFORM, face_value=D("10"), threshold=D("500"))
    offer = Offer(
        platform=Platform.PDD, sku_id="a", title="商品", list_price=D("99"), shop_name="A店", coupons=[counted, too_high]
    )
    payload = api_module.offer_payload(offer, compute_price(offer))
    by_id = {item["coupon_id"]: item for item in payload["coupons"]}
    assert by_id["c1"]["applied"] is True
    assert by_id["c1"]["reject_reason"] == ""
    assert by_id["c2"]["applied"] is False
    assert by_id["c2"]["reject_reason"]          # 门槛没到，必须给出原因
    assert by_id["c1"]["source"] == "api"        # 前端据此区分"可去 App 领"与"你补录的"


def test_payload_carries_shop_id_and_category_ids():
    """绑定的依据要从 payload 里下发，确认卡才能反查出候选商品的标识。"""
    offer = Offer(
        platform=Platform.JD,
        sku_id="j1",
        title="大米",
        list_price=D("99"),
        shop_id="154435483",
        category_ids=["34884", "34911", "34919"],
    )
    payload = api_module.offer_payload(offer, compute_price(offer))
    assert payload["shop_id"] == "154435483"
    assert payload["category_ids"] == ["34884", "34911", "34919"]


def test_to_payload_carries_signals_and_shop_kind():
    offer = Offer(
        platform=Platform.JD,
        sku_id="j1",
        title="QCY N20 降噪耳机",
        list_price=Decimal("99"),
        shop_kind=ShopKind.SELF_RUN,
        signals=Signals(good_comments_share=98.0, comments=2000, in_order_count_30d=12000),
    )
    groups = [OfferGroup(representative=offer, members=[offer], final_price=Decimal("99"), reason="源")]
    result = PipelineResult(
        keyword="k",
        pdd_count=0,
        jd_count=1,
        source=offer,
        confirmed=[ConfirmedItem(offer=offer, reason="源")],
        groups=groups,
        similar_count=0,
        excluded_summary={},
        breakdowns=[(offer, compute_price(offer))],
    )
    payload = api_module.to_payload(result)
    assert payload["offers"][0]["shop_kind"] == "self_run"
    assert payload["offers"][0]["signals"]["good_comments_share"] == 98.0
    assert payload["offers"][0]["signals"]["in_order_count_30d"] == 12000
    assert payload["offers"][0]["image_url"] == ""


def test_to_payload_carries_tags_brand_and_sales():
    """拼多多侧新字段（保障标签/结构化品牌/销量）需到达前端。"""
    offer = Offer(
        platform=Platform.PDD,
        sku_id="p1",
        title="QCY N20 降噪耳机",
        list_price=Decimal("99"),
        brand="QCY",
        tags=["正品险", "退货包运费"],
        signals=Signals(sales_tip="1.2万+", sales_estimate=12000),
    )
    result = PipelineResult(
        keyword="k",
        pdd_count=1,
        jd_count=0,
        source=offer,
        confirmed=[ConfirmedItem(offer=offer, reason="源")],
        groups=[OfferGroup(representative=offer, members=[offer], final_price=Decimal("99"), reason="源")],
        similar_count=0,
        excluded_summary={},
        breakdowns=[(offer, compute_price(offer))],
    )
    payload = api_module.to_payload(result)
    assert payload["offers"][0]["tags"] == ["正品险", "退货包运费"]
    assert payload["offers"][0]["brand"] == "QCY"
    assert payload["offers"][0]["signals"]["sales_tip"] == "1.2万+"
    assert payload["offers"][0]["signals"]["sales_estimate"] == 12000


def test_to_payload_defaults_new_fields_when_absent():
    """老数据没有这些字段时也不得炸：空列表 / 空串 / None。"""
    payload = api_module.to_payload(make_result())
    offer = payload["offers"][0]
    assert offer["tags"] == [] and offer["brand"] == ""
    assert offer["signals"] is None or offer["signals"]["sales_tip"] is None


def test_unexpected_error_returns_json_not_plain_text(monkeypatch):
    """未捕获异常也必须返回 JSON。

    否则 uvicorn 吐纯文本 "Internal Server Error"，前端 .json() 解析失败，
    用户只看到 "Unexpected token 'I'"——真正的原因被完全掩盖。
    """

    def boom(q, jd, llm, risk=False):
        raise RuntimeError("内部炸了")

    monkeypatch.setattr(api_module, "execute_search", boom)
    client = TestClient(api_module.app, raise_server_exceptions=False)
    resp = client.get("/api/search", params={"q": "x"})
    assert resp.status_code == 500
    assert resp.headers["content-type"].startswith("application/json")
    assert "RuntimeError" in resp.json()["detail"]


def test_network_error_maps_to_502_with_readable_message(monkeypatch):
    """外部接口不可达（DNS 失败 / 超时 / 连接被拒）→ 502 + 人话，而不是裸 500。"""

    def boom(q, jd, llm, risk=False):
        raise urllib.error.URLError("getaddrinfo failed")

    monkeypatch.setattr(api_module, "execute_search", boom)
    client = TestClient(api_module.app, raise_server_exceptions=False)
    resp = client.get("/api/search", params={"q": "x"})
    assert resp.status_code == 502
    assert "网络不可达" in resp.json()["detail"]


def test_to_payload_carries_shop_qualification():
    """店铺资质信号必须到达前端：中文标签 + 评分档位 + 资质集合（筛选用）。"""
    offer = Offer(
        platform=Platform.PDD,
        sku_id="p1",
        title="QCY N20 降噪耳机",
        list_price=Decimal("99"),
        shop_kind=ShopKind.EXCLUSIVE,
        signals=Signals(shop_score_tier="高"),
    )
    result = PipelineResult(
        keyword="k",
        pdd_count=1,
        jd_count=0,
        source=offer,
        confirmed=[ConfirmedItem(offer=offer, reason="源")],
        groups=[OfferGroup(representative=offer, members=[offer], final_price=Decimal("99"), reason="源")],
        similar_count=0,
        excluded_summary={},
        breakdowns=[(offer, compute_price(offer))],
    )
    payload = api_module.to_payload(result)
    assert payload["offers"][0]["shop_kind"] == "exclusive"
    assert payload["offers"][0]["shop_kind_label"] == "专卖店"
    assert payload["offers"][0]["signals"]["shop_score_tier"] == "高"
    assert "franchise" in payload["certified_shop_kinds"]
    assert "personal" not in payload["certified_shop_kinds"]


def test_pop_shop_has_no_label_but_keeps_kind():
    """普通店刻意不给标签（避免噪声），但 kind 仍要传出去供筛选判断。"""
    offer = Offer(platform=Platform.PDD, sku_id="p", title="t", list_price=Decimal("9"), shop_kind=ShopKind.POP)
    result = PipelineResult(
        keyword="k", pdd_count=1, jd_count=0, source=offer,
        confirmed=[ConfirmedItem(offer=offer, reason="源")],
        groups=[OfferGroup(representative=offer, members=[offer], final_price=Decimal("9"), reason="源")],
        similar_count=0, excluded_summary={}, breakdowns=[(offer, compute_price(offer))],
    )
    payload = api_module.to_payload(result)
    assert payload["offers"][0]["shop_kind"] == "pop"
    assert payload["offers"][0]["shop_kind_label"] == ""


def test_api_similar_route(monkeypatch):
    """「找相似」：以商品为锚点扩展，结果按到手价升序可直接比。"""
    payload = {
        "anchor_sign": "SIGN",
        "count": 1,
        "offers": [api_module.offer_payload(make_result().breakdowns[0][0], compute_price(make_result().breakdowns[0][0]))],
        "notes": ["相似商品由平台推荐返回，混合有无关商品"],
    }
    monkeypatch.setattr(api_module, "execute_similar", lambda sign, limit=20: payload)
    resp = TestClient(api_module.app).get("/api/similar", params={"goods_sign": "SIGN"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["offers"][0]["quality"]["tier"]          # 质量评估随相似结果一并给出


def test_api_similar_maps_pdd_error_to_502(monkeypatch):
    def boom(sign, limit=20):
        raise api_module.PddApiError("50001 业务服务错误")

    monkeypatch.setattr(api_module, "execute_similar", boom)
    resp = TestClient(api_module.app).get("/api/similar", params={"goods_sign": "S"})
    assert resp.status_code == 502


def test_api_similar_maps_empty_to_404(monkeypatch):
    def empty(sign, limit=20):
        raise ValueError("未找到相似商品")

    monkeypatch.setattr(api_module, "execute_similar", empty)
    resp = TestClient(api_module.app).get("/api/similar", params={"goods_sign": "S"})
    assert resp.status_code == 404


def test_offers_payload_exposes_goods_sign():
    """前端「找相似」要靠 goods_sign 回调，没它按钮就出不来。"""
    payload = api_module.to_payload(make_result())
    assert "goods_sign" in payload["offers"][0]


def test_merge_user_coupons_does_not_recompute_without_db(tmp_path, monkeypatch):
    """无用户库时必须原样返回（不重算）——管线算价可能带 as_of 等参数，重算未必等价。

    必须显式打桩 DEFAULT_DB：不打桩时读的是本机 `data/user_data.db`，
    本机一旦有补录券（手动测试留下的），「无库」前提就不成立。
    """
    import priceradar.user_store as user_store

    monkeypatch.setattr(user_store, "DEFAULT_DB", tmp_path / "none.db")
    offer = Offer(platform=Platform.PDD, sku_id="x", title="t", list_price=Decimal("10"))
    original = compute_price(offer)
    merged, breakdowns = api_module.merge_user_coupons([(offer, original)])
    assert merged == 0
    assert breakdowns[0][1] is original


def test_api_search_route(monkeypatch):
    payload = api_module.to_payload(make_result())
    monkeypatch.setattr(api_module, "execute_search", lambda q, jd, llm, risk=False: payload)
    client = TestClient(api_module.app)
    resp = client.get("/api/search", params={"q": "QCY N20 降噪耳机"})
    assert resp.status_code == 200
    assert resp.json()["keyword"] == "降噪耳机"


def test_api_search_passes_risk_flag(monkeypatch):
    captured = {}

    def fake(q, jd, llm, risk=False):
        captured["risk"] = risk
        return api_module.to_payload(make_result())

    monkeypatch.setattr(api_module, "execute_search", fake)
    TestClient(api_module.app).get("/api/search", params={"q": "x", "risk": "true"})
    assert captured["risk"] is True


def test_api_search_maps_pdd_error_to_502(monkeypatch):
    def boom(q, jd, llm, risk=False):
        raise api_module.PddApiError("50001 业务服务错误")

    monkeypatch.setattr(api_module, "execute_search", boom)
    resp = TestClient(api_module.app).get("/api/search", params={"q": "x"})
    assert resp.status_code == 502
    assert "拼多多检索失败" in resp.json()["detail"]


def test_api_search_maps_empty_pool_to_404(monkeypatch):
    def empty(q, jd, llm, risk=False):
        raise ValueError("关键词 'x' 无检索结果")

    monkeypatch.setattr(api_module, "execute_search", empty)
    resp = TestClient(api_module.app).get("/api/search", params={"q": "x"})
    assert resp.status_code == 404


def test_index_serves_html():
    resp = TestClient(api_module.app).get("/")
    assert resp.status_code == 200
    assert "真价雷达" in resp.text
    assert "/api/search" in resp.text
    assert "/api/snapshot" in resp.text
    assert "/api/watchlist-status" in resp.text


def test_watchlist_status_empty_when_db_missing(tmp_path):
    payload = api_module.watchlist_status(db_path=tmp_path / "none.db")
    assert payload == {"captured_today": False, "rows_today": 0, "total_rows": 0, "drops": [], "watchlist": []}


def test_watchlist_status_with_drop(tmp_path):
    from datetime import datetime

    from priceradar.domain.models import Offer, Platform
    from priceradar.jobs.snapshot import connect, record_offers

    db = tmp_path / "snap.db"
    connection = connect(db)
    for day, price in (("2026-10-01", "100"), ("2026-10-02", "88")):
        offer = Offer(platform=Platform.PDD, sku_id="a", title="QCY N20", list_price=Decimal(price))
        record_offers(connection, [offer], "kw", captured=datetime.fromisoformat(day + "T09:00:00"))
    connection.close()

    payload = api_module.watchlist_status(db_path=db)
    assert payload["captured_today"] is False  # 测试用日期非今天
    assert payload["total_rows"] == 2
    assert len(payload["drops"]) == 1
    assert payload["drops"][0]["drop_pct"] == 12.0
    assert payload["drops"][0]["new_low"] is True
    assert len(payload["watchlist"]) == 1
    watched = payload["watchlist"][0]
    assert watched["observed_days"] == 2
    assert Decimal(watched["latest_price"]) == Decimal("88")
    assert watched["verdict"] == "数据积累中"
    assert len(watched["series"]) == 2
    assert "url" in watched and "image_url" in watched


def test_watchlist_status_route_shape():
    resp = TestClient(api_module.app).get("/api/watchlist-status")
    assert resp.status_code == 200
    body = resp.json()
    assert {"captured_today", "rows_today", "total_rows", "drops", "watchlist"} <= set(body)


def test_favicon_served():
    resp = TestClient(api_module.app).get("/favicon.ico")
    assert resp.status_code == 200
    assert "svg" in resp.headers["content-type"]
    assert "<svg" in resp.text


# ---------- 识图接口 ----------

def test_vision_product_mode(monkeypatch):
    from priceradar.vision import ProductAttrs

    def fake_extract(payload, media_type, **kwargs):
        return ProductAttrs(
            is_product=True, category="耳机", brand="QCY", model="N20",
            keywords=["QCY N20", "降噪耳机"], uncertain_fields=["color"], confidence=0.9,
        )

    monkeypatch.setattr("priceradar.vision.extract_product", fake_extract)
    resp = TestClient(api_module.app).post(
        "/api/vision?mode=product", files={"file": ("a.jpg", b"\xff\xd8fake", "image/jpeg")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "N20"
    assert body["query"] == "QCY N20 降噪耳机"
    assert body["uncertain_fields"] == ["color"]


def test_vision_coupon_mode(monkeypatch):
    from priceradar.vision import CouponExtraction, CouponFields

    def fake_extract(payload, media_type, **kwargs):
        return CouponExtraction(platform="jd", coupons=[CouponFields(type="platform", face_value=Decimal("50"), threshold=Decimal("300"))])

    monkeypatch.setattr("priceradar.vision.extract_coupons", fake_extract)
    resp = TestClient(api_module.app).post(
        "/api/vision?mode=coupon", files={"file": ("a.png", b"\x89PNG fake", "image/png")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["platform"] == "jd"
    assert Decimal(str(body["coupons"][0]["face_value"])) == Decimal("50")
    assert "用户确认" in body["note"]


def test_vision_unavailable_returns_503(monkeypatch):
    from priceradar.vision import VisionError

    def boom(payload, media_type, **kwargs):
        raise VisionError("未配置 DASHSCOPE_API_KEY")

    monkeypatch.setattr("priceradar.vision.extract_product", boom)
    resp = TestClient(api_module.app).post(
        "/api/vision?mode=product", files={"file": ("a.jpg", b"\xff\xd8fake", "image/jpeg")}
    )
    assert resp.status_code == 503
    assert "识图暂不可用" in resp.json()["detail"]


def test_vision_rejects_bad_mode():
    resp = TestClient(api_module.app).post(
        "/api/vision?mode=magic", files={"file": ("a.jpg", b"\xff\xd8fake", "image/jpeg")}
    )
    assert resp.status_code == 400


# ---------- 券补录回路（§5.6 闭环） ----------

def test_apply_user_coupons_merges_and_recomputes(tmp_path, monkeypatch):
    import priceradar.user_store as user_store
    from priceradar.domain.models import Coupon, CouponType
    from priceradar.user_store import connect_user_db, save_coupons

    db = tmp_path / "user.db"
    connection = connect_user_db(db)
    save_coupons(
        connection,
        "pdd",
        [Coupon(coupon_id="u1", type=CouponType.PLATFORM, face_value=Decimal("10"), threshold=Decimal("50"), source="screenshot")],
    )
    connection.close()
    monkeypatch.setattr(user_store, "DEFAULT_DB", db)

    result = make_result()  # 源报价：拼多多 ¥99
    merged = api_module.apply_user_coupons(result)
    assert merged == 1
    _offer, breakdown = result.breakdowns[0]
    assert breakdown.final_price == Decimal("89")
    assert any(line.label.startswith("[补录]") for line in breakdown.applied)


def test_apply_user_coupons_no_db_is_noop(tmp_path, monkeypatch):
    import priceradar.user_store as user_store

    monkeypatch.setattr(user_store, "DEFAULT_DB", tmp_path / "none.db")
    result = make_result()
    assert api_module.apply_user_coupons(result) == 0
    assert result.breakdowns[0][1].final_price == Decimal("99")


def test_coupon_confirm_and_list_routes(tmp_path, monkeypatch):
    import priceradar.user_store as user_store

    monkeypatch.setattr(user_store, "DEFAULT_DB", tmp_path / "user.db")
    client = TestClient(api_module.app)
    payload = {
        "platform": "jd",
        "coupons": [
            {"type": "platform", "face_value": 50, "threshold": 300, "scope": "全平台", "valid_until": "2026-11-11", "confidence": 0.9}
        ],
    }
    resp = client.post("/api/coupons", json=payload)
    assert resp.status_code == 200
    assert resp.json()["stored"] == 1
    assert "确认" not in resp.json()["note"]  # 已确认后写入的说明
    listing = client.get("/api/coupons").json()["coupons"]
    assert len(listing) == 1
    assert listing[0]["platform"] == "jd"
    assert Decimal(str(listing[0]["face_value"])) == Decimal("50")


def test_coupon_delete_route_is_idempotent(tmp_path, monkeypatch):
    import priceradar.user_store as user_store

    monkeypatch.setattr(user_store, "DEFAULT_DB", tmp_path / "user.db")
    client = TestClient(api_module.app)
    client.post(
        "/api/coupons",
        json={"platform": "jd", "coupons": [{"type": "platform", "face_value": 5, "threshold": 0}]},
    )
    assert len(client.get("/api/coupons").json()["coupons"]) == 1

    resp = client.delete("/api/coupons", params={"platform": "jd", "coupon_id": "shot-jd-0"})
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    assert client.get("/api/coupons").json()["coupons"] == []

    # 再删一次：幂等，不 404（照 /api/watchlist 与 /api/sessions 的既有约定）
    again = client.delete("/api/coupons", params={"platform": "jd", "coupon_id": "shot-jd-0"})
    assert again.status_code == 200
    assert again.json()["deleted"] is False


def test_coupon_delete_without_db_file_does_not_create_one(tmp_path, monkeypatch):
    """库里本来就没东西时，DELETE 不该为了"成功"而凭空建一个库出来。"""
    import priceradar.user_store as user_store

    missing = tmp_path / "nope.db"
    monkeypatch.setattr(user_store, "DEFAULT_DB", missing)
    resp = TestClient(api_module.app).delete(
        "/api/coupons", params={"platform": "jd", "coupon_id": "x"}
    )
    assert resp.status_code == 200
    assert resp.json()["deleted"] is False
    assert not missing.exists()


def test_unbound_shop_coupon_is_reported_in_the_note(tmp_path, monkeypatch):
    """写入成功 ≠ 会生效——没绑定的店铺券必须当场说清楚，否则用户以为补录了就一定算。"""
    import priceradar.user_store as user_store

    monkeypatch.setattr(user_store, "DEFAULT_DB", tmp_path / "user.db")
    resp = TestClient(api_module.app).post(
        "/api/coupons",
        json={"platform": "pdd", "coupons": [{"type": "shop", "face_value": 10, "threshold": 50}]},
    )
    assert resp.json()["stored"] == 1
    assert "未绑定适用范围" in resp.json()["note"]


def test_snapshot_route_returns_summary(monkeypatch):
    fake = {
        "inserted": 20,
        "keywords": {"QCY N20": {"fetched": 50, "inserted": 20}},
        "errors": [],
        "report": "快照库概况：170 行",
    }
    monkeypatch.setattr(api_module, "execute_snapshot", lambda: fake)
    resp = TestClient(api_module.app).post("/api/snapshot")
    assert resp.status_code == 200
    assert resp.json()["inserted"] == 20


def test_snapshot_route_returns_409_when_running():
    assert api_module._snapshot_lock.acquire(blocking=False)
    try:
        resp = TestClient(api_module.app).post("/api/snapshot")
        assert resp.status_code == 409
        assert "运行中" in resp.json()["detail"]
    finally:
        api_module._snapshot_lock.release()
