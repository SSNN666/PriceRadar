"""价格历史洞察测试（时机建议分位 / 先涨后降检测 / 快照库读取 / 管线接入）。"""

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from priceradar.domain.models import Offer, Platform
from priceradar.insights.history import PricePoint
from priceradar.insights import (
    advise_timing,
    attach_history_insights,
    detect_fake_discount,
    load_history,
    sku_history_view,
)
from priceradar.jobs.snapshot import connect, record_offers

D = Decimal


def points_from(prices: list[str], list_prices: list[str] | None = None) -> list[PricePoint]:
    """构造观测序列。

    `list_prices` 省略时标价 = 到手价（**没有优惠**，最省事的默认）。
    要构造「有优惠」或「优惠在缩水」的序列，就显式传标价。
    """
    start = date(2026, 9, 1)
    return [
        PricePoint(
            (start + timedelta(days=index)).isoformat(),
            D(list_prices[index]) if list_prices else D(price),
            D(price),
        )
        for index, price in enumerate(prices)
    ]


# ---------- 时机建议 ----------

def test_advise_insufficient_days():
    advice = advise_timing(points_from(["99"] * 5))
    assert advice.status == "insufficient"
    assert advice.observed_days == 5
    assert "5/14" in advice.note


def test_advise_no_data():
    advice = advise_timing([])
    assert advice.status == "insufficient"
    assert advice.observed_days == 0


def test_advise_low_percentile_is_buy():
    advice = advise_timing(points_from(["100"] * 13 + ["80"]))
    assert advice.status == "ok"
    assert "观测低位" in advice.verdict
    assert advice.lowest.final_price == D("80")   # PricePoint 带标价与到手价两个字段
    assert advice.lowest.list_price == D("80")    # 本用例没构造优惠 → 标价 = 到手价


def test_advise_high_percentile_is_wait():
    advice = advise_timing(points_from(["80"] * 13 + ["100"]))
    assert "观测高位" in advice.verdict


def test_advise_mid_percentile():
    # 现价 90 落在 80×4 与 100×6 之间：8/14 ≈ 57% → 中位
    advice = advise_timing(points_from(["80"] * 4 + ["100"] * 6 + ["90"] * 4))
    assert "观测中位" in advice.verdict


# ---------- 先涨后降 ----------

def test_fake_discount_insufficient():
    report = detect_fake_discount(points_from(["100"] * 6))
    assert report.status == "insufficient"


def test_fake_discount_suspected():
    # 前期基线 100×7，近期峰值 140，现价 120（≤140×0.9）
    report = detect_fake_discount(points_from(["100"] * 7 + ["140"] * 6 + ["120"]))
    assert report.status == "suspected"
    assert "先涨后降" in report.detail and "140" in report.detail
    assert report.driver == "list_price"


def test_fake_discount_clean_flat():
    report = detect_fake_discount(points_from(["100"] * 14))
    assert report.status == "clean"


def test_fake_discount_clean_normal_drop():
    # 正常降价：无涨价阶段
    report = detect_fake_discount(points_from(["120"] * 7 + ["110", "105", "100", "99", "98", "97", "95"]))
    assert report.status == "clean"


# ---------- 区分「商家提价」与「优惠缩水」 ----------
# 这两种情况让**到手价曲线长得一模一样**，只有标价能分开它们。
# 对买家的含义相反：前者是套路该警惕，后者往往只是促销窗口结束。

_COUPON_FINALS = ["70"] * 7 + ["100"] * 6 + ["90"]     # 基线 ￥70 → 峰值 ￥100 → 现价 ￥90


def test_coupon_withdrawal_is_not_called_a_fake_discount():
    """标价没动、只有优惠缩水 → **不是**先涨后降。"""
    points = points_from(_COUPON_FINALS, list_prices=["100"] * 14)
    report = detect_fake_discount(points)
    assert report.status == "coupon_withdrawn"
    assert report.driver == "coupon"
    assert "不是商家提价" in report.detail
    assert "优惠" in report.detail


def test_same_curve_but_rising_list_price_is_a_fake_discount():
    """**同样的到手价曲线**，标价跟着涨了 → 这才是商家提价再打折。"""
    points = points_from(_COUPON_FINALS, list_prices=["70"] * 7 + ["100"] * 7)
    report = detect_fake_discount(points)
    assert report.status == "suspected"
    assert report.driver == "list_price"


def test_the_two_drivers_differ_only_by_list_price():
    """把上面两条并排：曲线完全相同，唯一差别是标价。"""
    same_finals = _COUPON_FINALS
    coupon = detect_fake_discount(points_from(same_finals, list_prices=["100"] * 14))
    merchant = detect_fake_discount(points_from(same_finals, list_prices=["70"] * 7 + ["100"] * 7))
    assert coupon.status != merchant.status
    assert coupon.driver != merchant.driver


def test_price_point_markdown_is_list_minus_final():
    point = PricePoint("2026-09-01", D("100"), D("70"))
    assert point.markdown == D("30")
    assert point.pair == ("2026-09-01", D("70"))


def test_no_discount_means_list_equals_final():
    points = points_from(["100"] * 14)
    assert all(point.markdown == 0 for point in points)
    assert detect_fake_discount(points).status == "clean"


# ---------- 快照库读取与视图 ----------

def test_load_history_and_view_from_sqlite(tmp_path):
    connection = connect(tmp_path / "snap.db")
    offer = Offer(platform=Platform.PDD, sku_id="a", title="QCY N20 降噪耳机", list_price=D("99"))
    for offset in range(3):
        record_offers(connection, [offer], "kw", captured=datetime(2026, 10, 1 + offset, 9, 0, 0))
    points = load_history(connection, "pdd", "a")
    assert len(points) == 3
    view = sku_history_view(connection, "pdd", "a", current=D("99"))
    assert view["observed_days"] == 3
    assert view["status"] == "insufficient"
    assert view["verdict"] == "数据积累中"
    connection.close()


def test_attach_history_insights_fills_result(tmp_path):
    connection = connect(tmp_path / "snap.db")
    offer = Offer(platform=Platform.PDD, sku_id="a", title="QCY N20 降噪耳机", list_price=D("99"))
    for offset in range(2):
        record_offers(connection, [offer], "kw", captured=datetime(2026, 10, 1 + offset, 9, 0, 0))
    connection.close()

    from priceradar.agent.pipeline import ConfirmedItem, OfferGroup, PipelineResult
    from priceradar.engine.pricing import compute_price

    groups = [OfferGroup(representative=offer, members=[offer], final_price=D("99"), reason="源")]
    result = PipelineResult(
        keyword="kw",
        pdd_count=1,
        jd_count=0,
        source=offer,
        confirmed=[ConfirmedItem(offer=offer, reason="源")],
        groups=groups,
        similar_count=0,
        excluded_summary={},
        breakdowns=[(offer, compute_price(offer))],
    )
    attach_history_insights(result, db_path=tmp_path / "snap.db")
    assert "pdd:a" in result.history
    assert result.history["pdd:a"]["observed_days"] == 2


def test_attach_history_insights_skips_missing_db(tmp_path):
    from priceradar.agent.pipeline import PipelineResult

    offer = Offer(platform=Platform.PDD, sku_id="a", title="t", list_price=D("99"))
    result = PipelineResult(
        keyword="kw", pdd_count=1, jd_count=0, source=offer,
        confirmed=[], groups=[], similar_count=0, excluded_summary={}, breakdowns=[],
    )
    attach_history_insights(result, db_path=tmp_path / "not-exists.db")
    assert result.history == {}


# ---------- 价格序列（sparkline 数据）与降价提醒 ----------

def test_view_includes_price_series(tmp_path):
    connection = connect(tmp_path / "snap.db")
    offer = Offer(platform=Platform.PDD, sku_id="a", title="t", list_price=D("99"))
    record_offers(connection, [offer], "kw", captured=datetime(2026, 10, 1))
    record_offers(connection, [offer], "kw", captured=datetime(2026, 10, 2))
    view = sku_history_view(connection, "pdd", "a")
    assert [point["date"] for point in view["series"]] == ["2026-10-01", "2026-10-02"]
    connection.close()


def test_price_drops_detects_and_flags_new_low(tmp_path):
    from priceradar.insights import price_drops

    connection = connect(tmp_path / "snap.db")
    for day, price in (("2026-10-01", "100"), ("2026-10-02", "99"), ("2026-10-03", "88")):
        offer = Offer(platform=Platform.PDD, sku_id="a", title="QCY N20 降噪耳机", list_price=D(price))
        record_offers(connection, [offer], "kw", captured=datetime.fromisoformat(day + "T09:00:00"))
    drops = price_drops(connection)
    assert len(drops) == 1
    assert drops[0].drop_pct == pytest.approx(11.1)
    assert drops[0].new_low is True
    assert drops[0].previous_price == D("99") and drops[0].latest_price == D("88")
    connection.close()


def test_price_drops_ignores_rises_and_singletons(tmp_path):
    from priceradar.insights import price_drops

    connection = connect(tmp_path / "snap.db")
    for day, price in (("2026-10-01", "90"), ("2026-10-02", "99")):
        offer = Offer(platform=Platform.PDD, sku_id="r", title="涨价商品", list_price=D(price))
        record_offers(connection, [offer], "kw", captured=datetime.fromisoformat(day + "T09:00:00"))
    single = Offer(platform=Platform.PDD, sku_id="s", title="单次观测", list_price=D("50"))
    record_offers(connection, [single], "kw", captured=datetime(2026, 10, 1))
    assert price_drops(connection) == []
    connection.close()
