"""用户券存储与补录标记测试（全离线）。"""

from datetime import date
from decimal import Decimal

from priceradar.domain.models import Coupon, CouponType
from priceradar.user_store import connect_user_db, list_coupons, load_active_coupons, save_coupons

D = Decimal


def make_coupon(coupon_id: str = "u1", face: str = "50", threshold: str = "300", **kwargs) -> Coupon:
    return Coupon(
        coupon_id=coupon_id,
        type=CouponType.PLATFORM,
        face_value=D(face),
        threshold=D(threshold),
        source="screenshot",
        **kwargs,
    )


def test_save_and_load_roundtrip(tmp_path):
    connection = connect_user_db(tmp_path / "u.db")
    assert save_coupons(connection, "jd", [make_coupon()]) == 1
    loaded = load_active_coupons(connection, "jd")
    assert len(loaded) == 1
    assert loaded[0].face_value == D("50")
    assert loaded[0].source == "screenshot"
    connection.close()


def test_same_id_replaces(tmp_path):
    connection = connect_user_db(tmp_path / "u.db")
    save_coupons(connection, "jd", [make_coupon(face="50")])
    save_coupons(connection, "jd", [make_coupon(face="80")])
    loaded = load_active_coupons(connection, "jd")
    assert len(loaded) == 1
    assert loaded[0].face_value == D("80")
    connection.close()


def test_expired_filtered_by_as_of(tmp_path):
    connection = connect_user_db(tmp_path / "u.db")
    save_coupons(connection, "jd", [make_coupon(valid_until=date(2026, 10, 1))])
    assert load_active_coupons(connection, "jd", as_of=date(2026, 10, 7)) == []
    assert len(load_active_coupons(connection, "jd", as_of=date(2026, 9, 30))) == 1
    connection.close()


def test_platform_isolation_and_list(tmp_path):
    connection = connect_user_db(tmp_path / "u.db")
    save_coupons(connection, "jd", [make_coupon("j1")])
    save_coupons(connection, "pdd", [make_coupon("p1")])
    assert len(load_active_coupons(connection, "pdd")) == 1
    assert len(load_active_coupons(connection, "jd")) == 1
    assert len(list_coupons(connection)) == 2
    assert [row["platform"] for row in list_coupons(connection, "jd")] == ["jd"]
    connection.close()


def test_coupon_label_marks_manual_sources():
    assert make_coupon().label().startswith("[补录]")  # screenshot 来源
    manual = Coupon(coupon_id="m1", type=CouponType.PLATFORM, face_value=D("50"), threshold=D("300"), source="manual")
    assert manual.label().startswith("[补录]")
    api_coupon = Coupon(coupon_id="a", type=CouponType.SHOP, face_value=D("10"), threshold=D("100"), source="api")
    assert not api_coupon.label().startswith("[补录]")
