"""用户券存储与补录标记测试（全离线）。"""

from datetime import date
from decimal import Decimal

from priceradar.domain.models import Coupon, CouponType
from priceradar.user_store import connect_user_db, delete_coupon, list_coupons, load_active_coupons, save_coupons

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


def test_binding_columns_roundtrip(tmp_path):
    """适用范围绑定必须能存下来再读回来——存不下来等于没绑。"""
    connection = connect_user_db(tmp_path / "u.db")
    save_coupons(
        connection,
        "jd",
        [
            Coupon(
                coupon_id="s1",
                type=CouponType.SHOP,
                face_value=D("10"),
                threshold=D("50"),
                source="screenshot",
                shop_id="154435483",
                shop_name="农合天下旗舰店",
                bound_label="东北大米 5kg",
            )
        ],
    )
    loaded = load_active_coupons(connection, "jd")[0]
    assert loaded.shop_id == "154435483"
    assert loaded.shop_name == "农合天下旗舰店"
    assert loaded.bound_label == "东北大米 5kg"
    connection.close()


def test_category_ids_roundtrip_as_list(tmp_path):
    """品类链是列表，存的是 JSON 字符串——读回来必须还是列表，不能变成字符串。"""
    connection = connect_user_db(tmp_path / "u.db")
    save_coupons(
        connection,
        "pdd",
        [
            Coupon(
                coupon_id="c1",
                type=CouponType.CATEGORY,
                face_value=D("20"),
                category_ids=["5752", "5826", "19146"],
                source="screenshot",
            )
        ],
    )
    assert load_active_coupons(connection, "pdd")[0].category_ids == ["5752", "5826", "19146"]
    assert list_coupons(connection, "pdd")[0]["category_ids"] == ["5752", "5826", "19146"]
    connection.close()


def test_migration_adds_binding_columns_to_old_db(tmp_path):
    """老库（没有绑定列）必须能自动补列。

    `CREATE TABLE IF NOT EXISTS` 对已存在的表是 no-op——不补列的话，
    此后任何读写新列都会 `no such column` 直接炸。
    """
    import sqlite3

    path = tmp_path / "old.db"
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE user_coupons ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, platform TEXT NOT NULL, coupon_id TEXT NOT NULL,"
        "type TEXT NOT NULL, face_value TEXT NOT NULL, threshold TEXT NOT NULL,"
        "scope_note TEXT NOT NULL DEFAULT '', valid_until TEXT, source TEXT NOT NULL,"
        "confidence REAL NOT NULL DEFAULT 1.0, created_at TEXT NOT NULL,"
        "UNIQUE(platform, coupon_id))"
    )
    legacy.execute(
        "INSERT INTO user_coupons (platform, coupon_id, type, face_value, threshold, source, confidence, created_at)"
        " VALUES ('jd', 'old1', 'platform', '5', '0', 'manual', 1.0, '2026-10-01T00:00:00')"
    )
    legacy.commit()
    legacy.close()

    connection = connect_user_db(path)  # 触发迁移
    columns = {row[1] for row in connection.execute("PRAGMA table_info(user_coupons)")}
    assert {"shop_id", "shop_name", "category_ids", "bound_label"} <= columns

    old_coupons = load_active_coupons(connection, "jd")
    assert len(old_coupons) == 1
    assert old_coupons[0].shop_id is None      # 老行没有绑定，读回 None 而不是炸
    assert old_coupons[0].category_ids == []
    connection.close()


def test_delete_coupon_is_idempotent(tmp_path):
    connection = connect_user_db(tmp_path / "u.db")
    save_coupons(connection, "jd", [make_coupon("j1")])
    assert delete_coupon(connection, "jd", "j1") is True
    assert load_active_coupons(connection, "jd") == []
    assert delete_coupon(connection, "jd", "j1") is False   # 再删一次不抛异常
    assert delete_coupon(connection, "pdd", "不存在") is False
    connection.close()


def test_coupon_label_marks_manual_sources():
    assert make_coupon().label().startswith("[补录]")  # screenshot 来源
    manual = Coupon(coupon_id="m1", type=CouponType.PLATFORM, face_value=D("50"), threshold=D("300"), source="manual")
    assert manual.label().startswith("[补录]")
    api_coupon = Coupon(coupon_id="a", type=CouponType.SHOP, face_value=D("10"), threshold=D("100"), source="api")
    assert not api_coupon.label().startswith("[补录]")
