"""价格快照采集测试（临时库，seller 注入，全离线）。"""

from datetime import datetime
from decimal import Decimal

from priceradar.domain.models import Offer, Platform
from priceradar.jobs.snapshot import collect, connect, history, record_offers, report

D = Decimal


def make_offer(sku: str, title: str = "QCY N20 降噪耳机", price: str = "99", platform: Platform = Platform.PDD) -> Offer:
    return Offer(platform=platform, sku_id=sku, title=title, list_price=D(price), shop_name="测试店")


def test_record_is_idempotent_within_same_day(tmp_path):
    connection = connect(tmp_path / "snap.db")
    offers = [make_offer("a"), make_offer("b", price="105")]
    first = record_offers(connection, offers, "QCY N20", captured=datetime(2026, 10, 5, 9, 0, 0))
    second = record_offers(connection, offers, "QCY N20", captured=datetime(2026, 10, 5, 21, 0, 0))
    assert first == 2
    assert second == 0  # 同日幂等
    third = record_offers(connection, offers, "QCY N20", captured=datetime(2026, 10, 6, 9, 0, 0))
    assert third == 2  # 次日新增
    connection.close()


def test_history_returns_ascending_prices(tmp_path):
    connection = connect(tmp_path / "snap.db")
    record_offers(connection, [make_offer("a", price="99")], "kw", captured=datetime(2026, 10, 5, 9, 0, 0))
    record_offers(connection, [make_offer("a", price="89")], "kw", captured=datetime(2026, 10, 6, 9, 0, 0))
    rows = history(connection, "pdd", "a")
    assert [day for day, _ in rows] == ["2026-10-05", "2026-10-06"]
    assert [D(price) for _, price in rows] == [D("99"), D("89")]
    connection.close()


def test_collect_with_injected_searchers(tmp_path):
    def pdd_searcher(keyword: str, size: int):
        return [make_offer("p1", price="99")]

    def jd_searcher(_keyword: str):
        return [make_offer("j1", price="88", platform=Platform.JD)]

    summary = collect(["QCY N20"], tmp_path / "snap.db", pdd_searcher=pdd_searcher, jd_searcher=jd_searcher)
    assert summary["inserted"] == 2
    assert summary["keywords"]["QCY N20"]["fetched"] == 2


def test_collect_survives_platform_failure(tmp_path):
    def boom(keyword: str, size: int):
        raise RuntimeError("pdd 挂了")

    def jd_searcher(_keyword: str):
        return [make_offer("j1", platform=Platform.JD)]

    summary = collect(["kw"], tmp_path / "snap.db", pdd_searcher=boom, jd_searcher=jd_searcher)
    assert summary["inserted"] == 1
    assert any("拼多多" in error for error in summary["errors"])


def test_record_offers_persists_url_and_image(tmp_path):
    connection = connect(tmp_path / "snap.db")
    offer = Offer(
        platform=Platform.PDD, sku_id="a", title="t", list_price=D("99"),
        url="https://x.com/1", image_url="https://img/x.jpg",
    )
    record_offers(connection, [offer], "kw")
    row = connection.execute("SELECT url, image_url FROM price_snapshots").fetchone()
    assert row[0] == "https://x.com/1"
    assert row[1] == "https://img/x.jpg"
    connection.close()


def test_backfill_fills_url_on_same_day_rerun(tmp_path):
    connection = connect(tmp_path / "snap.db")
    bare = Offer(platform=Platform.PDD, sku_id="a", title="t", list_price=D("99"))
    record_offers(connection, [bare], "kw")
    rich = Offer(
        platform=Platform.PDD, sku_id="a", title="t", list_price=D("99"),
        url="https://x.com/1", image_url="https://img/x.jpg",
    )
    record_offers(connection, [rich], "kw")  # 同日重跑：不新增行，但回填展示字段
    row = connection.execute("SELECT url, image_url FROM price_snapshots").fetchone()
    assert row[0] == "https://x.com/1"
    assert connection.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0] == 1
    connection.close()


def test_schema_migration_adds_missing_columns(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    legacy = sqlite3.connect(db)
    legacy.execute(
        "CREATE TABLE price_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT, captured_date TEXT NOT NULL, "
        "captured_at TEXT NOT NULL, platform TEXT NOT NULL, sku_id TEXT NOT NULL, title TEXT NOT NULL, "
        "shop_name TEXT, list_price TEXT NOT NULL, final_price TEXT NOT NULL, total_discount TEXT NOT NULL, keyword TEXT NOT NULL)"
    )
    legacy.commit()
    legacy.close()
    connection = connect(db)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(price_snapshots)")}
    assert {"url", "image_url"} <= columns
    connection.close()


def test_report_smoke(tmp_path):
    connection = connect(tmp_path / "snap.db")
    assert "为空" in report(connection)
    record_offers(connection, [make_offer("a")], "kw")
    text = report(connection)
    assert "1 行" in text and "观测 1 天" in text
    connection.close()
