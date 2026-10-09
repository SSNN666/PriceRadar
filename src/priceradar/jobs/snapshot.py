"""价格快照采集（append-only，幂等）。

每日对关注关键词执行检索（拼多多 + 京东京粉频道），把每个商品的
公开优惠口径到手价写入 SQLite。同平台 + 同 sku + 同日 已存在则跳过（可重复跑）。

用法：
    python -m priceradar.jobs.snapshot                        # 读取 data/watchlist.json
    python -m priceradar.jobs.snapshot --keywords "QCY N20 降噪耳机"
    python -m priceradar.jobs.snapshot --report               # 查看已采集概况

存储：data/snapshots.db（gitignored）。数据供后续"时机建议 / 假优惠识别"使用。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Callable

from priceradar.agent.tools import compute_best_price, search_jd_channel, search_pdd_goods
from priceradar.config import project_root
from priceradar.domain.models import Offer
from priceradar.formatting import fmt_amount
from priceradar.sources.jd import JdApiError

DEFAULT_DB = project_root() / "data" / "snapshots.db"
WATCHLIST_PATH = project_root() / "data" / "watchlist.json"

_DEFAULT_WATCHLIST = {
    "keywords": ["QCY N20 降噪耳机", "索尼 WH-1000XM6 头戴式降噪耳机"],
    "jd_elite_id": 0,   # 0 = 自动品类路由（原为固定 24=数码，导致跨品类搜不到）
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS price_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_date TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    platform TEXT NOT NULL,
    sku_id TEXT NOT NULL,
    title TEXT NOT NULL,
    shop_name TEXT,
    list_price TEXT NOT NULL,
    final_price TEXT NOT NULL,
    total_discount TEXT NOT NULL,
    keyword TEXT NOT NULL,
    url TEXT,
    image_url TEXT
);
CREATE INDEX IF NOT EXISTS idx_snap_lookup ON price_snapshots(platform, sku_id, captured_date);
CREATE UNIQUE INDEX IF NOT EXISTS idx_snap_unique ON price_snapshots(platform, sku_id, captured_date);
"""


def connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.executescript(SCHEMA)
    _migrate(connection)
    return connection


def _migrate(connection: sqlite3.Connection) -> None:
    """轻量迁移：老库补 url / image_url 列（关注面板展示用）。"""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(price_snapshots)")}
    for column in ("url", "image_url"):
        if column not in columns:
            connection.execute(f"ALTER TABLE price_snapshots ADD COLUMN {column} TEXT")
    connection.commit()


def record_offers(connection: sqlite3.Connection, offers: list[Offer], keyword: str, captured: datetime | None = None) -> int:
    """写入快照（幂等）。返回实际新增条数。"""
    captured = captured or datetime.now()
    date_iso = captured.date().isoformat()
    inserted = 0
    for offer in offers:
        breakdown = compute_best_price(offer)
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO price_snapshots
                (captured_date, captured_at, platform, sku_id, title, shop_name, list_price, final_price, total_discount, keyword, url, image_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date_iso,
                captured.isoformat(timespec="seconds"),
                offer.platform.value,
                offer.sku_id,
                offer.title,
                offer.shop_name or "",
                str(offer.list_price),
                str(breakdown.final_price),
                str(breakdown.total_discount),
                keyword,
                offer.url or "",
                offer.image_url or "",
            ),
        )
        inserted += 1 if cursor.rowcount and cursor.rowcount > 0 else 0
        # 回填：今日已存在但 url/图片为空的行（老库迁移后补上展示字段；不影响幂等计数）
        connection.execute(
            """
            UPDATE price_snapshots SET url = ?, image_url = ?
            WHERE platform = ? AND sku_id = ? AND captured_date = ? AND (url IS NULL OR url = '')
            """,
            (offer.url or "", offer.image_url or "", offer.platform.value, offer.sku_id, date_iso),
        )
    connection.commit()
    return inserted


def history(connection: sqlite3.Connection, platform: str, sku_id: str) -> list[tuple[str, str]]:
    """某商品的观测历史：[(日期, 到手价字符串), ...]（按日期升序）。"""
    rows = connection.execute(
        "SELECT captured_date, final_price FROM price_snapshots WHERE platform = ? AND sku_id = ? ORDER BY captured_date",
        (platform, sku_id),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def collect(
    keywords: list[str],
    db_path: Path = DEFAULT_DB,
    *,
    jd_enabled: bool = True,
    jd_elite_id: int | None = None,
    pdd_page_size: int = 20,
    jd_pool_size: int = 30,
    pdd_searcher: Callable[[str, int], list[Offer]] | None = None,
    jd_searcher: Callable[[str], list[Offer]] | None = None,
) -> dict:
    """对关键词列表执行一次采集。searcher 可注入（测试离线化）。

    京东侧默认走品类路由（与管线一致）；显式 `jd_elite_id` 则定点拉该频道。
    """
    pdd_searcher = pdd_searcher or (lambda kw, size: search_pdd_goods(kw, size))
    if jd_searcher is None:
        jd_searcher = (
            (lambda _kw: search_jd_channel(jd_elite_id, jd_pool_size))
            if jd_elite_id is not None
            else search_jd_goods
        )

    connection = connect(db_path)
    summary: dict = {"inserted": 0, "keywords": {}, "errors": []}
    try:
        for keyword in keywords:
            offers: list[Offer] = []
            try:
                offers.extend(pdd_searcher(keyword, pdd_page_size))
            except Exception as exc:  # 单平台失败不阻塞其他关键词
                summary["errors"].append(f"拼多多[{keyword}]：{exc}")
            if jd_enabled:
                try:
                    offers.extend(jd_searcher(keyword))
                except (JdApiError, OSError) as exc:
                    summary["errors"].append(f"京东[{keyword}]：{exc}")
            inserted = record_offers(connection, offers, keyword)
            summary["keywords"][keyword] = {"fetched": len(offers), "inserted": inserted}
            summary["inserted"] += inserted
    finally:
        connection.close()
    return summary


def report(connection: sqlite3.Connection) -> str:
    """输出快照库概况（商品数 / 观测天数 / 最新价格）。"""
    total = connection.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0]
    if total == 0:
        return "快照库为空——先运行一次采集。"
    spans = connection.execute("SELECT MIN(captured_date), MAX(captured_date) FROM price_snapshots").fetchone()
    lines = [f"快照库概况：{total} 行 ｜ 日期范围 {spans[0]} ~ {spans[1]}", ""]
    rows = connection.execute(
        """
        SELECT platform, sku_id, MAX(title), COUNT(DISTINCT captured_date) AS days,
               (SELECT final_price FROM price_snapshots p2 WHERE p2.platform = p.platform AND p2.sku_id = p.sku_id
                ORDER BY captured_date DESC LIMIT 1) AS latest
        FROM price_snapshots p GROUP BY platform, sku_id ORDER BY days DESC, sku_id LIMIT 10
        """,
    ).fetchall()
    lines.append("按商品（观测天数倒序，前 10）：")
    for platform, sku_id, title, days, latest in rows:
        lines.append(f"  - [{platform}] {title[:26]} ｜ 观测 {days} 天 ｜ 最新到手 ￥{fmt_amount(Decimal(latest))}")
    return "\n".join(lines)


def load_watchlist(path: Path | None = None) -> dict:
    """读取关注清单；不存在时写入模板并返回。

    ⚠️ `path` 默认值在**调用时**解析（不用 `= WATCHLIST_PATH` 当默认参数）——
    写成默认参数会在导入时绑定，测试打桩 `WATCHLIST_PATH` 就失效了。
    同一个坑本项目在 `sources/retry.py` 的 `sleeper` 上已经踩过一次。
    """
    path = Path(path or WATCHLIST_PATH)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_DEFAULT_WATCHLIST, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已创建关注清单模板：{path}（可编辑 keywords 后每日重跑）")
    return json.loads(path.read_text(encoding="utf-8"))


def save_watchlist(watchlist: dict, path: Path | None = None) -> None:
    """写回关注清单。

    与 `load_watchlist` 放在一起，免得读与写的默认值各在一处、日后改一个漏一个。
    """
    path = Path(path or WATCHLIST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(watchlist, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalise_keywords(keywords: list[str]) -> list[str]:
    """去空白、去重、保序（顺手丢掉空串）。"""
    seen: set[str] = set()
    result: list[str] = []
    for raw in keywords or []:
        keyword = str(raw).strip()
        if keyword and keyword not in seen:
            seen.add(keyword)
            result.append(keyword)
    return result


def add_watch_keyword(keyword: str, path: Path | None = None) -> tuple[list[str], bool]:
    """加入关注清单。返回 (清单, 是否新增)——已存在时不重复添加。"""
    keyword = str(keyword).strip()
    if not keyword:
        raise ValueError("关注关键词不能为空")
    watchlist = load_watchlist(path)
    keywords = _normalise_keywords(watchlist.get("keywords", []))
    if keyword in keywords:
        return keywords, False
    keywords.append(keyword)
    watchlist["keywords"] = keywords
    save_watchlist(watchlist, path)
    return keywords, True


def remove_watch_keyword(keyword: str, path: Path | None = None) -> tuple[list[str], bool]:
    """从关注清单移除。返回 (清单, 是否真的删掉了)。"""
    keyword = str(keyword).strip()
    watchlist = load_watchlist(path)
    keywords = _normalise_keywords(watchlist.get("keywords", []))
    if keyword not in keywords:
        return keywords, False
    keywords = [item for item in keywords if item != keyword]
    watchlist["keywords"] = keywords
    save_watchlist(watchlist, path)
    return keywords, True


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="真价雷达 · 价格快照采集（幂等，可每日复跑）")
    parser.add_argument("--keywords", nargs="*", default=None, help="临时关键词（缺省读取 data/watchlist.json）")
    parser.add_argument("--jd-channel", type=int, default=None, help="京东频道 eliteId（0 关闭；缺省用关注清单配置）")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="快照库路径")
    parser.add_argument("--report", action="store_true", help="仅查看已采集概况")
    args = parser.parse_args(argv)

    if args.report:
        connection = connect(args.db)
        print(report(connection))
        connection.close()
        return

    watchlist = load_watchlist()
    keywords = args.keywords or watchlist.get("keywords", [])
    jd_elite_id = args.jd_channel if args.jd_channel is not None else watchlist.get("jd_elite_id", 0)

    summary = collect(
        keywords,
        args.db,
        jd_enabled=jd_elite_id >= 0,
        jd_elite_id=jd_elite_id if jd_elite_id > 0 else None,
    )
    print(f"采集完成：新增 {summary['inserted']} 条快照")
    for keyword, stats in summary["keywords"].items():
        print(f"  - 「{keyword}」：检索 {stats['fetched']} 条，新增 {stats['inserted']} 条（同日重复已幂等跳过）")
    for error in summary["errors"]:
        print(f"  △ {error}")
    connection = connect(args.db)
    print()
    print(report(connection))
    connection.close()


if __name__ == "__main__":
    main()
