"""用户数据存储（本地单用户）：用户补录的优惠券（PRD §5.6）。

- 与快照库分离（`data/user_data.db`）：快照是"观测数据"，这里是"用户资产"
- 券按 (platform, coupon_id) 去重（重复补录覆盖）
- **入库前提是用户已确认**（调用方保证：上传截图 → 识别 → 用户点确认 → 才调 save_coupons）
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from priceradar.config import project_root
from priceradar.domain.models import Coupon, CouponType

DEFAULT_DB = project_root() / "data" / "user_data.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS user_coupons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform TEXT NOT NULL,
    coupon_id TEXT NOT NULL,
    type TEXT NOT NULL,
    face_value TEXT NOT NULL,
    threshold TEXT NOT NULL,
    scope_note TEXT NOT NULL DEFAULT '',
    valid_until TEXT,
    source TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    shop_id TEXT,
    shop_name TEXT,
    category_ids TEXT NOT NULL DEFAULT '[]',
    bound_label TEXT NOT NULL DEFAULT '',
    UNIQUE(platform, coupon_id)
);
"""

# 适用范围绑定的列（加列时只改这一处，_migrate 与 save 都从这里取）
_BINDING_COLUMNS = ("shop_id", "shop_name", "category_ids", "bound_label")


def _migrate(connection: sqlite3.Connection) -> None:
    """轻量迁移：老库补适用范围绑定的列（照 jobs/snapshot.py 的同款做法）。

    `CREATE TABLE IF NOT EXISTS` 对已存在的表是 no-op，新列不会自动出现；
    不补的话此后任何读写新列都会 `no such column` 直接炸。
    """
    columns = {row[1] for row in connection.execute("PRAGMA table_info(user_coupons)")}
    for column in _BINDING_COLUMNS:
        if column in columns:
            continue
        # category_ids 是唯一有默认值约束的（JSON 字符串），其余 TEXT 可为空
        default = " DEFAULT '[]'" if column == "category_ids" else ""
        connection.execute(f"ALTER TABLE user_coupons ADD COLUMN {column} TEXT{default}")
    connection.commit()


def connect_user_db(path: Path | None = None) -> sqlite3.Connection:
    path = path or DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    _migrate(connection)
    return connection


def save_coupons(connection: sqlite3.Connection, platform: str, coupons: list[Coupon]) -> int:
    """写入用户券（同平台同 id 覆盖）。返回写入条数。"""
    now = datetime.now().isoformat(timespec="seconds")
    for coupon in coupons:
        connection.execute(
            """
            INSERT OR REPLACE INTO user_coupons
                (platform, coupon_id, type, face_value, threshold, scope_note, valid_until,
                 source, confidence, created_at, shop_id, shop_name, category_ids, bound_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                platform,
                coupon.coupon_id,
                coupon.type.value,
                str(coupon.face_value),
                str(coupon.threshold),
                coupon.scope_note,
                coupon.valid_until.isoformat() if coupon.valid_until else None,
                coupon.source,
                coupon.confidence,
                now,
                coupon.shop_id,
                coupon.shop_name,
                json.dumps(coupon.category_ids, ensure_ascii=False),
                coupon.bound_label,
            ),
        )
    connection.commit()
    return len(coupons)


def _parse_category_ids(raw: str | None) -> list[str]:
    """库里存的是 JSON 字符串；任何异常一律当作"没有绑定"（宁可不用，不能算错）。"""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(value) for value in parsed] if isinstance(parsed, list) else []


def load_active_coupons(connection: sqlite3.Connection, platform: str, as_of: date | None = None) -> list[Coupon]:
    """读取某平台当前有效的用户券（过期不返回）。

    ⚠️ **这里刻意不按店铺/品类过滤**——它返回的是"这个平台的全部有效用户券"，
    适用范围判定由 `priceradar.coupon_scope.coupons_for_offer` 针对具体商品做。
    调用方**必须**过那道过滤，否则就会重现"店铺券打到别家店"的 bug。
    """
    as_of = as_of or date.today()
    rows = connection.execute(
        "SELECT coupon_id, type, face_value, threshold, scope_note, valid_until, source, confidence, "
        "shop_id, shop_name, category_ids, bound_label "
        "FROM user_coupons WHERE platform = ? ORDER BY created_at",
        (platform,),
    ).fetchall()
    coupons: list[Coupon] = []
    for (
        coupon_id, coupon_type, face_value, threshold, scope_note, valid_until, source, confidence,
        shop_id, shop_name, category_ids, bound_label,
    ) in rows:
        expiry = date.fromisoformat(valid_until) if valid_until else None
        if expiry is not None and expiry < as_of:
            continue
        coupons.append(
            Coupon(
                coupon_id=coupon_id,
                type=CouponType(coupon_type),
                face_value=Decimal(face_value),
                threshold=Decimal(threshold),
                scope_note=scope_note,
                valid_until=expiry,
                source=source,
                confidence=float(confidence),
                shop_id=shop_id,
                shop_name=shop_name,
                category_ids=_parse_category_ids(category_ids),
                bound_label=bound_label or "",
            )
        )
    return coupons


def delete_coupon(connection: sqlite3.Connection, platform: str, coupon_id: str) -> bool:
    """删除一张券。返回**是否真的删掉了**（幂等：不存在返回 False，不抛异常）。"""
    cursor = connection.execute(
        "DELETE FROM user_coupons WHERE platform = ? AND coupon_id = ?", (platform, coupon_id)
    )
    connection.commit()
    return bool(cursor.rowcount and cursor.rowcount > 0)


def list_coupons(connection: sqlite3.Connection, platform: str | None = None) -> list[dict]:
    """列出已存用户券（管理视图用）。"""
    query = (
        "SELECT platform, coupon_id, type, face_value, threshold, scope_note, valid_until, source, confidence, "
        "shop_id, shop_name, category_ids, bound_label, created_at "
        "FROM user_coupons"
    )
    params: tuple = ()
    if platform:
        query += " WHERE platform = ?"
        params = (platform,)
    query += " ORDER BY created_at DESC"
    rows = connection.execute(query, params).fetchall()
    return [
        {
            "platform": row[0],
            "coupon_id": row[1],
            "type": row[2],
            "face_value": row[3],
            "threshold": row[4],
            "scope": row[5],
            "valid_until": row[6],
            "source": row[7],
            "confidence": row[8],
            "shop_id": row[9] or "",
            "shop_name": row[10] or "",
            "category_ids": _parse_category_ids(row[11]),
            "bound_label": row[12] or "",
            "created_at": row[13],
        }
        for row in rows
    ]
