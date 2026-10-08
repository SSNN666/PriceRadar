"""用户数据存储（本地单用户）：用户补录的优惠券（PRD §5.6）。

- 与快照库分离（`data/user_data.db`）：快照是"观测数据"，这里是"用户资产"
- 券按 (platform, coupon_id) 去重（重复补录覆盖）
- **入库前提是用户已确认**（调用方保证：上传截图 → 识别 → 用户点确认 → 才调 save_coupons）
"""

from __future__ import annotations

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
    UNIQUE(platform, coupon_id)
);
"""


def connect_user_db(path: Path | None = None) -> sqlite3.Connection:
    path = path or DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    return connection


def save_coupons(connection: sqlite3.Connection, platform: str, coupons: list[Coupon]) -> int:
    """写入用户券（同平台同 id 覆盖）。返回写入条数。"""
    now = datetime.now().isoformat(timespec="seconds")
    for coupon in coupons:
        connection.execute(
            """
            INSERT OR REPLACE INTO user_coupons
                (platform, coupon_id, type, face_value, threshold, scope_note, valid_until, source, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
    connection.commit()
    return len(coupons)


def load_active_coupons(connection: sqlite3.Connection, platform: str, as_of: date | None = None) -> list[Coupon]:
    """读取某平台当前有效的用户券（过期不返回）。"""
    as_of = as_of or date.today()
    rows = connection.execute(
        "SELECT coupon_id, type, face_value, threshold, scope_note, valid_until, source, confidence "
        "FROM user_coupons WHERE platform = ? ORDER BY created_at",
        (platform,),
    ).fetchall()
    coupons: list[Coupon] = []
    for coupon_id, coupon_type, face_value, threshold, scope_note, valid_until, source, confidence in rows:
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
            )
        )
    return coupons


def list_coupons(connection: sqlite3.Connection, platform: str | None = None) -> list[dict]:
    """列出已存用户券（管理视图用）。"""
    query = (
        "SELECT platform, coupon_id, type, face_value, threshold, scope_note, valid_until, source, confidence "
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
        }
        for row in rows
    ]
