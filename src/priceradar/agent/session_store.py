"""会话存储：多轮对话的事实源（sqlite 标准库，零第三方依赖）。

**分库**：与 `data/snapshots.db`（观测数据）、`data/user_data.db`（用户资产）一样
刻意独立成 `data/sessions.db`——三类数据的生命周期与用途都不同，混在一起会让
"清空会话"变成一件危险的事。

约定沿用 `user_store.py`：
- 内联 `SCHEMA` + `connect_*()` 建表
- **函数一律接收 connection，不自己开连接**（事务边界留给调用方）
- 路由层 `try/finally: connection.close()`

设计要点（对应「多轮对话」面试里的那几个必答项）：
- `seq` **单调**：保证消息顺序，不靠时间戳（同一秒内会撞）
- `turn_id` **显式**：一轮 = 一次「用户提问 → 助手把话说完」。
  裁剪上下文时**只允许整轮丢弃**——半轮会让 `tool_calls` 与 `tool` 消息
  配不上对，DeepSeek 对 `tool_call_id` 是强校验，直接 400
- `state_json`：**结构化槽位**（当前关键词/商品/goods_sign），不是自然语言。
  它是历史被裁剪后仍然活下来的那部分
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from priceradar.config import project_root

DEFAULT_DB = project_root() / "data" / "sessions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    state_json  TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    session_id     TEXT    NOT NULL,
    seq            INTEGER NOT NULL,
    turn_id        INTEGER NOT NULL,
    role           TEXT    NOT NULL,
    content        TEXT,
    tool_calls_json TEXT,
    tool_call_id   TEXT,
    created_at     TEXT    NOT NULL,
    PRIMARY KEY (session_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_messages_turn ON messages(session_id, turn_id, seq);
"""

MAX_TURNS_KEPT = 40  # 单会话保留的轮次上限（防无限增长；超出从最老的整轮删起）


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect_session_db(path: Path | None = None) -> sqlite3.Connection:
    db_path = Path(path or DEFAULT_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.executescript(SCHEMA)
    return connection


# ---------- 会话 ----------

def new_session(connection: sqlite3.Connection, session_id: str | None = None) -> str:
    identifier = session_id or uuid.uuid4().hex[:16]
    timestamp = _now()
    connection.execute(
        "INSERT OR IGNORE INTO sessions(id, created_at, updated_at, state_json) VALUES (?, ?, ?, '{}')",
        (identifier, timestamp, timestamp),
    )
    connection.commit()
    return identifier


def session_exists(connection: sqlite3.Connection, session_id: str) -> bool:
    return connection.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is not None


def list_sessions(connection: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = connection.execute(
        """
        SELECT s.id, s.created_at, s.updated_at, s.state_json,
               (SELECT COUNT(DISTINCT turn_id) FROM messages m WHERE m.session_id = s.id) AS turns,
               (SELECT content FROM messages m WHERE m.session_id = s.id AND m.role = 'user'
                ORDER BY m.seq LIMIT 1) AS first_query
        FROM sessions s ORDER BY s.updated_at DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "id": session_id,
            "created_at": created_at,
            "updated_at": updated_at,
            "turns": turns,
            "first_query": first_query or "",
            "state": json.loads(state_json or "{}"),
        }
        for session_id, created_at, updated_at, state_json, turns, first_query in rows
    ]


def delete_session(connection: sqlite3.Connection, session_id: str) -> None:
    connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    connection.commit()


# ---------- 结构化槽位 ----------

def load_state(connection: sqlite3.Connection, session_id: str) -> dict:
    row = connection.execute("SELECT state_json FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        return {}
    try:
        state = json.loads(row[0] or "{}")
    except json.JSONDecodeError:
        return {}
    return state if isinstance(state, dict) else {}


def save_state(connection: sqlite3.Connection, session_id: str, state: dict) -> None:
    connection.execute(
        "UPDATE sessions SET state_json = ?, updated_at = ? WHERE id = ?",
        (json.dumps(state, ensure_ascii=False), _now(), session_id),
    )
    connection.commit()


# ---------- 消息 ----------

def append_message(
    connection: sqlite3.Connection,
    session_id: str,
    *,
    role: str,
    turn_id: int,
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    tool_call_id: str | None = None,
) -> int:
    """追加一条消息，返回它的 seq。seq 在会话内单调递增。"""
    seq = connection.execute(
        "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE session_id = ?", (session_id,)
    ).fetchone()[0]
    connection.execute(
        """
        INSERT INTO messages(session_id, seq, turn_id, role, content, tool_calls_json, tool_call_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            seq,
            turn_id,
            role,
            content,
            json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
            tool_call_id,
            _now(),
        ),
    )
    connection.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))
    connection.commit()
    return seq


def next_turn_id(connection: sqlite3.Connection, session_id: str) -> int:
    row = connection.execute(
        "SELECT COALESCE(MAX(turn_id), 0) + 1 FROM messages WHERE session_id = ?", (session_id,)
    ).fetchone()
    return int(row[0])


def _row_to_message(row) -> dict:
    _seq, role, content, tool_calls_json, tool_call_id = row
    message: dict = {"role": role}
    if content is not None:
        message["content"] = content
    if tool_calls_json:
        message["tool_calls"] = json.loads(tool_calls_json)
    if tool_call_id:
        message["tool_call_id"] = tool_call_id
    return message


def load_turns(connection: sqlite3.Connection, session_id: str, *, keep_turns: int | None = 2) -> list[list[dict]]:
    """取最近若干**完整轮次**，按轮分组返回（`keep_turns=None` 表示全取）。

    返回**分组**而不是打平，是为了让上层能做「整轮丢弃」的裁剪——
    留下半轮会让 `tool_calls` 与 `tool` 消息配不上对，下一轮请求会被服务端以
    `tool_call_id` 不成对直接拒掉（DeepSeek 对此是强校验）。
    """
    params: list = [session_id]
    limit_clause = ""
    if keep_turns is not None:
        turn_rows = connection.execute(
            "SELECT DISTINCT turn_id FROM messages WHERE session_id = ? ORDER BY turn_id DESC LIMIT ?",
            (session_id, keep_turns),
        ).fetchall()
        if not turn_rows:
            return []
        params.append(min(turn for (turn,) in turn_rows))
        limit_clause = " AND turn_id >= ?"

    rows = connection.execute(
        f"""
        SELECT turn_id, seq, role, content, tool_calls_json, tool_call_id FROM messages
        WHERE session_id = ?{limit_clause} ORDER BY seq
        """,
        params,
    ).fetchall()

    turns: list[list[dict]] = []
    current_id: int | None = None
    for turn_id, seq, role, content, tool_calls_json, tool_call_id in rows:
        if turn_id != current_id:
            turns.append([])
            current_id = turn_id
        turns[-1].append(_row_to_message((seq, role, content, tool_calls_json, tool_call_id)))
    return turns


def load_messages(connection: sqlite3.Connection, session_id: str, *, keep_turns: int | None = 2) -> list[dict]:
    """打平版的 `load_turns`（需要整轮裁剪时请直接用 `load_turns`）。"""
    return [message for turn in load_turns(connection, session_id, keep_turns=keep_turns) for message in turn]


def prune(connection: sqlite3.Connection, session_id: str, keep_turns: int = MAX_TURNS_KEPT) -> int:
    """只保留最近若干轮，返回删除的消息条数（防会话无限增长）。"""
    row = connection.execute(
        "SELECT DISTINCT turn_id FROM messages WHERE session_id = ? ORDER BY turn_id DESC LIMIT 1 OFFSET ?",
        (session_id, keep_turns - 1),
    ).fetchone()
    if row is None:
        return 0
    cutoff = row[0]
    cursor = connection.execute(
        "DELETE FROM messages WHERE session_id = ? AND turn_id < ?", (session_id, cutoff)
    )
    connection.commit()
    return cursor.rowcount
