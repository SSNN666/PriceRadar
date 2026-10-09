"""真价雷达 Web 服务（M1 首版）：FastAPI + 单页前端。

- GET /            ：单页界面（输入 → 同款组 + 到手价对比卡）
- GET /api/search  ：查询接口（q / jd_channel / llm），返回结构化 JSON

设计：**服务层只做 HTTP 与序列化**——业务编排在 `agent/actions`（与 MCP Server、
Agent 循环共用同一份实现），这里不再自己拼流程。
LLM 默认启用（有 key 时），失败自动确定性兜底。
"""

from __future__ import annotations

import json
import threading
import urllib.error
from datetime import date
from pathlib import Path
from queue import Empty, Queue
from typing import Iterator

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from priceradar.agent import actions
from priceradar.agent.actions import compare_prices
from priceradar.agent.loop import run_agent
from priceradar.agent.pipeline import PipelineResult
from priceradar.domain.models import CERTIFIED_SHOP_KINDS, SHOP_KIND_LABELS, Offer
from priceradar.engine.pricing import PriceBreakdown
from priceradar.formatting import fmt_amount
from priceradar.insights import assess_quality, campaign_context, decide, price_drops
from priceradar.report import PLATFORM_LABELS, REASON_LABELS
from priceradar.sources.pdd import PddApiError
from priceradar.vision import CouponExtraction, to_offer_coupons

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="PriceRadar 真价雷达", version="0.1.0")
_snapshot_lock = threading.Lock()  # 防止并发重复采集


# —— 错误出口：所有响应必须是 JSON ——
# 起因：未捕获的异常会被 uvicorn 渲染成纯文本 "Internal Server Error"，
# 前端 resp.json() 解析失败，用户只看到 "Unexpected token 'I'"——
# 真正的原因（网络断了 / 接口挂了）被完全掩盖。下面把每类失败都变成可读 JSON。


async def _network_error_handler(request, exc) -> JSONResponse:
    """外部接口不可达（DNS / 超时 / 连接被拒）→ 502 + 人话。"""
    reason = getattr(exc, "reason", None) or exc
    return JSONResponse(
        status_code=502,
        content={"detail": f"外部接口网络不可达：{reason}。请检查网络或代理后重试。"},
    )


for _network_exc in (urllib.error.URLError, TimeoutError, ConnectionError):
    app.add_exception_handler(_network_exc, _network_error_handler)


async def _unexpected_error_handler(request, exc) -> JSONResponse:
    """兜底：任何未预期异常也返回 JSON（含异常类型与消息，便于自查）。

    uvicorn 控制台仍会打印完整堆栈；这里只是不让用户对着解析错误猜。
    若日后公网部署担心泄露内部信息，可在此按环境变量收敛为通用文案。
    """
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务内部错误：{type(exc).__name__}: {exc}"},
    )


app.add_exception_handler(Exception, _unexpected_error_handler)


def offer_payload(offer: Offer, breakdown: PriceBreakdown, *, group_count: int = 1, history=None) -> dict:
    """单个报价 → 前端可消费的 dict（`/api/search` 与 `/api/similar` 共用）。"""
    return {
        "platform": PLATFORM_LABELS[offer.platform],
        "title": offer.title,
        "shop": offer.shop_name or "",
        "url": offer.url or "",
        "image_url": offer.image_url or "",
        "shop_kind": offer.shop_kind.value if offer.shop_kind else "",
        "shop_kind_label": SHOP_KIND_LABELS.get(offer.shop_kind, ""),
        "quality": assess_quality(offer).model_dump(mode="json"),
        "brand": offer.brand or "",
        "tags": list(offer.tags),
        "goods_sign": offer.source_extra.get("goods_sign", ""),  # 前端「找相似」的回调凭据
        "signals": (
            {
                "good_comments_share": offer.signals.good_comments_share,
                "comments": offer.signals.comments,
                "in_order_count_30d": offer.signals.in_order_count_30d,
                "sales_tip": offer.signals.sales_tip,
                "sales_estimate": offer.signals.sales_estimate,
                "shop_score_tier": offer.signals.shop_score_tier,
            }
            if offer.signals
            else None
        ),
        "final_price": fmt_amount(breakdown.final_price),
        "total_discount": fmt_amount(breakdown.total_discount),
        "formula": breakdown.formula(),
        "lines": [{"label": item.label, "amount": fmt_amount(item.amount)} for item in breakdown.applied],
        "rejected": [{"label": item.label, "reason": REASON_LABELS[item.reason]} for item in breakdown.rejected],
        "group_count": group_count,
        "history": history,
    }


def days_to_campaign(today: date | None = None) -> int | None:
    """距最近一场大促的天数（无则 None）——供决策融合的"观望"档使用。"""
    from priceradar.insights import upcoming_campaigns

    today = today or date.today()
    upcoming = upcoming_campaigns(today)
    if not upcoming:
        return None
    return max(0, min((campaign.start - today).days for campaign in upcoming))


def to_payload(result: PipelineResult) -> dict:
    """把管线结果序列化为前端可消费的 dict（纯函数，单测覆盖）。"""
    count_by_sku = {group.representative.sku_id: group.count for group in result.groups}
    offers = [
        offer_payload(
            offer,
            breakdown,
            group_count=count_by_sku.get(offer.sku_id, 1),
            history=result.history.get(f"{offer.platform.value}:{offer.sku_id}"),
        )
        for offer, breakdown in result.breakdowns
    ]
    risk_topics = (
        [(issue["topic"], issue["mention_count"]) for issue in result.risk.issues] if result.risk is not None else None
    )
    decision = decide(
        result.breakdowns,
        history_by_sku=result.history,
        days_to_campaign=days_to_campaign(),
        risk_topics=risk_topics,
    )

    return {
        "keyword": result.keyword,
        "decision": decision.model_dump(mode="json"),
        # 资质店铺集合由后端定义（单一事实来源），前端筛选直接消费，避免两边各写一份
        "certified_shop_kinds": sorted(kind.value for kind in CERTIFIED_SHOP_KINDS),
        "pdd_count": result.pdd_count,
        "jd_count": result.jd_count,
        "summary": result.summary,
        "notes": result.notes,
        "cross_platform_hit": result.cross_platform_hit,
        "campaign_context": campaign_context(),
        "risk": (
            {
                "queries": result.risk.queries,
                "item_count": len(result.risk.items),
                "issues": result.risk.issues,
                "note": result.risk.note,
                "degraded": result.risk.degraded,
            }
            if result.risk is not None
            else None
        ),
        "source": {
            "title": result.source.title,
            "platform": PLATFORM_LABELS[result.source.platform],
            "price": fmt_amount(result.source.list_price),
        },
        "groups": [
            {
                "platform": PLATFORM_LABELS[group.representative.platform],
                "title": group.representative.title,
                "count": group.count,
                "reason": group.reason,
                "is_source_group": index == 0,
                "url": group.representative.url or "",
                "image_url": group.representative.image_url or "",
            }
            for index, group in enumerate(result.groups)
        ],
        "offers": offers,
        "similar_items": [
            {
                "platform": PLATFORM_LABELS[item.offer.platform],
                "title": item.offer.title,
                "url": item.offer.url or "",
                "image_url": item.offer.image_url or "",
                "price": fmt_amount(item.offer.list_price),
                "reason": item.reason,
            }
            for item in result.similar_items
        ],
        "candidates": {"similar": result.similar_count, "excluded": result.excluded_summary},
        "trace": result.trace.summary_line() if result.trace else None,
    }


def execute_search(query: str, jd_channel: int, use_llm: bool, use_risk: bool = False) -> dict:
    """执行一次查询（业务编排在 `agent.actions`，这里只做 HTTP 层的序列化）。

    测试可直接打桩此函数（签名与返回值契约保持不变）。
    """
    result = compare_prices(query, jd_channel=jd_channel, use_risk=use_risk, use_llm=use_llm)
    return to_payload(result)


# 用户补录券并入的实现已下沉到 `agent.actions`（MCP 侧也要用，放服务层会造成反向依赖）。
# 这里保留同名再导出，既有的调用点与测试打桩不受影响。
merge_user_coupons = actions.merge_user_coupons
apply_user_coupons = actions.apply_user_coupons


def execute_similar(goods_sign: str, limit: int = 20) -> dict:
    """以某个商品为锚点找相似商品（平台原生 `channel_type=3`）。

    与关键词搜索的区别：关键词搜"保温杯"会返回一堆款式各异的商品；以某个商品为锚点
    则由**平台自己**给出相似商品——这才对应"看到一件东西，想找一样的但更便宜的"。

    编排在 `agent.actions.find_similar`；这里只做序列化。
    """
    result = actions.find_similar(goods_sign, limit=limit)
    return {
        "anchor_sign": result.anchor_sign,
        "count": len(result.breakdowns),
        "offers": [offer_payload(offer, breakdown, group_count=1) for offer, breakdown in result.breakdowns],
        "notes": result.notes,
    }


# 本层独占的事件名（上游误发同名事件会被丢弃，见 `execute_agent`）
_TRANSPORT_EVENTS = frozenset({"final", "error"})


def _sse(event: dict, *, event_id: int | None = None) -> str:
    """一个 SSE 帧。`id:` 用于断线重连时的 Last-Event-ID 回放。"""
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event['type']}")
    lines.append(f"data: {json.dumps(event, ensure_ascii=False)}")
    return "\n".join(lines) + "\n\n"


KEEP_TURNS = 3              # 上下文里保留的最近完整轮数
CONTEXT_CHAR_BUDGET = 16000  # 组装历史时的预算（给本轮工具结果留出余量）


def _prepare_session(query: str, session_id: str | None) -> tuple[str, int, list[dict], str]:
    """准备会话：取槽位与历史 → 组装本轮消息 → 落一条 user 消息。

    返回 `(session_id, turn_id, messages, trim_note)`。全部是本地 sqlite 操作，毫秒级。
    """
    from priceradar.agent.context import SessionState, build_messages
    from priceradar.agent.loop import SYSTEM_PROMPT
    from priceradar.agent.session_store import (
        append_message,
        connect_session_db,
        load_state,
        load_turns,
        new_session,
        next_turn_id,
        session_exists,
    )

    connection = connect_session_db()
    try:
        sid = session_id or new_session(connection)
        if not session_exists(connection, sid):
            new_session(connection, sid)
        state = SessionState.from_dict(load_state(connection, sid))
        turns = load_turns(connection, sid, keep_turns=KEEP_TURNS)
        messages, trim_note = build_messages(
            SYSTEM_PROMPT, state, turns, query, char_budget=CONTEXT_CHAR_BUDGET
        )
        turn_id = next_turn_id(connection, sid)
        append_message(connection, sid, role="user", turn_id=turn_id, content=query)
        return sid, turn_id, messages, trim_note
    finally:
        connection.close()


def _persist_turn(session_id: str, turn_id: int, run, initial_messages: list[dict]) -> None:
    """把本轮新产生的 assistant / tool 消息与更新后的槽位落库。

    落库失败不能影响已经拿到的结果——由调用方转成提示事件。
    """
    from priceradar.agent.context import SessionState, update_state
    from priceradar.agent.session_store import (
        append_message,
        connect_session_db,
        load_state,
        prune,
        save_state,
    )

    produced = list(run.messages[len(initial_messages):]) if len(run.messages) > len(initial_messages) else []
    if not produced and run.final_text:
        # 降级路径可能一条助手消息都没产生（例如 LLM 不可用）——补一条，
        # 否则下一轮会读到「只有用户发言」的半截历史
        produced = [{"role": "assistant", "content": run.final_text}]

    connection = connect_session_db()
    try:
        for message in produced:
            append_message(
                connection,
                session_id,
                role=message["role"],
                turn_id=turn_id,
                content=message.get("content"),
                tool_calls=message.get("tool_calls"),
                tool_call_id=message.get("tool_call_id"),
            )
        state = update_state(SessionState.from_dict(load_state(connection, session_id)), run)
        save_state(connection, session_id, state.to_dict())
        prune(connection, session_id)
    finally:
        connection.close()


def execute_agent(
    query: str,
    jd_channel: int = 0,
    use_llm: bool = True,
    use_risk: bool = False,
    *,
    session_id: str | None = None,
    heartbeat: float = 15.0,
) -> Iterator[str]:
    """Agent 循环 → SSE 帧的**同步生成器**。

    为什么用同步生成器：内部全是阻塞的 urllib 调用。Starlette 的
    `StreamingResponse` 对非 async 迭代器会走 `iterate_in_threadpool`（不阻塞事件
    循环），而 async 生成器里做阻塞调用会卡死整个进程。

    为什么把 Agent 放进**后台线程 + 队列**：`on_event` 是在循环内部同步调用的，
    没法从回调里 yield。用队列把两侧解耦，事件才能真正**边跑边发**——
    用户在工具还在执行时就能看到进度，而不是等全部跑完一次性涌出来。

    断连处理（诚实版）：Starlette 的 `iterate_in_threadpool` **不会** close 底层
    迭代器（已读源码确认），所以生成器的 `finally` 不会被及时执行，取消逻辑不能
    建在它上面。这里的取舍是：断连后这次运行照常跑完（会话/结果不丢），
    代价是一次白跑。真要取消得靠 on_event 里检查连接状态，暂不做。
    """
    if not use_llm:
        yield _sse({"type": "degraded", "reason": "已关闭 LLM，走确定性比价"}, event_id=0)
        yield _sse(_final_event(query, jd_channel, use_risk, llm_client=None), event_id=1)
        return

    sid, turn_id, messages, trim_note = _prepare_session(query, session_id)

    events: Queue[dict] = Queue()
    prelude: list[dict] = [{"type": "session", "session_id": sid}]
    if trim_note:
        prelude.append({"type": "notice", "message": trim_note})

    def worker() -> None:
        try:
            run = run_agent(
                query,
                messages=messages,
                on_event=events.put,
                jd_channel=jd_channel,
                use_risk=use_risk,
            )
        except Exception as exc:  # 首个 chunk 之后异常处理器已失效，必须自己转成事件
            events.put({"type": "__error", "message": f"{type(exc).__name__}: {exc}"})
            return
        try:
            _persist_turn(sid, turn_id, run, messages)
        except Exception as exc:  # 落库失败不能吞掉已经拿到的结果，如实提示
            events.put({"type": "notice", "message": f"本轮对话未能存入会话（不影响结果）：{exc}"})
        events.put({"type": "__done", "run": run})

    threading.Thread(target=worker, daemon=True).start()

    counter = 0
    for event in prelude:
        counter += 1
        yield _sse(event, event_id=counter)

    while True:
        try:
            event = events.get(timeout=heartbeat)
        except Empty:
            yield ": ping\n\n"  # SSE 注释行做心跳，防代理按空闲超时切断
            continue
        if event["type"] == "__done":
            run = event["run"]
            counter += 1
            yield _sse(_agent_result_event(run, jd_channel, use_risk, session_id=sid), event_id=counter)
            return
        if event["type"] == "__error":
            counter += 1
            yield _sse({"type": "error", "message": event["message"]}, event_id=counter)
            return
        if event.get("type") in _TRANSPORT_EVENTS:
            # 本层**独占** final / error：上游（或日后新增的节点）误发同名事件一律丢弃。
            # 起因：loop 曾自己 emit 一个 {"type":"final"}（只有 text、没有 payload），
            # 前端收到就先 close 了，真正的 final 再也到不了，比价卡片永远出不来。
            continue
        counter += 1
        yield _sse(event, event_id=counter)


def _agent_result_event(run, jd_channel: int, use_risk: bool, *, session_id: str = "") -> dict:
    """终帧：既有前端 payload + Agent 自己的元信息。"""
    payload = to_payload(run.result) if run.result is not None else None
    return {
        "type": "final",
        "stopped_by": run.stopped_by,
        "degraded": run.degraded,
        "final_text": run.final_text,
        "tools": [execution.name for step in run.steps for execution in step.tool_calls],
        "rounds": len(run.steps),
        "session_id": session_id,
        "usage": {
            "prompt_tokens": run.usage.prompt_tokens,
            "completion_tokens": run.usage.completion_tokens,
            "cache_hit_rate": round(run.usage.cache_hit_rate, 4),
        },
        "payload": payload,
    }


def _final_event(query: str, jd_channel: int, use_risk: bool, *, llm_client) -> dict:
    """不走 Agent 的直通终帧（`llm=false` 时）。"""
    result = compare_prices(query, jd_channel=jd_channel, use_risk=use_risk, llm_client=llm_client)
    return {
        "type": "final",
        "stopped_by": "llm_disabled",
        "degraded": True,
        "final_text": result.summary or "",
        "tools": [],
        "rounds": 0,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cache_hit_rate": 0.0},
        "payload": to_payload(result),
    }


@app.get("/api/agent/stream")
def api_agent_stream(
    q: str = Query(..., min_length=1, description="自然语言或关键词"),
    jd_channel: int = 0,
    llm: bool = True,
    risk: bool = False,
    session_id: str | None = Query(None, description="会话 ID；不传则新建，追问时带上上一轮返回的那个"),
) -> StreamingResponse:
    """Agent 流式查询：边跑边推工具进度，最后给完整 payload。

    响应头三件套（部署到 nginx 时缺 `X-Accel-Buffering: no` 流会被缓冲成一坨）。
    """
    return StreamingResponse(
        execute_agent(q, jd_channel=jd_channel, use_llm=llm, use_risk=risk, session_id=session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/api/sessions")
def api_list_sessions(limit: int = 20) -> dict:
    """最近的会话（供「继续上次」用）。"""
    from priceradar.agent.session_store import connect_session_db, list_sessions

    connection = connect_session_db()
    try:
        return {"sessions": list_sessions(connection, limit=limit)}
    finally:
        connection.close()


@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str) -> dict:
    """单个会话的消息与槽位（刷新页面后恢复对话用）。"""
    from priceradar.agent.session_store import connect_session_db, load_state, load_turns, session_exists

    connection = connect_session_db()
    try:
        if not session_exists(connection, session_id):
            raise HTTPException(status_code=404, detail="会话不存在")
        return {
            "session_id": session_id,
            "state": load_state(connection, session_id),
            "turns": load_turns(connection, session_id, keep_turns=None),
        }
    finally:
        connection.close()


@app.delete("/api/sessions/{session_id}")
def api_delete_session(session_id: str) -> dict:
    """删除会话（消息与槽位一并删除）。"""
    from priceradar.agent.session_store import connect_session_db, delete_session, session_exists

    connection = connect_session_db()
    try:
        existed = session_exists(connection, session_id)
        delete_session(connection, session_id)
    finally:
        connection.close()
    return {"deleted": existed, "session_id": session_id}


def execute_snapshot() -> dict:
    """执行一次快照采集（关注清单 → collect → 概况）。测试可直接打桩此函数。"""
    from priceradar.jobs.snapshot import DEFAULT_DB, collect, connect, load_watchlist, report

    watchlist = load_watchlist()
    keywords = watchlist.get("keywords", [])
    summary = collect(keywords, DEFAULT_DB, jd_elite_id=watchlist.get("jd_elite_id", 24) or None)

    connection = connect(DEFAULT_DB)
    try:
        snapshot_report = report(connection)
    finally:
        connection.close()
    return {
        "inserted": summary["inserted"],
        "keywords": summary["keywords"],
        "errors": summary["errors"],
        "report": snapshot_report,
    }


@app.get("/api/search")
def api_search(
    q: str = Query(..., min_length=1, description="自然语言或关键词"),
    jd_channel: int = 0,
    llm: bool = True,
    risk: bool = False,
) -> dict:
    try:
        return execute_search(q, jd_channel, llm, risk)
    except PddApiError as exc:
        raise HTTPException(status_code=502, detail=f"拼多多检索失败：{exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/similar")
def api_similar(goods_sign: str = Query(..., min_length=1), limit: int = 20) -> dict:
    """以某个商品为锚点找相似商品（前端「找相似」按钮）。"""
    try:
        return execute_similar(goods_sign, limit=limit)
    except PddApiError as exc:
        raise HTTPException(status_code=502, detail=f"拼多多检索失败：{exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/snapshot")
def api_snapshot() -> dict:
    """手动触发一次快照采集（按 data/watchlist.json 关注清单，同日幂等）。"""
    if not _snapshot_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="采集已在运行中，请稍候")
    try:
        return execute_snapshot()
    finally:
        _snapshot_lock.release()


def _watchlist_overview(connection, limit: int = 8) -> list[dict]:
    """关注面板数据：观测天数最多的商品（标题/最新价/观测天数/时机判断/趋势序列）。"""
    from priceradar.insights import sku_history_view

    rows = connection.execute(
        """
        SELECT p.platform, p.sku_id, MAX(p.title) AS title,
               (SELECT p2.url FROM price_snapshots p2
                WHERE p2.platform = p.platform AND p2.sku_id = p.sku_id AND p2.url IS NOT NULL AND p2.url != ''
                ORDER BY p2.captured_date DESC LIMIT 1) AS url,
               (SELECT p2.image_url FROM price_snapshots p2
                WHERE p2.platform = p.platform AND p2.sku_id = p.sku_id AND p2.image_url IS NOT NULL AND p2.image_url != ''
                ORDER BY p2.captured_date DESC LIMIT 1) AS image_url
        FROM price_snapshots p GROUP BY p.platform, p.sku_id
        ORDER BY COUNT(DISTINCT p.captured_date) DESC, MAX(p.captured_at) DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    overview: list[dict] = []
    for platform, sku_id, title, url, image_url in rows:
        view = sku_history_view(connection, platform, sku_id)
        series = view.get("series", [])
        overview.append(
            {
                "platform": platform,
                "title": title,
                "url": url or "",
                "image_url": image_url or "",
                "observed_days": view["observed_days"],
                "latest_price": series[-1]["price"] if series else "",
                "verdict": view["verdict"],
                "series": series,
                "fake_discount": view.get("fake_discount"),
            }
        )
    return overview


def watchlist_status(db_path: Path | None = None) -> dict:
    """快照按钮配套状态：今日是否已采集 + 降价提醒 + 关注面板（库不存在返回空态；测试可注入路径）。"""
    from priceradar.jobs.snapshot import DEFAULT_DB, connect

    path = db_path or DEFAULT_DB
    payload: dict = {"captured_today": False, "rows_today": 0, "total_rows": 0, "drops": [], "watchlist": []}
    if not path.exists():
        return payload
    connection = connect(path)
    try:
        today = date.today().isoformat()
        payload["rows_today"] = connection.execute(
            "SELECT COUNT(*) FROM price_snapshots WHERE captured_date = ?", (today,)
        ).fetchone()[0]
        payload["total_rows"] = connection.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0]
        payload["captured_today"] = payload["rows_today"] > 0
        payload["drops"] = [
            {
                "platform": drop.platform,
                "title": drop.title,
                "previous_price": fmt_amount(drop.previous_price),
                "latest_price": fmt_amount(drop.latest_price),
                "drop_pct": drop.drop_pct,
                "previous_date": drop.previous_date,
                "latest_date": drop.latest_date,
                "new_low": drop.new_low,
            }
            for drop in price_drops(connection)[:10]
        ]
        payload["watchlist"] = _watchlist_overview(connection)
    finally:
        connection.close()
    return payload


@app.get("/api/watchlist-status")
def api_watchlist_status() -> dict:
    """快照按钮配套：今日采集状态 + 降价提醒清单 + 关注面板。"""
    return watchlist_status()


class WatchRequest(BaseModel):
    keyword: str = Field(min_length=1, max_length=80, description="要关注的检索关键词")


@app.get("/api/watchlist")
def api_get_watchlist() -> dict:
    """关注清单（每日快照采集的输入）。

    PRD §4.2 承诺的「帮我盯着降价」原先没有落点——只能手改
    `data/watchlist.json`，多轮 UI 上点它就是个死胡同。这三个接口把它补上。
    """
    from priceradar.jobs.snapshot import load_watchlist

    watchlist = load_watchlist()
    return {"keywords": watchlist.get("keywords", []), "jd_elite_id": watchlist.get("jd_elite_id", 0)}


@app.post("/api/watchlist")
def api_add_watch(item: WatchRequest) -> dict:
    """加入关注清单（幂等：已存在时不重复添加）。"""
    from priceradar.jobs.snapshot import add_watch_keyword

    try:
        keywords, added = add_watch_keyword(item.keyword)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "keywords": keywords,
        "added": added,
        "note": f"已关注「{item.keyword}」，下次采集快照时一并采集" if added else f"「{item.keyword}」已在关注清单里",
    }


@app.delete("/api/watchlist")
def api_remove_watch(keyword: str = Query(..., min_length=1)) -> dict:
    """从关注清单移除（已采集的历史数据**不受影响**，只是不再继续跟踪）。"""
    from priceradar.jobs.snapshot import remove_watch_keyword

    keywords, removed = remove_watch_keyword(keyword)
    return {
        "keywords": keywords,
        "removed": removed,
        "note": f"已取消关注「{keyword}」（历史快照仍保留）" if removed else f"「{keyword}」不在关注清单里",
    }


def execute_vision(image_bytes: bytes, media_type: str, mode: str) -> dict:
    """识图提取（mode: product=商品识别 / coupon=券识别）。

    VLM 只做"看"：产出结构化字段由前端展示，商品识别后一键按关键词检索；
    券识别结果需用户确认后才用于计算（PRD §5.6）。未配置 key 时给出可读降级说明。
    """
    from priceradar.vision import extract_coupons, extract_product
    from priceradar.vision.vlm import VisionError

    try:
        if mode == "coupon":
            extraction = extract_coupons(image_bytes, media_type)
            return {
                "mode": "coupon",
                "platform": extraction.platform,
                "coupons": [coupon.model_dump(mode="json") for coupon in extraction.coupons],
                "note": "识别结果需用户确认后才会用于到手价计算（截图补录回路，PRD §5.6）",
            }
        attrs = extract_product(image_bytes, media_type)
        query = " ".join(attrs.keywords) if attrs.keywords else (attrs.model or attrs.category or "")
        return {
            "mode": "product",
            "is_product": attrs.is_product,
            "category": attrs.category,
            "brand": attrs.brand,
            "model": attrs.model,
            "color": attrs.color,
            "specs": attrs.specs,
            "keywords": attrs.keywords,
            "uncertain_fields": attrs.uncertain_fields,
            "confidence": attrs.confidence,
            "query": query,
        }
    except VisionError as exc:
        raise HTTPException(status_code=503, detail=f"识图暂不可用：{exc}") from exc


@app.post("/api/vision")
async def api_vision(file: UploadFile = File(...), mode: str = "product") -> dict:
    """上传图片做结构化识别（商品图 / 券截图）。"""
    if mode not in ("product", "coupon"):
        raise HTTPException(status_code=400, detail="mode 仅支持 product / coupon")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="空文件")
    if len(payload) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="图片过大（≤8MB，上传前请压缩）")
    return execute_vision(payload, file.content_type or "image/jpeg", mode)


@app.post("/api/coupons")
def api_confirm_coupons(extraction: CouponExtraction) -> dict:
    """用户确认后的券写入（§5.6：识别 → 用户确认 → 才入库；入库后查询自动并入重算）。"""
    from priceradar.user_store import connect_user_db, save_coupons

    coupons = to_offer_coupons(extraction)
    if not coupons:
        raise HTTPException(status_code=400, detail="没有可写入的券")
    connection = connect_user_db()
    try:
        stored = save_coupons(connection, extraction.platform, coupons)
    finally:
        connection.close()
    return {
        "stored": stored,
        "platform": extraction.platform,
        "note": f"已写入 {stored} 张用户券；下次查询将自动并入重算（明细含 [补录] 标记）",
    }


@app.get("/api/coupons")
def api_list_coupons(platform: str | None = None) -> dict:
    """列出已存用户券（管理视图）。"""
    from priceradar.user_store import DEFAULT_DB, connect_user_db, list_coupons

    if not Path(DEFAULT_DB).exists():
        return {"coupons": []}
    connection = connect_user_db()
    try:
        return {"coupons": list_coupons(connection, platform)}
    finally:
        connection.close()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


_FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect width="64" height="64" rx="14" fill="#0a6cff"/>
<circle cx="32" cy="32" r="21" fill="none" stroke="#ffffff" stroke-width="3" opacity="0.5"/>
<circle cx="32" cy="32" r="12" fill="none" stroke="#ffffff" stroke-width="3" opacity="0.75"/>
<circle cx="32" cy="32" r="3.5" fill="#ffffff"/>
<line x1="32" y1="32" x2="48" y2="16" stroke="#ffffff" stroke-width="3" stroke-linecap="round"/>
</svg>"""


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


def main() -> None:
    import argparse
    import sys
    import threading
    import time
    import urllib.request
    import webbrowser

    import uvicorn

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="真价雷达 Web 服务（默认自动打开浏览器）")
    parser.add_argument("--port", type=int, default=8000, help="服务端口（默认 8000）")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    if not args.no_open:

        def _open_when_ready() -> None:
            """轮询直到服务就绪再打开浏览器（最多等 15 秒，避免白屏）。"""
            deadline = time.time() + 15
            while time.time() < deadline:
                try:
                    urllib.request.urlopen(url, timeout=1)
                    webbrowser.open(url)
                    return
                except Exception:
                    time.sleep(0.3)

        threading.Thread(target=_open_when_ready, daemon=True).start()

    uvicorn.run("priceradar.service.api:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
