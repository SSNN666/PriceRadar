"""真价雷达 Web 服务（M1 首版）：FastAPI + 单页前端。

- GET /            ：单页界面（输入 → 同款组 + 到手价对比卡）
- GET /api/search  ：查询接口（q / jd_channel / llm），返回结构化 JSON

设计：**服务层只做 HTTP 与序列化**——业务编排在 `agent/actions`（与 MCP Server、
Agent 循环共用同一份实现），这里不再自己拼流程。
LLM 默认启用（有 key 时），失败自动确定性兜底。
"""

from __future__ import annotations

import threading
import urllib.error
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from priceradar.agent import actions
from priceradar.agent.actions import compare_prices
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
