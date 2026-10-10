"""意图级能力层（actions）：Agent 循环 / MCP Server / Web 服务共用的编排入口。

为什么需要这一层：
- 此前「查询 → 检索 → 算价 → 报告」的编排在**三个地方各写了一遍**
  （`service/api.execute_search`、`mcp_server._compare_text`、以及各自的历史查询），
  且语义已经分叉（MCP 侧不含用户补录券、不做风险聚类）。这里统一成一份实现。
- Agent 的工具调用循环需要一份**可导出的参数 schema**——`agent/tools.py` 的
  `TOOLS` 注册表只有 name/description/func，没有 schema，且当时零消费者。
  本模块的 `Action.params` 是唯一来源：既能导出 OpenAI 兼容 schema，
  也能直接用于入参校验。

层次：本层**只编排、不序列化**——返回领域对象（`PipelineResult` /
`SimilarResult` / `RiskSummary` / `PriceHistory`），
序列化留在各自边缘（Web 用 `to_payload`、MCP 用 `_*_text`、Agent 用 `summarize_*`）。

重试策略（每条都有实测依据，不要凭直觉改）：
- **pdd 系一律 retry=none**：`sources/pdd.py` 的 `call_with_retry` 已经在客户端内部
  重试 3 次；到这一层还能看到的失败，要么是已经重试过 3 次的网络故障，要么是
  业务错误（签名/权限/参数）。再重试 = 3 倍配额，且会触发已知的**账号限流**
  （判据是「调用量 vs 出单量」，本项目账号天然不出单，见 docs/pdd-api-findings.md §10）。
- **bocha retry=once**：`sources/bocha.py` **没有**内部重试，且它用
  `raise BochaError(...) from error` **保留了 `__cause__`**——补一次重试是值的。
- **本地 sqlite retry=none**：没有网络面。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from priceradar.agent.llm import DeepSeekClient, LlmError
from priceradar.agent.llm_nodes import compose_summary, extract_keyword
from priceradar.agent.pipeline import PipelineResult, run
from priceradar.domain.models import Offer
from priceradar.engine.pricing import PriceBreakdown, compute_price
from priceradar.formatting import fmt_amount
from priceradar.insights import sku_history_view
from priceradar.risk import RiskSummary, cluster_issues, scan_with_env
from priceradar.sources.pdd import PddApiError, PddClient
from priceradar.sources.retry import is_transient

SUMMARY_LIMIT = 1200  # 回灌给 LLM 的工具结果上限（字符）。全量结果留在服务端，不进上下文。


# ============================================================
# 用户补录券并入（自 service 层下沉；两个消费者都要用）
# ============================================================

def merge_user_coupons(
    breakdowns: list[tuple[Offer, PriceBreakdown]],
) -> tuple[int, list[tuple[Offer, PriceBreakdown]]]:
    """在**已算好**的 breakdown 上并入用户补录券（无用户库/无券时原样返回、不重算）。

    刻意不重算：管线算价可能带 as_of 等参数，重算未必等价。
    """
    from priceradar.user_store import DEFAULT_DB, connect_user_db, load_active_coupons

    if not Path(DEFAULT_DB).exists():
        return 0, breakdowns
    connection = connect_user_db()
    try:
        merged = 0
        updated: list[tuple[Offer, PriceBreakdown]] = []
        for offer, breakdown in breakdowns:
            user_coupons = load_active_coupons(connection, offer.platform.value)
            if not user_coupons:
                updated.append((offer, breakdown))
                continue
            merged_offer = offer.model_copy(update={"coupons": list(offer.coupons) + user_coupons})
            updated.append((merged_offer, compute_price(merged_offer)))
            merged += len(user_coupons)
        return merged, updated
    finally:
        connection.close()


def apply_user_coupons(result: PipelineResult) -> int:
    """把已确认的用户补录券并入到手价重算。返回并入张数。"""
    merged, breakdowns = merge_user_coupons(result.breakdowns)
    result.breakdowns = breakdowns
    return merged


# ============================================================
# LLM 客户端（模块级名字：可被打桩，符合项目「client 可为 None」的约定）
# ============================================================

def _llm_client() -> DeepSeekClient | None:
    try:
        return DeepSeekClient.from_env()
    except LlmError:
        return None


_UNSET = object()  # 区分「没传」与「显式传 None（= 禁用 LLM）」


# ============================================================
# 意图级能力
# ============================================================

def compare_prices(
    query: str,
    jd_channel: int = 0,
    use_risk: bool = False,
    *,
    use_llm: bool = True,
    llm_client: Any = _UNSET,
    trace: Any = None,
) -> PipelineResult:
    """一个查询 → 完整比价结果（关键词理解 → 管线 → 用户券并入 → 导语 → 可选风险聚类）。

    这是 Web `/api/search` 与 MCP `compare_prices` 的**共同实现**——此前两边各写一份，
    且 MCP 侧漏了用户补录券与风险聚类。

    jd_channel：0=自动品类路由（默认）；-1=关闭京东侧；>0=定点拉该京粉频道。
    use_llm：关闭后走全确定性兜底（CLI `--no-llm` 与 Web `llm=false` 用）。
    """
    client = (_llm_client() if use_llm else None) if llm_client is _UNSET else llm_client
    sink = getattr(trace, "usage_sink", None)
    # 链接短路在 extract_keyword 内部（粘拼多多商品链接时原样透传，绝不清洗 URL）
    keyword, _ = extract_keyword(query, client, usage_sink=sink)
    result = run(
        keyword,
        jd_enabled=jd_channel >= 0,
        jd_elite_id=jd_channel if jd_channel > 0 else None,
        with_risk=use_risk,
        trace=trace,          # 管线内部的分页/检索次数也计入同一条 trace
    )
    merged = apply_user_coupons(result)
    if merged:
        result.notes.append(f"已并入 {merged} 张用户补录券（明细含 [补录] 标记）")
    result.summary, _ = compose_summary(result, client, usage_sink=sink)
    if result.risk is not None:
        result.risk = cluster_issues(result.risk, client, usage_sink=sink)
    return result


@dataclass
class SimilarResult:
    anchor_sign: str
    breakdowns: list[tuple[Offer, PriceBreakdown]]
    merged_coupons: int = 0
    notes: list[str] = field(default_factory=list)


def find_similar(goods_sign: str, limit: int = 20, *, llm_client: Any = _UNSET, trace: Any = None) -> SimilarResult:
    """以商品为锚点找相似（平台原生 `channel_type=3`）。

    关键词搜「保温杯」会返回一堆款式各异的商品；以商品为锚点则由**平台自己**
    给出相似商品——这才对应「看到一件东西，想找一样的但更便宜的」。
    """
    offers = PddClient.from_env().recommend_similar(goods_sign, limit=limit)
    if trace is not None:
        trace.count("pdd")
    if not offers:
        raise ValueError("未找到相似商品（该商品可能已下架，或平台无相似推荐）")

    merged, breakdowns = merge_user_coupons([(offer, compute_price(offer)) for offer in offers])
    notes = [
        "相似商品由平台推荐（pdd.ddk.goods.recommend.get, channel_type=3）返回，"
        "**混合有无关商品**——实测精度随品类差异大（保温杯 100% / 降噪耳机 ~47% / 连衣裙 ~33%），"
        "请自行核对。",
    ]
    if merged:
        notes.append(f"已并入 {merged} 张用户补录券（明细含 [补录] 标记）")
    return SimilarResult(anchor_sign=goods_sign, breakdowns=breakdowns, merged_coupons=merged, notes=notes)


def scan_risk(product: str, *, llm_client: Any = _UNSET, trace: Any = None) -> RiskSummary:
    """全网风险扫描（博查检索 → LLM 跨词聚类 → 覆盖声明）。

    无 LLM 时原样返回检索清单并保留 `degraded=True`——**如实标注未做归类**，
    不假装聚类过。
    """
    client = _llm_client() if llm_client is _UNSET else llm_client
    summary = scan_with_env(product)
    if trace is not None:
        # 博查按次计费（￥0.05/次），一组关键词一次检索——**它是单次查询里最贵的部分**，
        # 比循环自己的 LLM 花费高两个数量级，不计进来成本就是假的
        trace.count("search_bocha", len(summary.queries))
    return cluster_issues(summary, client, usage_sink=getattr(trace, "usage_sink", None))


@dataclass(frozen=True)
class HistoryRow:
    platform: str
    sku_id: str
    title: str
    observed_days: int
    verdict: str
    lowest: str | None = None
    fake_discount: str | None = None


@dataclass(frozen=True)
class PriceHistory:
    keyword: str
    rows: tuple[HistoryRow, ...] = ()
    campaign: str = ""
    note: str = ""  # 空态 / 降级说明（有 rows 时为空）


def check_price_history(keyword: str, *, db_path: Path | None = None) -> PriceHistory:
    """自采快照的价格观测（观测天数 / 分位判断 / 先涨后降 / 大促日历）。

    **基于自观测数据，非平台全历史**——样本不足时不出时机结论。
    """
    from priceradar.insights import campaign_context
    from priceradar.jobs.snapshot import DEFAULT_DB, connect

    path = db_path or DEFAULT_DB
    if not Path(path).exists():
        return PriceHistory(
            keyword=keyword,
            note="暂无快照数据：请先在 Web 界面点「采集今日快照」或运行 python -m priceradar.jobs.snapshot",
        )

    connection = connect(path)
    try:
        found = connection.execute(
            """
            SELECT platform, sku_id, MAX(title) AS title, COUNT(DISTINCT captured_date) AS days
            FROM price_snapshots WHERE keyword LIKE ? GROUP BY platform, sku_id
            ORDER BY days DESC, sku_id LIMIT 5
            """,
            (f"%{keyword}%",),
        ).fetchall()
        if not found:
            return PriceHistory(
                keyword=keyword,
                note=f"快照库中没有与「{keyword}」相关的商品（关注清单见 data/watchlist.json）",
            )
        rows = []
        for platform, sku_id, title, _days in found:
            view = sku_history_view(connection, platform, sku_id)
            rows.append(
                HistoryRow(
                    platform=platform,
                    sku_id=sku_id,
                    title=title,
                    observed_days=view["observed_days"],
                    verdict=view["verdict"],
                    lowest=view.get("lowest"),
                    fake_discount=view.get("fake_discount"),
                )
            )
        return PriceHistory(keyword=keyword, rows=tuple(rows), campaign=campaign_context())
    finally:
        connection.close()


# ============================================================
# 摘要：回灌给 LLM 的工具结果（定长 + 可行动）
# ============================================================

def _clip(text: str, limit: int = SUMMARY_LIMIT) -> str:
    if len(text) <= limit:
        return text
    suffix = "…（已截断）"
    return text[: limit - len(suffix)] + suffix


def _short(text: str, width: int = 34) -> str:
    return text if len(text) <= width else text[:width] + "…"


def summarize_compare(result: PipelineResult) -> str:
    """比价结果摘要。

    **必须「可行动」**：带上 goods_sign / sku_id 等关键 ID，否则模型下一轮
    调不动 `find_similar`（它需要 goods_sign），链条就断在这里。
    数值一律用 `fmt_amount` 原样给出，**不让 LLM 转述数字**。
    """
    lines = [
        f"关键词：{result.keyword} ｜ 检索：拼多多 {result.pdd_count} 条 / 京东 {result.jd_count} 条",
        f"源商品：[{result.source.platform.value}] {_short(result.source.title)}"
        f"（标价 ￥{fmt_amount(result.source.list_price)}）sku={result.source.sku_id}",
    ]
    if result.notes:
        lines.append("备注：" + "；".join(_short(note, 60) for note in result.notes[:3]))
    lines.append(f"同款组 {len(result.groups)} 组，到手价：")
    for offer, breakdown in result.breakdowns[:8]:
        sign = offer.source_extra.get("goods_sign", "")
        sign_part = f" ｜ goods_sign={sign}" if sign else ""
        lines.append(
            f"  - [{offer.platform.value}] ￥{fmt_amount(breakdown.final_price)}"
            f" ｜ sku={offer.sku_id}{sign_part} ｜ {_short(offer.title, 26)}"
        )
    if any(offer.source_extra.get("goods_sign") for offer, _ in result.breakdowns):
        lines.append("（想找更便宜的相似款：用上面任一 goods_sign 调 find_similar）")
    return _clip("\n".join(lines))


def summarize_similar(result: SimilarResult) -> str:
    lines = [f"锚点商品相似推荐：{len(result.breakdowns)} 条（按平台推荐返回，混合有无关商品）"]
    for offer, breakdown in result.breakdowns[:8]:
        sign = offer.source_extra.get("goods_sign", "")
        sign_part = f" ｜ goods_sign={sign}" if sign else ""
        lines.append(
            f"  - [{offer.platform.value}] ￥{fmt_amount(breakdown.final_price)}"
            f" ｜ sku={offer.sku_id}{sign_part} ｜ {_short(offer.title, 26)}"
        )
    for note in result.notes:
        lines.append(f"备注：{_short(note, 60)}")
    return _clip("\n".join(lines))


def summarize_risk(result: RiskSummary) -> str:
    lines = [f"风险扫描：{result.product}（{len(result.queries)} 组关键词 · 命中 {len(result.items)} 条讨论）"]
    if result.issues:
        for issue in result.issues[:8]:
            lines.append(f"  - {issue['topic']}：{issue['mention_count']} 条提及")
    else:
        lines.append(f"  △ 未做问题归类（{result.note}）")
        for item in result.items[:5]:
            lines.append(f"    · [{item.site_name}] {_short(item.title, 40)}")
    lines.append("  检索结果不代表全部用户反馈")
    return _clip("\n".join(lines))


def summarize_history(result: PriceHistory) -> str:
    if not result.rows:
        return _clip(f"价格观测：{result.keyword} —— {result.note}")
    lines = [f"价格观测：与「{result.keyword}」相关的商品（基于自采快照，非平台全历史）"]
    if result.campaign:
        lines.append(f"  大促日历（参考往年规律）：{result.campaign}")
    for row in result.rows:
        lines.append(f"  - [{row.platform}] {_short(row.title, 30)}：已观测 {row.observed_days} 天 ｜ {row.verdict}")
        if row.lowest:
            lines.append(f"    观测最低：￥{row.lowest}")
        if row.fake_discount:
            lines.append(f"    △ {row.fake_discount}")
    lines.append("  ※ 样本不足时不出时机结论")
    return _clip("\n".join(lines))


# ============================================================
# 工具注册表（Agent 循环的 schema 来源）
# ============================================================

class ComparePricesArgs(BaseModel):
    """比价工具的参数。

    ⚠️ **刻意不含 `use_risk`**：风险扫描另有 `scan_risk` 工具。若两者都能触发，
    实测中模型会把 `use_risk=true` 与 `scan_risk` **同时**调一遍，白跑一次检索
    （约 ¥0.2 与 4 次搜索调用）。重叠的能力不能同时暴露给模型。
    `compare_prices()` 函数本身仍保留 `use_risk`，供 Web API 的 `risk` 开关使用。
    """

    query: str = Field(description="用户的查询，原文照传：关键词（如「QCY N20 降噪耳机」）或自然语言整句；也可以是拼多多商品链接")
    jd_channel: int = Field(default=0, description="京东京粉频道：0=自动品类路由（默认），-1=关闭京东侧，>0=定点拉该频道号")


class FindSimilarArgs(BaseModel):
    goods_sign: str = Field(description="锚点商品的 goods_sign（从 compare_prices 的结果摘要里取）")
    limit: int = Field(default=20, description="返回条数上限")


class ScanRiskArgs(BaseModel):
    product: str = Field(description="商品名或型号，如「QCY N20」")


class CheckPriceHistoryArgs(BaseModel):
    keyword: str = Field(description="要查观测历史的关键词（需在 data/watchlist.json 的关注范围内才有数据）")


class ActionArgumentError(ValueError):
    """工具入参不合法（多传、少传、类型错）。"""


@dataclass(frozen=True)
class Action:
    name: str
    description: str
    params: type[BaseModel]
    func: Callable[..., Any]
    summarize: Callable[[Any], str]
    retry: str = "none"   # "none" | "once"
    cost_hint: str = ""
    # 不在 LLM schema 里、但允许透传的内部参数（如 db_path）。
    # **不在这个白名单里的参数一律拒绝**——见 invoke 的说明。
    internal_params: frozenset[str] = frozenset()

    def invoke(self, arguments: dict, *, context: dict | None = None) -> Any:
        """校验入参并执行。

        ⚠️ 必须**白名单过滤**，不能让调用方传进来的键直接落到函数上。
        起因：`use_risk` 被刻意从 `ComparePricesArgs` 里移除（它与独立的 `scan_risk`
        能力重叠，实测模型会把两者同时调一遍、白跑一次全网检索）。但如果这里只做
        「校验模型字段、其余原样透传」，模型只要自己编一个 `use_risk=true` 就能绕过
        ——同样的道理，`db_path` 之类的内部参数也会被一个幻觉出来的键打开任意文件。
        """
        allowed = set(self.params.model_fields) | set(self.internal_params)
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ActionArgumentError(
                f"不接受的参数：{'、'.join(unknown)}（可用：{'、'.join(sorted(allowed))}）"
            )
        visible = {key: arguments[key] for key in self.params.model_fields if key in arguments}
        validated = self.params(**visible)
        # context 是**运行期注入**（如 trace），既不受白名单约束、也**不进 memo 键**
        # ——否则同一个调用的键会因为 trace 对象不同而永远不命中
        return self.func(**{**arguments, **validated.model_dump(), **(context or {})})


ACTIONS: dict[str, Action] = {
    action.name: action
    for action in (
        Action(
            name="compare_prices",
            description=(
                "跨平台比价：输入关键词或自然语言，返回同款组与各平台到手价（公开优惠口径测算）。"
                "耗时约 3-10 秒。这是最主要的工具，用户问「哪里买划算/多少钱」时先用它。"
                "query 必须原文照传（含拼多多链接时不要改写 URL）。"
            ),
            params=ComparePricesArgs,
            func=compare_prices,
            summarize=summarize_compare,
            retry="none",
            cost_hint="约 3-10 秒",
        ),
        Action(
            name="find_similar",
            description=(
                "给定一个商品的 goods_sign，找平台推荐的相似商品并按到手价升序返回。"
                "用户说「想找一样的但更便宜的」时用它。goods_sign 只能从 compare_prices 的"
                "结果摘要里取，不要自己编。耗时约 2-5 秒。"
            ),
            params=FindSimilarArgs,
            func=find_similar,
            summarize=summarize_similar,
            retry="none",
            cost_hint="约 2-5 秒",
        ),
        Action(
            name="scan_risk",
            description=(
                "对商品做全网风险扫描（避雷/翻车/踩坑/问题四组检索词 → 跨词聚类），"
                "返回风险主题与提及数。用户问「有没有什么坑/口碑怎么样」时用它。"
                "成本约 ¥0.2/次、耗时 5-10 秒，不要对同一个商品重复调用。"
            ),
            params=ScanRiskArgs,
            func=scan_risk,
            summarize=summarize_risk,
            retry="once",
            cost_hint="约 ¥0.2/次、5-10 秒",
        ),
        Action(
            name="check_price_history",
            description=(
                "查询自采价格快照的观测历史与时机判断（观测天数/分位/先涨后降/大促日历）。"
                "用户问「现在是不是好价/该不该等」时用它。只读本地快照库，毫秒级、零 API 花费；"
                "样本不足时它会如实说明。"
            ),
            params=CheckPriceHistoryArgs,
            func=check_price_history,
            summarize=summarize_history,
            retry="none",
            internal_params=frozenset({"db_path"}),   # 测试与命令行可注入路径；模型看不到
            cost_hint="毫秒级、零 API 花费",
        ),
    )
}


def accepted_params(name: str, registry: dict[str, "Action"] | None = None) -> frozenset[str]:
    """该工具接受的参数名（LLM 可见字段 + 白名单内部参数）。未知工具返回空集。

    调用方用它来**过滤要补进工具调用的默认值**——不筛的话，`jd_channel` 这类
    只有比价工具认的参数会被塞给 `scan_risk`，然后被白名单拒绝，整个工具就废了。
    """
    action = (registry or ACTIONS).get(name)
    if action is None:
        return frozenset()
    return frozenset(action.params.model_fields) | action.internal_params


def _parameters_schema(model: type[BaseModel]) -> dict:
    """pydantic 模型 → OpenAI 兼容的 parameters JSON Schema。"""
    schema = model.model_json_schema()
    payload = {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }
    if "$defs" in schema:  # 嵌套模型时才有
        payload["$defs"] = schema["$defs"]
    return payload


def openai_tools(actions: dict[str, Action] | None = None) -> list[dict]:
    """导出 OpenAI 兼容的 tools 数组（DeepSeek 用同一套协议）。

    顺序固定（按 `ACTIONS` 的插入序）——**稳定的前缀有利于 prompt 缓存命中**：
    缓存命中与未命中的单价差 30-50 倍，而工具定义是每轮都重复的长前缀。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": action.name,
                "description": action.description,
                "parameters": _parameters_schema(action.params),
            },
        }
        for action in (actions or ACTIONS).values()
    ]


# ============================================================
# 执行器：结构化错误 + 按工具的重试策略
# ============================================================

@dataclass
class ActionResult:
    ok: bool
    action: str
    value: Any = None
    summary: str = ""
    error: str = ""
    retryable: bool = False
    hint: str = ""


def _is_transient_deep(error: BaseException) -> bool:
    """沿 `__cause__` / `__context__` 链判断是否瞬时故障。

    必须走链：`BochaError` / `LlmError` 都是**包装异常**，它们自己在
    `is_transient` 里一律返回 False，真正的网络故障藏在 `__cause__` 里。
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if is_transient(current):
            return True
        current = current.__cause__ or current.__context__
    return False


_HINT_TRANSIENT = "网络瞬时故障。可以稍后原样重试；若连续失败，改用其它工具或先如实告知用户暂时拿不到数据。"
_HINT_BUSINESS = "这是参数/权限/凭证类错误，原样重试没有用。请调整参数，或改用其它工具。"
_HINT_NO_RESULT = "这不是故障：该关键词/类目在数据源侧没有可比商品。请换关键词或品类，或请用户直接粘贴商品链接。"

# 平台对「拿普通商品页 URL 来反查」的报错是技术性的，用户看不懂。
# 实测：接口返回的商品 URL、以及从网页复制来的 URL **都不是**「原始链接」，
# 只有 App 里「分享 → 复制链接」拿到的那种（含 uin 参数）才行。
_ORIGINAL_LINK_MARKER = "原始链接"
_ORIGINAL_LINK_MESSAGE = (
    "这个链接不是拼多多 App 分享出来的「原始链接」（缺少 uin 等参数），平台会拒绝反查。"
    "请在拼多多 App 里用「分享 → 复制链接」重新获取，并**原样粘贴、不要删改参数**"
    "（尤其是 uin——它是平台校验原始链接的凭据）。"
)


def explain_pdd_error(error: BaseException) -> str:
    """把拼多多的技术性报错翻成用户能照着做的话；其余原样返回。"""
    text = str(error)
    if _ORIGINAL_LINK_MARKER in text:
        return _ORIGINAL_LINK_MESSAGE
    return text


def _hint_for(error: BaseException, retryable: bool) -> str:
    if isinstance(error, ActionArgumentError):
        return "参数不对：请严格按工具的参数 schema 重新发起，不要自造参数名。"
    if _ORIGINAL_LINK_MARKER in str(error):
        return _ORIGINAL_LINK_MESSAGE
    if isinstance(error, ValueError):   # 类目空洞 / 无结果
        return _HINT_NO_RESULT
    return _HINT_TRANSIENT if retryable else _HINT_BUSINESS


def run_action(
    name: str,
    arguments: dict,
    *,
    actions: dict[str, Action] | None = None,
    context: dict | None = None,
) -> ActionResult:
    """执行一个工具，**永不抛异常**——失败一律变成结构化结果喂回模型。

    为什么不让它抛：工具失败要能被模型看到并自行决策（换关键词 / 换工具 / 放弃），
    而不是打断整个循环。同时 `retryable` 如实标注「同一次调用稍后是否可能可行」。
    """
    registry = actions or ACTIONS
    action = registry.get(name)
    if action is None:
        return ActionResult(
            ok=False,
            action=name,
            error=f"未知工具：{name}（可用：{'、'.join(registry)}）",
            retryable=False,
            hint="请改用可用工具列表中的工具。",
        )

    attempts = 2 if action.retry == "once" else 1
    last: BaseException | None = None
    for _ in range(attempts):
        try:
            value = action.invoke(arguments, context=context)
        except ValidationError as exc:
            return ActionResult(
                ok=False,
                action=name,
                error=f"参数不合法：{exc.error_count()} 项错误 —— " + "；".join(
                    f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors()[:4]
                ),
                retryable=False,
                hint="请按工具的参数 schema 修正后重试。",
            )
        except Exception as exc:  # 工具失败不打断循环
            last = exc
            # 只有显式声明 retry=once 且确属瞬时故障才补试一次；
            # pdd 系声明 none——它内部已重试 3 次，再试会烧配额并触发账号限流。
            if action.retry == "once" and _is_transient_deep(exc):
                continue
            break
        else:
            return ActionResult(ok=True, action=name, value=value, summary=action.summarize(value))

    assert last is not None
    retryable = _is_transient_deep(last)
    return ActionResult(
        ok=False,
        action=name,
        error=f"{type(last).__name__}: {last}",
        retryable=retryable,
        hint=_hint_for(last, retryable),
    )


__all__ = [
    "ACTIONS",
    "ActionArgumentError",
    "SUMMARY_LIMIT",
    "Action",
    "ActionResult",
    "HistoryRow",
    "PriceHistory",
    "SimilarResult",
    "accepted_params",
    "apply_user_coupons",
    "check_price_history",
    "explain_pdd_error",
    "compare_prices",
    "find_similar",
    "merge_user_coupons",
    "openai_tools",
    "run_action",
    "scan_risk",
    "summarize_compare",
    "summarize_history",
    "summarize_risk",
    "summarize_similar",
]

# PddApiError 在此处仅为导出（供调用方 except）；避免 lint 报未使用
_ = PddApiError
