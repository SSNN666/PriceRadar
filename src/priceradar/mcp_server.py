"""PriceRadar MCP Server：把跨平台比价能力开放给 Claude / Cursor 等 MCP 客户端。

工具（单一职责切分）：
- compare_prices      ：关键词/自然语言 → 同款组 + 到手价对比卡（全管线）
- scan_risk           ：商品名 → 全网风险扫描（博查检索 + LLM 聚类 + 覆盖声明）
- check_price_history ：关键词 → 自采快照的价格观测（观测天数/分位/先涨后降/大促日历）

运行（stdio 传输，基于 mcp SDK 2.x 的 MCPServer）：
    python -m priceradar.mcp_server

Claude Desktop / Cursor 配置示例：
    {
      "mcpServers": {
        "priceradar": {
          "command": "E:\\\\projects\\\\which_one\\\\.venv\\\\Scripts\\\\python.exe",
          "args": ["-m", "priceradar.mcp_server"],
          "cwd": "E:\\\\projects\\\\which_one"
        }
      }
    }
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.mcpserver import MCPServer

from priceradar.agent.llm import DeepSeekClient, LlmError
from priceradar.agent.llm_nodes import compose_summary, extract_keyword
from priceradar.agent.pipeline import render, run
from priceradar.insights import campaign_context, sku_history_view
from priceradar.risk import COVERAGE_NOTE, cluster_issues, scan_with_env
from priceradar.sources.bocha import BochaError
from priceradar.sources.pdd import PddApiError

mcp = MCPServer(
    "priceradar",
    instructions=(
        "真价雷达：跨平台到手价决策。所有金额均为平台公开优惠口径的测算值"
        "（未含个性化券，以结算页为准）；数据来自拼多多/京东联盟官方 API 与合法搜索服务；"
        "证据不足时会明确说明，不会编造结论。"
    ),
)


def _llm_client() -> DeepSeekClient | None:
    try:
        return DeepSeekClient.from_env()
    except LlmError:
        return None


def _compare_text(query: str, jd_channel: int = 0) -> str:
    """全管线：查询理解 → 检索 → 匹配 → 到手价 → 对比卡（纯文本输出）。"""
    llm_client = _llm_client()
    keyword, _ = extract_keyword(query, llm_client)
    try:
        result = run(keyword, jd_enabled=jd_channel >= 0, jd_elite_id=jd_channel if jd_channel > 0 else None)
    except PddApiError as exc:
        return f"拼多多检索失败：{exc}"
    except ValueError as exc:
        return str(exc)
    result.summary, _ = compose_summary(result, llm_client)
    return render(result)


def _risk_text(product: str) -> str:
    """L2 风险扫描文本输出（有证据列主题，无证据如实说明，固定附覆盖声明）。"""
    try:
        summary = scan_with_env(product)
    except BochaError as exc:
        return f"风险扫描失败：{exc}"
    summary = cluster_issues(summary, _llm_client())

    lines = [f"风险扫描：{summary.product}（{len(summary.queries)} 组关键词 · 命中 {len(summary.items)} 条讨论）"]
    if summary.issues:
        for issue in summary.issues:
            lines.append(f"- {issue['topic']}：{issue['mention_count']} 条提及")
        lines.append(f"说明：{summary.note}")
    else:
        lines.append(f"未做问题归类：{summary.note}")
        for item in summary.items[:5]:
            lines.append(f"- [{item.site_name}] {item.title[:40]}（{item.url}）")
    lines.append(f"※ {COVERAGE_NOTE}")
    return "\n".join(lines)


def _price_history_text(keyword: str, db_path: Path | None = None) -> str:
    """快照库价格观测文本输出（样本不足时诚实降级）。"""
    from priceradar.jobs.snapshot import DEFAULT_DB, connect

    path = db_path or DEFAULT_DB
    if not Path(path).exists():
        return "暂无快照数据：请先在 Web 界面点「采集今日快照」或运行 python -m priceradar.jobs.snapshot"

    connection = connect(path)
    try:
        rows = connection.execute(
            """
            SELECT platform, sku_id, MAX(title) AS title, COUNT(DISTINCT captured_date) AS days
            FROM price_snapshots WHERE keyword LIKE ? GROUP BY platform, sku_id
            ORDER BY days DESC, sku_id LIMIT 5
            """,
            (f"%{keyword}%",),
        ).fetchall()
        if not rows:
            return f"快照库中没有与「{keyword}」相关的商品（关注清单见 data/watchlist.json）"

        lines = [f"价格观测：与「{keyword}」相关的商品（基于自采快照，非平台全历史）"]
        context = campaign_context()
        if context:
            lines.append(f"大促日历（参考往年规律）：{context}")
        for platform, sku_id, title, _days in rows:
            view = sku_history_view(connection, platform, sku_id)
            lines.append(f"- [{platform}] {title[:36]}：已观测 {view['observed_days']} 天 ｜ {view['verdict']}")
            if view.get("lowest"):
                lines.append(f"  观测最低：￥{view['lowest']}")
            if view.get("fake_discount"):
                lines.append(f"  △ {view['fake_discount']}")
        lines.append("※ 样本不足时不出时机结论（诚实降级）")
        return "\n".join(lines)
    finally:
        connection.close()


@mcp.tool()
def compare_prices(query: str, jd_channel: int = 0) -> str:
    """按关键词或一句话查询跨平台到手价：返回同款组、到手价对比卡、价格观测与降级说明。

    query 例："QCY N20 降噪耳机" 或 "帮我看看这个哪里买划算"。
    jd_channel 为京东京粉频道 ID（0=自动品类路由，默认；-1=关闭京东侧；也可指定具体频道如 24=数码）。
    """
    return _compare_text(query, jd_channel)


@mcp.tool()
def scan_risk(product: str) -> str:
    """对商品做全网风险扫描（"避雷/翻车/踩坑/问题" 四组检索词 → 跨词聚类）。

    product 例："QCY N20" 或 "索尼 WH-1000XM6"。仅在检索到证据时输出风险主题；
    证据不足会如实说明，并固定附覆盖声明。
    """
    return _risk_text(product)


@mcp.tool()
def check_price_history(keyword: str) -> str:
    """查询自采价格快照的观测历史与时机判断（含大促日历、先涨后降检测）。

    keyword 例："QCY N20"（需在 data/watchlist.json 的关注范围内才有数据）。
    """
    return _price_history_text(keyword)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
