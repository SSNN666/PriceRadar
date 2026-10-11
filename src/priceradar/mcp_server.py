"""PriceRadar MCP Server：把跨平台比价能力开放给 Claude / Cursor 等 MCP 客户端。

工具（单一职责切分）：
- compare_prices      ：关键词/自然语言 → 同款组 + 到手价对比卡（全管线）
- scan_risk           ：商品名 → 全网风险扫描（博查检索 + LLM 聚类 + 覆盖声明）
- check_price_history ：关键词 → 自采快照的价格观测（观测天数/分位/先涨后降/大促日历）

**本模块只做协议适配与文本渲染**——编排在 `agent.actions`，与 Web `/api/search`、
Agent 循环共用同一份实现。此前三处各写一遍，语义已经分叉（MCP 侧漏了用户补录券）。
工具的**签名保持平铺**（`query` / `product` / `keyword`）：它由 mcp SDK 从函数签名
推导 inputSchema，是已经在 Claude/Cursor 里配置好的对外契约，改成收对象即破坏兼容。

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

# ⚠️ 必须用别名导入：本模块的 MCP 工具函数恰好与 action **同名**
# （compare_prices / scan_risk / check_price_history），直接用原名会被后面的
# def 遮蔽——`_compare_text` 里调到的就成了 MCP 工具包装函数本身，静默拿到 str。
from priceradar.agent.actions import (
    check_price_history as action_check_price_history,
    compare_prices as action_compare_prices,
    scan_risk as action_scan_risk,
    summarize_history,
)
from priceradar.agent.pipeline import render
from priceradar.risk import COVERAGE_NOTE
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


def _compare_text(query: str, jd_channel: int = 0) -> str:
    """全管线：查询理解 → 检索 → 匹配 → 到手价 → 用户券并入 → 对比卡（纯文本输出）。

    编排在 `agent.actions.compare_prices`（与 Web `/api/search`、Agent 循环**同一份实现**）。
    此前这里自己拼了一遍，且漏掉了用户补录券与风险聚类——同一个「比价」在两处给出不同
    结果，是个既存不一致，现已消除。
    """
    try:
        result = action_compare_prices(query, jd_channel=jd_channel)
    except PddApiError as exc:
        return f"拼多多检索失败：{exc}"
    except ValueError as exc:  # 类目空洞 / 无检索结果：如实转达
        return str(exc)
    return render(result)


def _risk_text(product: str) -> str:
    """L2 风险扫描文本输出（有证据列主题，无证据如实说明，固定附覆盖声明）。"""
    try:
        summary = action_scan_risk(product)
    except BochaError as exc:
        return f"风险扫描失败：{exc}"

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
    """快照库价格观测文本输出（样本不足时诚实降级）。

    渲染直接复用 `actions.summarize_history`——它本来就是同一个格式，
    再手写一遍就是第四份重复了。
    """
    return summarize_history(action_check_price_history(keyword, db_path=db_path))


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
