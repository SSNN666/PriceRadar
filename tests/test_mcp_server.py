"""MCP Server 工具层测试（打桩管线/检索，全离线；不依赖 LLM 网络）。"""

from datetime import datetime
from decimal import Decimal

import priceradar.mcp_server as server
from priceradar.agent.pipeline import ConfirmedItem, OfferGroup, PipelineResult
from priceradar.domain.models import Offer, Platform
from priceradar.engine.pricing import compute_price
from priceradar.jobs.snapshot import connect, record_offers
from priceradar.risk import RiskItem, RiskSummary
from priceradar.sources.bocha import BochaError

D = Decimal


def fake_result() -> PipelineResult:
    offer = Offer(platform=Platform.PDD, sku_id="a", title="QCY N20 降噪耳机", list_price=D("99"))
    return PipelineResult(
        keyword="QCY N20 降噪耳机",
        pdd_count=20,
        jd_count=0,
        source=offer,
        confirmed=[ConfirmedItem(offer=offer, reason="检索结果（源商品）")],
        groups=[OfferGroup(representative=offer, members=[offer], final_price=D("99"), reason="源")],
        similar_count=0,
        excluded_summary={},
        breakdowns=[(offer, compute_price(offer))],
    )


def offline(monkeypatch) -> None:
    """禁用 LLM（走确定性兜底路径，避免测试触网）。

    打桩点已随「编排下沉」搬到 `agent.actions`——MCP 侧不再自己持有 LLM 客户端。
    """
    import priceradar.agent.actions as actions

    monkeypatch.setattr(actions, "_llm_client", lambda: None)


# ---------- compare_prices ----------
# 打桩点从 `server.run`（管线）上移到 `server.action_compare_prices`（意图级能力）：
# 编排现在与 Web / Agent 循环共用同一份实现，MCP 侧只负责渲染。

def test_compare_text_renders_pipeline(monkeypatch):
    offline(monkeypatch)
    monkeypatch.setattr(server, "action_compare_prices", lambda query, **kwargs: fake_result())
    text = server._compare_text("QCY N20 降噪耳机", 0)
    assert "同款组" in text
    assert "到手价明细" in text
    assert "以结算页为准" in text  # 口径声明随卡输出


def test_compare_text_handles_pdd_error(monkeypatch):
    offline(monkeypatch)

    def boom(query, **kwargs):
        raise server.PddApiError("50001 业务服务错误")

    monkeypatch.setattr(server, "action_compare_prices", boom)
    text = server._compare_text("x", 0)
    assert "拼多多检索失败" in text


def test_compare_text_handles_empty_pool(monkeypatch):
    offline(monkeypatch)

    def empty(query, **kwargs):
        raise ValueError("关键词 'x' 无检索结果")

    monkeypatch.setattr(server, "action_compare_prices", empty)
    assert server._compare_text("x", 0) == "关键词 'x' 无检索结果"


# ---------- scan_risk ----------

def test_risk_text_with_issues(monkeypatch):
    offline(monkeypatch)
    summary = RiskSummary(
        product="QCY N20",
        queries=["QCY N20 避雷"],
        items=[RiskItem(query="QCY N20 避雷", title="t", url="https://x.com/1", site_name="小红书", snippet="s")],
        issues=[{"topic": "续航缩水", "mention_count": 3, "urls": []}],
        note="由 LLM 聚类",
        degraded=False,
    )
    monkeypatch.setattr(server, "action_scan_risk", lambda product: summary)
    text = server._risk_text("QCY N20")
    assert "续航缩水" in text and "3 条提及" in text
    assert "不代表全部用户反馈" in text  # 覆盖声明固定输出


def test_risk_text_degraded_lists_items(monkeypatch):
    offline(monkeypatch)
    summary = RiskSummary(
        product="QCY N20",
        queries=["QCY N20 避雷"],
        items=[RiskItem(query="q", title="翻车实录", url="https://x.com/1", site_name="知乎", snippet="")],
    )
    monkeypatch.setattr(server, "action_scan_risk", lambda product: summary)
    text = server._risk_text("QCY N20")
    assert "未做问题归类" in text
    assert "翻车实录" in text


def test_compare_text_delegates_to_the_shared_action(monkeypatch):
    """MCP 必须走与 Web / Agent 循环**同一份编排**。

    此前 MCP 自己拼了一遍管线，漏掉了用户补录券与风险聚类——同一个「比价」
    在两处给出不同结果。这条断言的就是那个接缝，防止日后再分叉。
    """
    offline(monkeypatch)
    seen = {}

    def spy(query, **kwargs):
        seen.update({"query": query, **kwargs})
        return fake_result()

    monkeypatch.setattr(server, "action_compare_prices", spy)
    server._compare_text("QCY N20", 24)
    assert seen["query"] == "QCY N20"
    assert seen["jd_channel"] == 24


def test_risk_text_delegates_to_the_shared_action(monkeypatch):
    offline(monkeypatch)
    summary = RiskSummary(product="x", queries=["q"], items=[], note="ok", degraded=True)
    seen = []
    monkeypatch.setattr(server, "action_scan_risk", lambda product: seen.append(product) or summary)
    server._risk_text("QCY N20")
    assert seen == ["QCY N20"]


def test_risk_text_handles_bocha_error(monkeypatch):
    def boom(product):
        raise BochaError("HTTP 403: no quota")

    monkeypatch.setattr(server, "action_scan_risk", boom)
    assert "风险扫描失败" in server._risk_text("x")


# ---------- check_price_history ----------

def test_price_history_missing_db(tmp_path):
    text = server._price_history_text("QCY N20", db_path=tmp_path / "none.db")
    assert "暂无快照数据" in text


def test_price_history_with_data(tmp_path):
    db = tmp_path / "snap.db"
    connection = connect(db)
    offer = Offer(platform=Platform.PDD, sku_id="a", title="QCY N20降噪蓝牙耳机", list_price=D("99"))
    record_offers(connection, [offer], "QCY N20 降噪耳机", captured=datetime(2026, 10, 5, 9, 0, 0))
    connection.close()

    text = server._price_history_text("QCY N20", db_path=db)
    assert "已观测 1 天" in text
    assert "数据积累中" in text
    assert "样本不足时不出时机结论" in text


def test_price_history_no_keyword_match(tmp_path):
    db = tmp_path / "snap.db"
    connection = connect(db)
    offer = Offer(platform=Platform.PDD, sku_id="a", title="别的商品", list_price=D("99"))
    record_offers(connection, [offer], "别的关键词", captured=datetime(2026, 10, 5, 9, 0, 0))
    connection.close()
    assert "没有与" in server._price_history_text("QCY N20", db_path=db)


# ---------- 工具注册 ----------

def test_tools_registered():
    import asyncio

    tools = asyncio.run(server.mcp.list_tools())
    names = {tool.name for tool in tools}
    assert {"compare_prices", "scan_risk", "check_price_history"} <= names
