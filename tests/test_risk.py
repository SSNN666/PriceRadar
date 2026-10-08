"""风险扫描（L2）测试：博查客户端解析 + 查询扩展 + 去重 + LLM 聚类与兜底。"""

import pytest

from priceradar.agent.llm import DeepSeekClient
from priceradar.risk import (
    RiskIssues,
    RiskSummary,
    build_risk_queries,
    cluster_issues,
    scan_risk,
)
from priceradar.sources.bocha import BochaClient, BochaError, SearchResult

from tests.test_llm import ScriptedTransport, fake_response


def canned_pages(*urls: str) -> dict:
    return {
        "code": 200,
        "data": {
            "webPages": {
                "value": [
                    {"name": f"标题{i}", "url": url, "siteName": "小红书", "snippet": f"摘要{i}"} for i, url in enumerate(urls)
                ]
            }
        },
    }


class FakeBocha:
    """按查询词返回预置结果的假客户端（鸭子类型，与 BochaClient 同接口）。"""

    def __init__(self, results: dict[str, list[SearchResult]]) -> None:
        self.results = results
        self.queries_called: list[str] = []

    def web_search(self, query: str, count: int = 8) -> list[SearchResult]:
        self.queries_called.append(query)
        return self.results.get(query, [])


# ---------- 博查客户端 ----------

def test_bocha_web_search_parses_pages():
    client = BochaClient("k", transport=lambda body: canned_pages("https://a.com/1", "https://b.com/2"))
    results = client.web_search("测试", count=5)
    assert len(results) == 2
    assert results[0].url == "https://a.com/1"
    assert results[0].site_name == "小红书"


def test_bocha_raises_on_error_without_pages():
    client = BochaClient("k", transport=lambda body: {"code": 403, "message": "no quota"})
    with pytest.raises(BochaError):
        client.web_search("测试")


# ---------- 查询扩展与检索 ----------

def test_build_risk_queries_covers_risk_words():
    queries = build_risk_queries("QCY N20 降噪耳机")
    assert len(queries) == 4
    assert all("QCY N20 降噪耳机" in q for q in queries)
    for word in ("避雷", "翻车", "踩坑", "问题"):
        assert any(word in q for q in queries)


def test_scan_risk_dedupes_urls_across_queries():
    shared = SearchResult(title="共享结果", url="https://x.com/same", site_name="知乎", snippet="s")
    fake = FakeBocha(
        {
            "QCY N20 降噪耳机 避雷": [shared, SearchResult("A", "https://a.com", "贴吧", "")],
            "QCY N20 降噪耳机 翻车": [shared],
            "QCY N20 降噪耳机 踩坑": [],
            "QCY N20 降噪耳机 问题": [SearchResult("B", "https://b.com", "黑猫投诉", "")],
        }
    )
    summary = scan_risk("QCY N20 降噪耳机", fake)
    assert len(fake.queries_called) == 4
    assert len(summary.items) == 3  # 跨查询去重
    assert summary.degraded is True


# ---------- LLM 聚类与兜底 ----------

def make_summary() -> RiskSummary:
    fake = FakeBocha(
        {
            "X 避雷": [SearchResult("续航缩水吐槽", "https://a.com/1", "小红书", "充满电只能用两小时")],
            "X 翻车": [SearchResult("续航翻车实录", "https://b.com/2", "知乎", "续航虚标严重")],
        }
    )
    return scan_risk("X", fake)


def test_cluster_issues_without_llm_stays_degraded():
    summary = cluster_issues(make_summary(), None)
    assert summary.degraded is True
    assert summary.issues == []


def test_cluster_issues_with_fake_llm():
    transport = ScriptedTransport(
        fake_response('{"issues": [{"topic": "续航缩水", "mention_count": 2, "urls": ["https://a.com/1"]}, {"topic": "客服差", "mention_count": 1, "urls": []}], "note": ""}')
    )
    client = DeepSeekClient("k", transport=transport)
    summary = cluster_issues(make_summary(), client)
    assert summary.degraded is False
    assert summary.issues[0]["topic"] == "续航缩水"
    assert summary.issues[0]["mention_count"] == 2


def test_cluster_issues_llm_failure_keeps_degraded():
    def boom(body: dict) -> dict:
        raise RuntimeError("unexpected")

    client = DeepSeekClient("k", transport=boom)
    with pytest.raises(RuntimeError):
        # 非 LlmError 的异常不在此层捕获（由管线兜底）——保持行为显式
        cluster_issues(make_summary(), client)
