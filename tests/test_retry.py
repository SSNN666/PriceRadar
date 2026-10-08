"""瞬时故障重试测试（含 2026-10-07 真实 DNS 故障的回归用例）。"""

import json
import urllib.error

import pytest

from priceradar.sources.pdd import PddApiError, PddClient
from priceradar.sources.jd import JdApiError, JdClient
from priceradar.sources.retry import call_with_retry, is_transient


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.com", code, "boom", {}, None)


# —— 分类：什么值得重试 ——


def test_dns_failure_is_transient():
    """2026-10-07 真实故障：DNS 解析失败，是典型的"再试一次可能就好"。"""
    assert is_transient(urllib.error.URLError("getaddrinfo failed")) is True


def test_timeout_and_connection_reset_are_transient():
    assert is_transient(TimeoutError("timed out")) is True
    assert is_transient(ConnectionResetError("reset by peer")) is True


def test_server_errors_and_throttling_are_transient():
    assert is_transient(http_error(500)) is True
    assert is_transient(http_error(502)) is True
    assert is_transient(http_error(429)) is True


def test_client_errors_are_not_transient():
    """4xx 是请求本身的问题（凭证/参数/权限），重试无用。"""
    assert is_transient(http_error(400)) is False
    assert is_transient(http_error(403)) is False
    assert is_transient(http_error(404)) is False


def test_business_errors_are_not_transient():
    """业务错误必须立刻抛出——重试既无效又浪费额度，还把真问题拖成慢失败。"""
    assert is_transient(PddApiError("备案未通过", code=50001, sub_code="60001")) is False
    assert is_transient(JdApiError("无权限", code=403)) is False
    assert is_transient(ValueError("参数非法")) is False


def test_truncated_response_is_transient():
    """响应被截断（网关抖动）值得重试，不是数据格式变了。"""
    try:
        json.loads("{ truncated")
    except json.JSONDecodeError as error:
        assert is_transient(error) is True
    else:  # pragma: no cover
        pytest.fail("应当抛出 JSONDecodeError")


# —— 重试行为 ——


def test_success_on_first_attempt_does_not_sleep():
    delays: list[float] = []
    assert call_with_retry(lambda: "ok", sleeper=delays.append) == "ok"
    assert delays == []


def test_retries_transient_then_succeeds():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.URLError("getaddrinfo failed")
        return "ok"

    delays: list[float] = []
    assert call_with_retry(flaky, sleeper=delays.append) == "ok"
    assert attempts["n"] == 3
    assert delays == [0.5, 1.0]  # 指数退避


def test_exhausting_attempts_reraises_last_error():
    def always_fail():
        raise urllib.error.URLError("getaddrinfo failed")

    with pytest.raises(urllib.error.URLError):
        call_with_retry(always_fail, attempts=3, sleeper=lambda _s: None)


def test_non_transient_error_is_not_retried():
    attempts = {"n": 0}

    def business_error():
        attempts["n"] += 1
        raise PddApiError("备案未通过", code=50001, sub_code="60001")

    delays: list[float] = []
    with pytest.raises(PddApiError):
        call_with_retry(business_error, sleeper=delays.append)
    assert attempts["n"] == 1  # 一次就放弃
    assert delays == []        # 且不等待


# —— 两个数据源的接入 ——


def test_pdd_client_survives_dns_blip(monkeypatch):
    """端到端回归：DNS 抖动两次后恢复，查询照常成功（而不是丢一天数据）。"""
    monkeypatch.setattr("priceradar.sources.retry.time.sleep", lambda _s: None)
    calls: list[dict] = []

    def flaky(payload):
        calls.append(payload)
        if len(calls) < 3:
            raise urllib.error.URLError("getaddrinfo failed")
        return {"goods_search_response": {"goods_list": []}}

    client = PddClient("id", "secret", transport=flaky)
    assert client.call("pdd.ddk.goods.search", keyword="x") == {"goods_search_response": {"goods_list": []}}
    assert len(calls) == 3


def test_pdd_client_does_not_retry_business_error(monkeypatch):
    """备案类业务错误不得被重试掩盖。"""
    monkeypatch.setattr("priceradar.sources.retry.time.sleep", lambda _s: None)
    calls: list[dict] = []

    def business_fail(payload):
        calls.append(payload)
        raise PddApiError("未备案", code=50001, sub_code="60001")

    client = PddClient("id", "secret", transport=business_fail)
    with pytest.raises(PddApiError):
        client.call("pdd.ddk.goods.search", keyword="x")
    assert len(calls) == 1


def test_jd_client_survives_dns_blip(monkeypatch):
    monkeypatch.setattr("priceradar.sources.retry.time.sleep", lambda _s: None)
    calls: list[dict] = []

    def flaky(params):
        calls.append(params)
        if len(calls) < 2:
            raise urllib.error.URLError("getaddrinfo failed")
        return {"jd_union_open_goods_jingfen_query_responce": {"result": {"code": 200, "data": []}}}

    client = JdClient("key", "secret", transport=flaky)
    assert client.call("jd.union.open.goods.jingfen.query", {})["code"] == 200
    assert len(calls) == 2
