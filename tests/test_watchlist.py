"""关注清单接口测试（PRD §4.2 承诺的「帮我盯着降价」的落点）。

此前只能手改 `data/watchlist.json`，多轮 UI 上点「帮我盯着降价」是个死胡同。
"""

import pytest
from fastapi.testclient import TestClient

import priceradar.jobs.snapshot as snapshot
import priceradar.service.api as api_module


@pytest.fixture(autouse=True)
def isolated_watchlist(tmp_path, monkeypatch):
    """切到临时文件——绝不碰真实的 data/watchlist.json。

    刻意先写一份**空清单**：默认模板自带示例关键词，不写的话每个断言都要
    先把那几个词减掉，测试会变得难读。
    """
    path = tmp_path / "watchlist.json"
    path.write_text('{"keywords": [], "jd_elite_id": 0}', encoding="utf-8")
    monkeypatch.setattr(snapshot, "WATCHLIST_PATH", path)
    return path


# ---------- 存储层 ----------

def test_load_creates_template_when_missing(isolated_watchlist):
    """文件不存在时写入模板（新用户第一次跑采集的路径）。"""
    isolated_watchlist.unlink()
    watchlist = snapshot.load_watchlist()
    assert isolated_watchlist.exists()
    assert "keywords" in watchlist and watchlist["keywords"]


def test_add_then_remove(isolated_watchlist):
    keywords, added = snapshot.add_watch_keyword("QCY N20")
    assert added is True and keywords == ["QCY N20"]
    _, added_again = snapshot.add_watch_keyword("QCY N20")
    assert added_again is False          # 幂等
    keywords, removed = snapshot.remove_watch_keyword("QCY N20")
    assert removed is True and keywords == []


def test_add_strips_whitespace_and_dedupes():
    snapshot.add_watch_keyword("  保温杯  ")
    keywords, _ = snapshot.add_watch_keyword("保温杯")
    assert keywords == ["保温杯"]


def test_add_rejects_blank():
    with pytest.raises(ValueError):
        snapshot.add_watch_keyword("   ")


def test_removing_absent_keyword_is_not_an_error():
    keywords, removed = snapshot.remove_watch_keyword("从来没加过")
    assert removed is False and keywords == []


def test_existing_keywords_are_preserved():
    snapshot.add_watch_keyword("第一个")
    snapshot.add_watch_keyword("第二个")
    keywords, _ = snapshot.add_watch_keyword("第三个")
    assert keywords == ["第一个", "第二个", "第三个"]     # 保序


def test_jd_elite_id_survives_keyword_edits():
    """改动关注关键词不能把京东频道配置冲掉。"""
    watchlist = snapshot.load_watchlist()
    watchlist["jd_elite_id"] = 24
    snapshot.save_watchlist(watchlist)
    snapshot.add_watch_keyword("QCY N20")
    assert snapshot.load_watchlist()["jd_elite_id"] == 24


# ---------- 路由层 ----------

def test_watchlist_routes_roundtrip():
    client = TestClient(api_module.app)

    assert client.get("/api/watchlist").json()["keywords"] == []

    added = client.post("/api/watchlist", json={"keyword": "QCY N20"}).json()
    assert added["added"] is True
    assert added["keywords"] == ["QCY N20"]
    assert "下次采集快照" in added["note"]

    again = client.post("/api/watchlist", json={"keyword": "QCY N20"}).json()
    assert again["added"] is False
    assert "已在关注清单" in again["note"]

    assert client.get("/api/watchlist").json()["keywords"] == ["QCY N20"]

    removed = client.delete("/api/watchlist", params={"keyword": "QCY N20"}).json()
    assert removed["removed"] is True
    assert removed["keywords"] == []


def test_watchlist_route_rejects_blank_keyword():
    resp = TestClient(api_module.app).post("/api/watchlist", json={"keyword": ""})
    assert resp.status_code == 422          # pydantic min_length 拦住


def test_watchlist_route_requires_keyword_param():
    assert TestClient(api_module.app).delete("/api/watchlist").status_code == 422
