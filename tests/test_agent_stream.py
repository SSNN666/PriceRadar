"""Agent SSE 流式接口测试（打桩 run_agent，全离线）。"""

import json
import time
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

import priceradar.agent.session_store as session_store
import priceradar.service.api as api_module
from priceradar.agent.loop import AgentRun, AgentStep, ToolExecution
from priceradar.agent.pipeline import ConfirmedItem, OfferGroup, PipelineResult
from priceradar.domain.models import Offer, Platform
from priceradar.engine.pricing import compute_price

D = Decimal


@pytest.fixture(autouse=True)
def isolated_sessions(tmp_path, monkeypatch):
    """会话库切到临时目录——否则测试会往真实的 data/sessions.db 里写垃圾。"""
    monkeypatch.setattr(session_store, "DEFAULT_DB", tmp_path / "sessions.db")
    return tmp_path / "sessions.db"


def make_result(keyword="QCY N20") -> PipelineResult:
    offer = Offer(platform=Platform.PDD, sku_id="a", title="QCY N20 降噪耳机", list_price=D("99"),
                  source_extra={"goods_sign": "SIGN_A"})
    return PipelineResult(
        keyword=keyword,
        pdd_count=20,
        jd_count=0,
        source=offer,
        confirmed=[ConfirmedItem(offer=offer, reason="源商品")],
        groups=[OfferGroup(representative=offer, members=[offer], final_price=D("99"), reason="源")],
        similar_count=0,
        excluded_summary={},
        breakdowns=[(offer, compute_price(offer))],
        summary="最低 ￥99",
    )


def make_run(*, stopped_by="finish", with_result=True) -> AgentRun:
    return AgentRun(
        query="QCY N20",
        stopped_by=stopped_by,
        final_text="最低 ￥99，建议直接买",
        steps=[
            AgentStep(
                index=0,
                content="我来查一下",
                tool_calls=[
                    # value 是工具的**真实返回值**——槽位只从这里取（确定性事实），
                    # 不从模型给的参数里猜
                    ToolExecution(id="c1", name="compare_prices", arguments={"query": "QCY N20"},
                                  ok=True, summary="比价摘要", value=make_result())
                ],
            )
        ],
        result=make_result() if with_result else None,
        usage=__import__("priceradar.agent.llm", fromlist=["Usage"]).Usage(
            prompt_tokens=1000, completion_tokens=50, cache_hit_tokens=800, cache_miss_tokens=200
        ),
    )


def patch_runner(monkeypatch, run=None, *, emits=(), boom=None, delay=0.0):
    def fake(query, on_event=None, jd_channel=0, use_risk=False, **kwargs):
        for event in emits:
            if on_event:
                on_event(event)
        if delay:
            time.sleep(delay)
        if boom is not None:
            raise boom
        return run

    monkeypatch.setattr(api_module, "run_agent", fake)


def frames(body: str) -> list[dict]:
    """把 SSE 响应体解析成事件列表。"""
    parsed = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):   # 心跳是注释行
            continue
        lines = block.splitlines()
        event_id = next((int(line[3:]) for line in lines if line.startswith("id:")), None)
        data = next((line[5:] for line in lines if line.startswith("data:")), None)
        if data is not None:
            parsed.append({"id": event_id, **json.loads(data)})
    return parsed


# ============================================================
# 事件序列
# ============================================================

def test_stream_emits_steps_tools_and_final(monkeypatch):
    patch_runner(
        monkeypatch,
        make_run(),
        emits=[
            {"type": "step", "index": 0, "content": "我来查一下"},
            {"type": "tool", "index": 0, "name": "compare_prices", "arguments": {"query": "x"}, "ok": True, "summary": "比价摘要"},
        ],
    )
    resp = TestClient(api_module.app).get("/api/agent/stream", params={"q": "QCY N20"})
    assert resp.status_code == 200
    events = frames(resp.text)
    assert [event["type"] for event in events] == ["session", "step", "tool", "final"]
    assert events[0]["session_id"]                                  # 首帧带会话 ID，前端据此续接
    assert events[2]["name"] == "compare_prices"
    assert events[2]["summary"] == "比价摘要"


def test_final_event_carries_payload_and_agent_metadata(monkeypatch):
    patch_runner(monkeypatch, make_run())
    body = TestClient(api_module.app).get("/api/agent/stream", params={"q": "x"}).text
    final = frames(body)[-1]
    assert final["type"] == "final"
    assert final["stopped_by"] == "finish"
    assert final["degraded"] is False
    assert final["tools"] == ["compare_prices"]
    assert final["rounds"] == 1
    assert final["payload"]["keyword"] == "QCY N20"          # 复用既有 payload 形状
    assert final["payload"]["offers"][0]["final_price"]
    assert final["usage"]["cache_hit_rate"] == 0.8


def test_every_frame_has_an_id_for_reconnect_replay(monkeypatch):
    patch_runner(monkeypatch, make_run(), emits=[{"type": "step", "index": 0, "content": "x"}])
    events = frames(TestClient(api_module.app).get("/api/agent/stream", params={"q": "x"}).text)
    assert all(event["id"] is not None for event in events)
    assert [event["id"] for event in events] == list(range(1, len(events) + 1))


def test_exactly_one_final_event(monkeypatch):
    """有且只有一个 final——前端收到它就 close，多了会漏掉真正的 payload。

    真实端到端跑出来的 bug：loop 自己 emit 了一个 {"type":"final"}（只有 text），
    SSE 层又发了一个真正的 final（带 payload）。前端收到第一个就把连接关了，
    比价卡片永远出不来。当时的单测全绿——因为桩没模拟这个事件名。
    """
    patch_runner(monkeypatch, make_run(), emits=[
        {"type": "step", "index": 0, "content": "x"},
        {"type": "tool", "index": 0, "name": "compare_prices", "ok": True, "summary": "s"},
    ])
    events = frames(TestClient(api_module.app).get("/api/agent/stream", params={"q": "x"}).text)
    finals = [event for event in events if event["type"] == "final"]
    assert len(finals) == 1
    assert finals[0]["payload"] is not None      # 活下来的必须是带 payload 的那个


def test_runner_emitting_a_rogue_final_is_dropped(monkeypatch):
    """防御边界：上游若误发 final / error，本层直接丢弃，不让前端提前收工。"""
    patch_runner(monkeypatch, make_run(), emits=[
        {"type": "final", "stopped_by": "finish", "text": "我是冒牌货"},
        {"type": "error", "message": "我是冒牌错误"},
        {"type": "tool", "index": 0, "name": "compare_prices", "ok": True, "summary": "s"},
    ])
    events = frames(TestClient(api_module.app).get("/api/agent/stream", params={"q": "x"}).text)
    assert [event["type"] for event in events] == ["session", "tool", "final"]
    assert events[-1]["payload"] is not None


def test_frames_are_well_formed_sse(monkeypatch):
    patch_runner(monkeypatch, make_run())
    body = TestClient(api_module.app).get("/api/agent/stream", params={"q": "x"}).text
    assert "event: final" in body
    assert body.endswith("\n\n")           # 每帧以空行结束
    # 空行是 SSE 的**帧分隔符**，协议要求；其余每行必须是四种合法字段之一
    assert all(not line or line.startswith(("id:", "event:", "data:", ":")) for line in body.splitlines())


# ============================================================
# 响应头（部署到 nginx 时缺一不可）
# ============================================================

def test_stream_headers_are_sse_ready(monkeypatch):
    patch_runner(monkeypatch, make_run())
    resp = TestClient(api_module.app).get("/api/agent/stream", params={"q": "x"})
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    # 没有这一条，nginx 会把整个流缓冲成一坨再吐出来，流式效果全丢
    assert resp.headers["x-accel-buffering"] == "no"


# ============================================================
# 失败路径：首 chunk 之后 app 级异常处理器已失效，必须自己转成事件
# ============================================================

def test_runner_exception_becomes_error_event_not_500(monkeypatch):
    patch_runner(monkeypatch, boom=RuntimeError("拼多多挂了"))
    resp = TestClient(api_module.app).get("/api/agent/stream", params={"q": "x"})
    assert resp.status_code == 200          # 流已经开始了，不能再改状态码
    events = frames(resp.text)
    assert events[-1]["type"] == "error"
    assert "拼多多挂了" in events[-1]["message"]


def test_degraded_run_still_emits_final(monkeypatch):
    """降级（触顶/LLM 失败）也必须有终帧，否则前端永远等不到结束。"""
    patch_runner(monkeypatch, make_run(stopped_by="max_steps"))
    events = frames(TestClient(api_module.app).get("/api/agent/stream", params={"q": "x"}).text)
    assert events[-1]["type"] == "final"
    assert events[-1]["degraded"] is True
    assert events[-1]["stopped_by"] == "max_steps"


def test_run_without_result_emits_final_with_null_payload(monkeypatch):
    patch_runner(monkeypatch, make_run(with_result=False))
    final = frames(TestClient(api_module.app).get("/api/agent/stream", params={"q": "x"}).text)[-1]
    assert final["payload"] is None


# ============================================================
# 参数与直通路径
# ============================================================

def test_llm_false_bypasses_agent(monkeypatch):
    """关闭 LLM 时不该跑 Agent（那会白花一次调用），直接给确定性结果。"""
    calls = []
    patch_runner(monkeypatch, make_run())
    monkeypatch.setattr(api_module, "compare_prices", lambda *a, **kw: calls.append(kw) or make_result())

    events = frames(TestClient(api_module.app).get("/api/agent/stream", params={"q": "x", "llm": "false"}).text)
    assert calls, "应当直通 compare_prices"
    assert events[-1]["type"] == "final"
    assert events[-1]["stopped_by"] == "llm_disabled"
    assert events[-1]["payload"]["keyword"] == "QCY N20"


def test_params_are_forwarded_to_the_runner(monkeypatch):
    captured = {}

    def fake(query, on_event=None, jd_channel=0, use_risk=False, **kwargs):
        captured.update({"query": query, "jd_channel": jd_channel, "use_risk": use_risk})
        return make_run()

    monkeypatch.setattr(api_module, "run_agent", fake)
    TestClient(api_module.app).get("/api/agent/stream", params={"q": "耳机", "jd_channel": "24", "risk": "true"})
    assert captured == {"query": "耳机", "jd_channel": 24, "use_risk": True}


def test_missing_query_is_rejected():
    assert TestClient(api_module.app).get("/api/agent/stream").status_code == 422


# ============================================================
# 多轮会话
# ============================================================

def capture_runner(monkeypatch, run=None):
    """记录 runner 收到的 kwargs（尤其是 messages），返回该列表。"""
    seen: list[dict] = []

    def fake(query, on_event=None, **kwargs):
        seen.append({"query": query, **kwargs})
        return run if run is not None else make_run()

    monkeypatch.setattr(api_module, "run_agent", fake)
    return seen


def test_first_turn_creates_a_session_and_persists_the_turn(monkeypatch):
    capture_runner(monkeypatch, make_run())
    events = frames(TestClient(api_module.app).get("/api/agent/stream", params={"q": "QCY N20 哪里买"}).text)
    sid = events[0]["session_id"]
    assert sid and events[-1]["session_id"] == sid

    from priceradar.agent.session_store import connect_session_db, load_turns

    connection = connect_session_db()
    try:
        turns = load_turns(connection, sid, keep_turns=None)
    finally:
        connection.close()
    assert len(turns) == 1
    assert turns[0][0] == {"role": "user", "content": "QCY N20 哪里买"}


def test_second_turn_reuses_the_session_and_sends_history(monkeypatch):
    """追问必须带上历史——这正是「多轮」的意义所在。"""
    seen = capture_runner(monkeypatch, make_run())
    client = TestClient(api_module.app)
    sid = frames(client.get("/api/agent/stream", params={"q": "QCY N20 哪里买"}).text)[0]["session_id"]

    client.get("/api/agent/stream", params={"q": "那有没有踩坑的", "session_id": sid})

    assert len(seen) == 2
    first_messages = seen[0]["messages"]
    second_messages = seen[1]["messages"]
    assert len(second_messages) > len(first_messages), "第二轮必须比第一轮多带历史"
    contents = [message.get("content") for message in second_messages]
    assert "QCY N20 哪里买" in contents          # 上一轮的提问还在
    assert second_messages[-1] == {"role": "user", "content": "那有没有踩坑的"}


def test_a_new_session_does_not_see_the_previous_one(monkeypatch):
    """不传 session_id 就是新会话——历史不能串味。"""
    seen = capture_runner(monkeypatch, make_run())
    client = TestClient(api_module.app)
    client.get("/api/agent/stream", params={"q": "第一问"})
    client.get("/api/agent/stream", params={"q": "第二问"})       # 不带 session_id
    assert seen[1]["messages"][-1] == {"role": "user", "content": "第二问"}
    assert not any(message.get("content") == "第一问" for message in seen[1]["messages"])


def test_state_carries_the_current_object_across_turns(monkeypatch):
    """槽位是历史被裁掉后仍然活下来的锚点——追问「那有没有坑」时模型得知道在说哪个商品。"""
    capture_runner(monkeypatch, make_run())
    client = TestClient(api_module.app)
    sid = frames(client.get("/api/agent/stream", params={"q": "QCY N20 哪里买"}).text)[0]["session_id"]

    from priceradar.agent.session_store import connect_session_db, load_state

    connection = connect_session_db()
    try:
        state = load_state(connection, sid)
    finally:
        connection.close()
    assert state.get("keyword") == "QCY N20"
    assert state.get("goods_sign") == "SIGN_A"


def test_degraded_run_still_leaves_an_assistant_message(monkeypatch):
    """降级时也要落一条助手消息，否则下一轮会读到「只有用户发言」的半截历史。"""
    capture_runner(monkeypatch, make_run(stopped_by="llm_error"))
    client = TestClient(api_module.app)
    sid = frames(client.get("/api/agent/stream", params={"q": "第一问"}).text)[0]["session_id"]

    from priceradar.agent.session_store import connect_session_db, load_turns

    connection = connect_session_db()
    try:
        turns = load_turns(connection, sid, keep_turns=None)
    finally:
        connection.close()
    assert [message["role"] for message in turns[0]] == ["user", "assistant"]


def test_session_endpoints(monkeypatch):
    capture_runner(monkeypatch, make_run())
    client = TestClient(api_module.app)
    sid = frames(client.get("/api/agent/stream", params={"q": "QCY N20"}).text)[0]["session_id"]

    listing = client.get("/api/sessions").json()
    assert any(item["id"] == sid for item in listing["sessions"])

    detail = client.get(f"/api/sessions/{sid}").json()
    assert detail["session_id"] == sid
    assert detail["state"]["keyword"] == "QCY N20"

    assert client.delete(f"/api/sessions/{sid}").json()["deleted"] is True
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_unknown_session_id_is_adopted(monkeypatch):
    """伪造/过期的 session_id 不该报错——直接建一个同 ID 的新会话。"""
    capture_runner(monkeypatch, make_run())
    events = frames(
        TestClient(api_module.app).get("/api/agent/stream", params={"q": "x", "session_id": "deadbeef"}).text
    )
    assert events[0]["session_id"] == "deadbeef"


def test_persistence_failure_does_not_lose_the_result(monkeypatch):
    """会话落库失败不能吞掉已经拿到的结果，只提示。"""
    capture_runner(monkeypatch, make_run())
    import priceradar.service.api as module

    def boom(*args, **kwargs):
        raise RuntimeError("磁盘满了")

    monkeypatch.setattr(module, "_persist_turn", boom)
    events = frames(TestClient(api_module.app).get("/api/agent/stream", params={"q": "x"}).text)
    assert events[-1]["type"] == "final"                       # 结果照常给出
    assert events[-1]["payload"] is not None
    assert any("未能存入会话" in (event.get("message") or "") for event in events)


# ============================================================
# 心跳：防代理按空闲超时切断
# ============================================================

def test_heartbeat_pings_while_the_agent_is_still_working(monkeypatch):
    """Agent 跑得慢时要周期性发心跳，否则中间代理会按空闲超时把连接掐了。"""
    patch_runner(monkeypatch, make_run(), delay=0.25)
    body = "".join(api_module.execute_agent("x", heartbeat=0.05))
    assert ": ping" in body
    assert frames(body)[-1]["type"] == "final"
