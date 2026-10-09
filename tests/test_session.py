"""会话存储与上下文组装测试（全离线，tmp_path 隔离 sqlite）。"""

import json
from decimal import Decimal

import pytest

from priceradar.agent.context import SessionState, build_messages, update_state
from priceradar.agent.loop import AgentRun, AgentStep, ToolExecution
from priceradar.agent.pipeline import ConfirmedItem, OfferGroup, PipelineResult
from priceradar.agent.session_store import (
    append_message,
    connect_session_db,
    delete_session,
    list_sessions,
    load_messages,
    load_state,
    load_turns,
    new_session,
    next_turn_id,
    prune,
    save_state,
    session_exists,
)
from priceradar.domain.models import Offer, Platform
from priceradar.engine.pricing import compute_price

D = Decimal


@pytest.fixture()
def connection(tmp_path):
    conn = connect_session_db(tmp_path / "sessions.db")
    yield conn
    conn.close()


def make_result(keyword="QCY N20", sign="SIGN_A") -> PipelineResult:
    offer = Offer(platform=Platform.PDD, sku_id="a", title="QCY N20 降噪耳机", list_price=D("99"),
                  source_extra={"goods_sign": sign})
    return PipelineResult(
        keyword=keyword, pdd_count=20, jd_count=0, source=offer,
        confirmed=[ConfirmedItem(offer=offer, reason="源商品")],
        groups=[OfferGroup(representative=offer, members=[offer], final_price=D("99"), reason="源")],
        similar_count=0, excluded_summary={}, breakdowns=[(offer, compute_price(offer))],
    )


def seed_turn(connection, session_id, turn_id, question="问点什么"):
    """写入一轮完整对话：user → assistant(带 tool_calls) → tool → assistant(结语)。"""
    append_message(connection, session_id, role="user", turn_id=turn_id, content=question)
    append_message(
        connection, session_id, role="assistant", turn_id=turn_id, content=None,
        tool_calls=[{"id": f"c{turn_id}", "type": "function",
                     "function": {"name": "compare_prices", "arguments": "{}"}}],
    )
    append_message(connection, session_id, role="tool", turn_id=turn_id, content="摘要", tool_call_id=f"c{turn_id}")
    append_message(connection, session_id, role="assistant", turn_id=turn_id, content="结论")


# ============================================================
# 会话存储
# ============================================================

def test_new_session_is_idempotent(connection):
    sid = new_session(connection)
    assert session_exists(connection, sid)
    assert new_session(connection, sid) == sid          # 重入不报错
    assert not session_exists(connection, "不存在")


def test_append_message_assigns_monotonic_seq(connection):
    sid = new_session(connection)
    seqs = [append_message(connection, sid, role="user", turn_id=1, content=f"第{i}条") for i in range(4)]
    assert seqs == [1, 2, 3, 4]


def test_next_turn_id_increments(connection):
    sid = new_session(connection)
    assert next_turn_id(connection, sid) == 1
    seed_turn(connection, sid, 1)
    assert next_turn_id(connection, sid) == 2


def test_load_turns_groups_by_turn(connection):
    sid = new_session(connection)
    seed_turn(connection, sid, 1, "第一问")
    seed_turn(connection, sid, 2, "第二问")
    turns = load_turns(connection, sid, keep_turns=None)
    assert len(turns) == 2
    assert [message["role"] for message in turns[0]] == ["user", "assistant", "tool", "assistant"]
    assert turns[0][0]["content"] == "第一问"
    assert turns[1][0]["content"] == "第二问"


def test_load_turns_keeps_only_recent_whole_turns(connection):
    """裁历史只能**整轮**丢——半轮会让 tool_calls 与 tool 消息配不上对，直接 400。"""
    sid = new_session(connection)
    for turn in range(1, 5):
        seed_turn(connection, sid, turn, f"第{turn}问")
    turns = load_turns(connection, sid, keep_turns=2)
    assert len(turns) == 2
    assert turns[0][0]["content"] == "第3问"
    for turn in turns:
        roles = [message["role"] for message in turn]
        assert roles == ["user", "assistant", "tool", "assistant"]   # 每轮都是完整的


def test_load_turns_empty_session(connection):
    sid = new_session(connection)
    assert load_turns(connection, sid, keep_turns=2) == []
    assert load_messages(connection, sid) == []


def test_tool_calls_and_tool_call_id_roundtrip(connection):
    sid = new_session(connection)
    seed_turn(connection, sid, 1)
    messages = load_messages(connection, sid)
    assert messages[1]["tool_calls"][0]["id"] == "c1"
    assert messages[2]["tool_call_id"] == "c1"


def test_state_roundtrip(connection):
    sid = new_session(connection)
    assert load_state(connection, sid) == {}
    save_state(connection, sid, {"keyword": "QCY N20", "goods_sign": "S1"})
    assert load_state(connection, sid) == {"keyword": "QCY N20", "goods_sign": "S1"}


def test_state_survives_corruption(connection):
    """库里被写进坏 JSON 时不能把整个会话打挂——如实返回空槽位。"""
    sid = new_session(connection)
    connection.execute("UPDATE sessions SET state_json = ? WHERE id = ?", ("{坏掉的", sid))
    connection.commit()
    assert load_state(connection, sid) == {}


def test_prune_keeps_recent_turns(connection):
    sid = new_session(connection)
    for turn in range(1, 8):
        seed_turn(connection, sid, turn)
    removed = prune(connection, sid, keep_turns=3)
    assert removed == 4 * 4                       # 4 整轮 × 4 条
    assert len(load_turns(connection, sid, keep_turns=None)) == 3


def test_prune_noop_when_within_limit(connection):
    sid = new_session(connection)
    seed_turn(connection, sid, 1)
    assert prune(connection, sid, keep_turns=5) == 0


def test_delete_session_removes_messages_too(connection):
    sid = new_session(connection)
    seed_turn(connection, sid, 1)
    delete_session(connection, sid)
    assert not session_exists(connection, sid)
    assert load_messages(connection, sid) == []


def test_list_sessions_summarises(connection):
    sid = new_session(connection)
    seed_turn(connection, sid, 1, "QCY N20 哪里买划算")
    rows = list_sessions(connection)
    assert rows[0]["id"] == sid
    assert rows[0]["turns"] == 1
    assert rows[0]["first_query"] == "QCY N20 哪里买划算"


def test_sessions_are_isolated(connection):
    first, second = new_session(connection), new_session(connection)
    seed_turn(connection, first, 1, "只属于第一个")
    assert load_messages(connection, second) == []


# ============================================================
# 结构化槽位
# ============================================================

def test_empty_state_produces_no_prompt():
    assert SessionState().to_prompt() == ""
    assert SessionState().is_empty()


def test_state_prompt_mentions_the_object_and_the_evidence_rule():
    text = SessionState(keyword="QCY N20", product="QCY N20", goods_sign="SIGN_A").to_prompt()
    assert "QCY N20" in text and "SIGN_A" in text
    assert "不要自己编" in text          # 明确禁止模型自己造 goods_sign


def test_state_dict_roundtrip_and_partial_tolerance():
    state = SessionState(keyword="a", goods_sign="s")
    assert SessionState.from_dict(state.to_dict()) == state
    assert SessionState.from_dict({"keyword": "only"}) == SessionState(keyword="only")
    assert SessionState.from_dict(None) == SessionState()
    assert SessionState.from_dict({"keyword": None}) == SessionState()


# ============================================================
# 从运行结果更新槽位
# ============================================================

def make_run(*, ok=True, value=None, steps=1) -> AgentRun:
    calls = [
        ToolExecution(id=f"c{i}", name="compare_prices", arguments={}, ok=ok, summary="s", value=value)
        for i in range(steps)
    ]
    return AgentRun(query="q", stopped_by="finish", final_text="x", steps=[AgentStep(index=0, content=None, tool_calls=calls)])


def test_update_state_picks_up_keyword_and_goods_sign():
    state = update_state(SessionState(), make_run(value=make_result()))
    assert state.keyword == "QCY N20"
    assert state.goods_sign == "SIGN_A"
    assert state.platform == "pdd"


def test_update_state_ignores_failed_tool_calls():
    """失败的工具调用没有值可取——不该污染槽位。"""
    state = update_state(SessionState(), make_run(ok=False, value=None))
    assert state.is_empty()


def test_update_state_falls_back_to_any_offer_with_a_sign():
    result = make_result(sign="")
    result.source.source_extra.clear()
    result.breakdowns[0][0].source_extra["goods_sign"] = "FROM_BREAKDOWN"
    state = update_state(SessionState(), make_run(value=result))
    assert state.goods_sign == "FROM_BREAKDOWN"


def test_update_state_preserves_previous_values():
    """本轮没提供新信息时，旧槽位必须留着——它是历史被裁掉后的唯一锚点。"""
    state = update_state(SessionState(keyword="旧的", goods_sign="OLD"), make_run(ok=False))
    assert state.keyword == "旧的" and state.goods_sign == "OLD"


def test_update_state_does_not_mutate_the_input():
    original = SessionState(keyword="旧的")
    update_state(original, make_run(value=make_result()))
    assert original.keyword == "旧的"


# ============================================================
# 上下文组装
# ============================================================

def turns_of(*questions) -> list[list[dict]]:
    return [
        [{"role": "user", "content": q}, {"role": "assistant", "content": f"答复 {q}"}]
        for q in questions
    ]


def test_build_messages_order_is_system_state_history_query():
    messages, note = build_messages(
        "SYSTEM", SessionState(keyword="QCY N20"), turns_of("第一问"), "第二问", char_budget=10_000
    )
    assert messages[0] == {"role": "system", "content": "SYSTEM"}
    assert messages[1]["role"] == "system" and "QCY N20" in messages[1]["content"]
    assert messages[2]["content"] == "第一问"
    assert messages[-1] == {"role": "user", "content": "第二问"}
    assert note == ""


def test_system_prompt_stays_first_and_constant():
    """固定前缀必须排在最前——它是 prompt 缓存能命中的前提。"""
    state = SessionState(keyword="X" * 50)
    messages, _ = build_messages("SYSTEM", state, turns_of("q"), "n", char_budget=10_000)
    assert messages[0]["content"] == "SYSTEM"


def test_no_state_message_when_state_is_empty():
    messages, _ = build_messages("SYSTEM", SessionState(), [], "问", char_budget=10_000)
    assert [message["role"] for message in messages] == ["system", "user"]


def test_build_messages_drops_oldest_whole_turns_when_over_budget():
    turns = turns_of(*[f"第{i}问" + "长" * 200 for i in range(5)])
    messages, note = build_messages("SYSTEM", SessionState(), turns, "最新问", char_budget=900)
    contents = [message.get("content") for message in messages]
    assert contents[-1] == "最新问"
    assert "第0问" + "长" * 200 not in contents        # 最老的被丢了
    assert "已省略" in note                            # 如实告知，不静默丢


def test_dropped_turns_never_leave_half_a_turn():
    """裁剪必须整轮——半轮会让 tool_calls 配不上 tool 消息。"""
    turn = [
        {"role": "user", "content": "问题" + "长" * 100},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "结果"},
    ]
    messages, note = build_messages("S", SessionState(), [turn] * 4, "新问", char_budget=700)
    kept = [m for m in messages if m.get("tool_call_id")]
    assistants_with_calls = [m for m in messages if m.get("tool_calls")]
    assert len(kept) == len(assistants_with_calls), "留下的 tool 消息必须都有对应的 assistant"
    if note:
        assert "已省略" in note


def test_history_never_dropped_below_minimum_rounds():
    messages, note = build_messages(
        "S", SessionState(), turns_of("唯一一问" + "长" * 500), "新问", char_budget=50
    )
    assert messages[-1]["content"] == "新问"
    # 预算小到极致时，宁可把历史全丢掉也不能把消息列表搞坏
    assert all(message["role"] in ("system", "user", "assistant", "tool") for message in messages)


def test_build_messages_with_no_history(tmp_path):
    messages, note = build_messages("S", SessionState(), [], "首问", char_budget=10_000)
    assert note == ""
    assert len(messages) == 2
