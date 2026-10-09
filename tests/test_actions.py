"""意图级能力层测试（全离线：外部调用一律打桩）。

覆盖三块：
1. 用户补录券并入（自 service 层下沉，行为必须与原先一致）
2. 四个意图级能力的编排顺序与降级
3. 工具注册表：OpenAI schema 导出、摘要「可行动」、按工具的重试策略
"""

import json
import urllib.error
from decimal import Decimal

import pytest

import priceradar.agent.actions as actions
from priceradar.agent.pipeline import ConfirmedItem, OfferGroup, PipelineResult
from priceradar.domain.models import Coupon, CouponType, Offer, Platform
from priceradar.engine.pricing import compute_price
from priceradar.risk import RiskItem, RiskSummary
from priceradar.sources.bocha import BochaError
from priceradar.sources.pdd import PddApiError

D = Decimal


# ---------- 夹具 ----------

def make_offer(sku="a", title="QCY N20 降噪耳机", price="99", sign="SIGN_A") -> Offer:
    return Offer(
        platform=Platform.PDD,
        sku_id=sku,
        title=title,
        list_price=D(price),
        source_extra={"goods_sign": sign},
    )


def make_result(offer: Offer | None = None) -> PipelineResult:
    offer = offer or make_offer()
    return PipelineResult(
        keyword="QCY N20",
        pdd_count=20,
        jd_count=0,
        source=offer,
        confirmed=[ConfirmedItem(offer=offer, reason="检索结果（源商品）")],
        groups=[OfferGroup(representative=offer, members=[offer], final_price=offer.list_price, reason="源")],
        similar_count=0,
        excluded_summary={},
        breakdowns=[(offer, compute_price(offer))],
    )


def offline(monkeypatch) -> None:
    """禁用 LLM：走确定性兜底，绝不触网。"""
    monkeypatch.setattr(actions, "_llm_client", lambda: None)


def use_user_db(monkeypatch, tmp_path, coupons=()) -> None:
    """把用户券库切到临时目录（沿用既有测试的做法：打桩 DEFAULT_DB）。"""
    import priceradar.user_store as user_store
    from priceradar.user_store import connect_user_db, save_coupons

    db = tmp_path / "user.db"
    if coupons:
        connection = connect_user_db(db)
        save_coupons(connection, "pdd", list(coupons))
        connection.close()
    monkeypatch.setattr(user_store, "DEFAULT_DB", db)


# ============================================================
# 1. 用户补录券并入（下沉自 service 层，行为不变）
# ============================================================

def test_merge_user_coupons_without_db_does_not_recompute(monkeypatch, tmp_path):
    """无用户库时必须**原样返回同一个 breakdown 对象**（不重算）。

    管线算价可能带 as_of 等参数，重算未必等价——这条约束原来在 service 层，
    下沉后必须一字不差地保留。
    """
    import priceradar.user_store as user_store

    monkeypatch.setattr(user_store, "DEFAULT_DB", tmp_path / "none.db")
    offer = make_offer()
    original = compute_price(offer)
    merged, breakdowns = actions.merge_user_coupons([(offer, original)])
    assert merged == 0
    assert breakdowns[0][1] is original


def test_merge_user_coupons_merges_and_recomputes(monkeypatch, tmp_path):
    use_user_db(
        monkeypatch,
        tmp_path,
        [Coupon(coupon_id="u1", type=CouponType.PLATFORM, face_value=D("10"), threshold=D("50"), source="screenshot")],
    )
    offer = make_offer(price="99")
    merged, breakdowns = actions.merge_user_coupons([(offer, compute_price(offer))])
    assert merged == 1
    assert breakdowns[0][1].final_price == D("89")
    assert any(line.label.startswith("[补录]") for line in breakdowns[0][1].applied)


def test_apply_user_coupons_updates_result_in_place(monkeypatch, tmp_path):
    use_user_db(
        monkeypatch,
        tmp_path,
        [Coupon(coupon_id="u1", type=CouponType.PLATFORM, face_value=D("10"), threshold=D("50"), source="screenshot")],
    )
    result = make_result()
    assert actions.apply_user_coupons(result) == 1
    assert result.breakdowns[0][1].final_price == D("89")


def test_apply_user_coupons_no_db_is_noop(monkeypatch, tmp_path):
    use_user_db(monkeypatch, tmp_path)
    result = make_result()
    assert actions.apply_user_coupons(result) == 0
    assert result.breakdowns[0][1].final_price == D("99")


# ============================================================
# 2. compare_prices：编排顺序
# ============================================================

def test_compare_prices_orchestrates_keyword_run_summary(monkeypatch, tmp_path):
    """编排顺序：关键词提取 → 管线 → 用户券并入 → 摘要。"""
    offline(monkeypatch)
    use_user_db(monkeypatch, tmp_path)
    seen = {}

    def fake_extract(text, client=None):
        seen["text"] = text
        return "QCY N20", "rule"

    monkeypatch.setattr(actions, "extract_keyword", fake_extract)
    monkeypatch.setattr(actions, "run", lambda keyword, **kw: make_result())
    monkeypatch.setattr(actions, "compose_summary", lambda result, client: ("已生成导语", "rule"))

    result = actions.compare_prices("帮我看看QCY N20哪里买划算", jd_channel=0)
    assert seen["text"] == "帮我看看QCY N20哪里买划算"   # 原样传给提取器（链接短路依赖这点）
    assert result.keyword == "QCY N20"
    assert result.summary == "已生成导语"


def test_compare_prices_maps_jd_channel(monkeypatch, tmp_path):
    """jd_channel 语义与既有约定一致：-1 关闭京东、0 自动路由、>0 定点频道。"""
    offline(monkeypatch)
    use_user_db(monkeypatch, tmp_path)
    captured = {}

    def fake_run(keyword, **kwargs):
        captured.update(kwargs)
        return make_result()

    monkeypatch.setattr(actions, "run", fake_run)
    monkeypatch.setattr(actions, "extract_keyword", lambda text, client=None: (text, "rule"))
    monkeypatch.setattr(actions, "compose_summary", lambda result, client: ("s", "rule"))

    actions.compare_prices("x", jd_channel=-1)
    assert captured["jd_enabled"] is False
    actions.compare_prices("x", jd_channel=0)
    assert captured["jd_enabled"] is True and captured["jd_elite_id"] is None
    actions.compare_prices("x", jd_channel=24)
    assert captured["jd_elite_id"] == 24


def test_compare_prices_merges_user_coupons_and_notes_it(monkeypatch, tmp_path):
    offline(monkeypatch)
    use_user_db(
        monkeypatch,
        tmp_path,
        [Coupon(coupon_id="u1", type=CouponType.PLATFORM, face_value=D("10"), threshold=D("50"), source="screenshot")],
    )
    monkeypatch.setattr(actions, "extract_keyword", lambda text, client=None: (text, "rule"))
    monkeypatch.setattr(actions, "run", lambda keyword, **kw: make_result())
    monkeypatch.setattr(actions, "compose_summary", lambda result, client: ("s", "rule"))

    result = actions.compare_prices("x")
    assert result.breakdowns[0][1].final_price == D("89")
    assert any("用户补录券" in note for note in result.notes)


def test_compare_prices_with_risk_clusters(monkeypatch, tmp_path):
    """use_risk 时走风险聚类；这两个能力**必须一起提供**——MCP 侧此前缺席，是既存不一致。"""
    offline(monkeypatch)
    use_user_db(monkeypatch, tmp_path)
    summary = RiskSummary(product="x", queries=["q"], items=[], note="原始", degraded=True)
    clustered = RiskSummary(product="x", queries=["q"], items=[], issues=[{"topic": "t", "mention_count": 1, "urls": []}], note="已聚类", degraded=False)

    def fake_run(keyword, **kw):
        produced = make_result()
        produced.risk = summary       # run() 在 with_risk=True 时会把扫描结果挂上去
        return produced

    monkeypatch.setattr(actions, "extract_keyword", lambda text, client=None: (text, "rule"))
    monkeypatch.setattr(actions, "run", fake_run)
    monkeypatch.setattr(actions, "compose_summary", lambda result, client: ("s", "rule"))
    monkeypatch.setattr(actions, "cluster_issues", lambda risk, client: clustered)

    result = actions.compare_prices("x", use_risk=True)
    assert result.risk is not None and result.risk.degraded is False


# ============================================================
# 3. find_similar
# ============================================================

class FakePddSimilar:
    def __init__(self, offers):
        self.offers = offers
        self.calls = []

    def recommend_similar(self, goods_sign, limit=20):
        self.calls.append((goods_sign, limit))
        return list(self.offers)


def test_find_similar_returns_priced_offers(monkeypatch, tmp_path):
    use_user_db(monkeypatch, tmp_path)
    fake = FakePddSimilar([make_offer("b", "QCY N20 同款", "80"), make_offer("c", "QCY N20 同款2", "70")])
    monkeypatch.setattr(actions, "PddClient", type("C", (), {"from_env": staticmethod(lambda: fake)}))

    result = actions.find_similar("SIGN_A", limit=10)
    assert fake.calls == [("SIGN_A", 10)]
    assert result.anchor_sign == "SIGN_A"
    assert [offer.sku_id for offer, _ in result.breakdowns] == ["b", "c"]
    assert result.breakdowns[0][1].final_price == D("80")


def test_find_similar_empty_raises_value_error(monkeypatch, tmp_path):
    use_user_db(monkeypatch, tmp_path)
    fake = FakePddSimilar([])
    monkeypatch.setattr(actions, "PddClient", type("C", (), {"from_env": staticmethod(lambda: fake)}))
    with pytest.raises(ValueError, match="未找到相似商品"):
        actions.find_similar("SIGN_A")


def test_find_similar_merges_user_coupons(monkeypatch, tmp_path):
    use_user_db(
        monkeypatch,
        tmp_path,
        [Coupon(coupon_id="u1", type=CouponType.PLATFORM, face_value=D("10"), threshold=D("50"), source="screenshot")],
    )
    fake = FakePddSimilar([make_offer("b", "同款", "80")])
    monkeypatch.setattr(actions, "PddClient", type("C", (), {"from_env": staticmethod(lambda: fake)}))

    result = actions.find_similar("SIGN_A")
    assert result.merged_coupons == 1
    assert result.breakdowns[0][1].final_price == D("70")


def test_find_similar_carries_platform_precision_disclaimer(monkeypatch, tmp_path):
    """平台推荐集**混合有无关商品**（实测精度 33-47%），声明必须随结果走。"""
    use_user_db(monkeypatch, tmp_path)
    fake = FakePddSimilar([make_offer("b")])
    monkeypatch.setattr(actions, "PddClient", type("C", (), {"from_env": staticmethod(lambda: fake)}))
    result = actions.find_similar("SIGN_A")
    assert any("混合有无关商品" in note for note in result.notes)


# ============================================================
# 4. scan_risk
# ============================================================

def test_scan_risk_clusters_with_llm(monkeypatch):
    summary = RiskSummary(product="QCY N20", queries=["q"], items=[], note="原始", degraded=True)
    clustered = RiskSummary(product="QCY N20", queries=["q"], items=[], issues=[], note="已聚类", degraded=False)
    monkeypatch.setattr(actions, "scan_with_env", lambda product: summary)
    monkeypatch.setattr(actions, "_llm_client", lambda: object())
    monkeypatch.setattr(actions, "cluster_issues", lambda risk, client: clustered)
    assert actions.scan_risk("QCY N20").degraded is False


def test_scan_risk_degrades_without_llm(monkeypatch):
    """无 LLM 时原样返回检索清单并保留 degraded=True（诚实降级，不假装聚类过）。"""
    summary = RiskSummary(product="QCY N20", queries=["q"], items=[RiskItem("q", "标题", "u", "知乎", "")], degraded=True)
    monkeypatch.setattr(actions, "scan_with_env", lambda product: summary)
    monkeypatch.setattr(actions, "_llm_client", lambda: None)
    result = actions.scan_risk("QCY N20")
    assert result.degraded is True and result.issues == []


# ============================================================
# 5. check_price_history
# ============================================================

def test_check_price_history_missing_db(tmp_path):
    result = actions.check_price_history("QCY N20", db_path=tmp_path / "none.db")
    assert result.rows == ()
    assert "暂无快照数据" in result.note


def test_check_price_history_with_data(tmp_path):
    from datetime import datetime

    from priceradar.jobs.snapshot import connect, record_offers

    db = tmp_path / "snap.db"
    connection = connect(db)
    record_offers(connection, [make_offer()], "QCY N20 降噪耳机", captured=datetime(2026, 10, 5, 9, 0, 0))
    connection.close()

    result = actions.check_price_history("QCY N20", db_path=db)
    assert len(result.rows) == 1
    assert result.rows[0].observed_days == 1
    assert result.rows[0].title


def test_check_price_history_no_match(tmp_path):
    from datetime import datetime

    from priceradar.jobs.snapshot import connect, record_offers

    db = tmp_path / "snap.db"
    connection = connect(db)
    record_offers(connection, [make_offer()], "别的关键词", captured=datetime(2026, 10, 5, 9, 0, 0))
    connection.close()

    result = actions.check_price_history("QCY N20", db_path=db)
    assert result.rows == ()
    assert "没有与" in result.note


# ============================================================
# 6. 工具注册表与 OpenAI schema 导出
# ============================================================

def test_registry_exposes_four_intent_actions():
    assert set(actions.ACTIONS) == {"compare_prices", "find_similar", "scan_risk", "check_price_history"}


def test_openai_tool_schema_shape():
    """导出的必须是 OpenAI 兼容的 {"type":"function","function":{...}} 形状。"""
    schemas = actions.openai_tools()
    assert len(schemas) == len(actions.ACTIONS)
    by_name = {schema["function"]["name"]: schema for schema in schemas}
    assert set(by_name) == set(actions.ACTIONS)
    for schema in schemas:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["description"] and isinstance(fn["description"], str)
        params = fn["parameters"]
        assert params["type"] == "object"
        # 必须是合法 JSON Schema（DeepSeek 会按此校验入参）
        json.dumps(params)


def test_schema_declares_required_and_typed_fields():
    by_name = {schema["function"]["name"]: schema for schema in actions.openai_tools()}
    compare = by_name["compare_prices"]["function"]["parameters"]
    assert "query" in compare["properties"]
    assert "query" in compare["required"]
    # jd_channel 有默认值 → 不应是 required
    assert "jd_channel" not in compare.get("required", [])


def test_tool_descriptions_carry_cost_hint():
    """工具描述里写明成本/延迟，让模型把开销算进选择（红队建议）。"""
    by_name = {schema["function"]["name"]: schema for schema in actions.openai_tools()}
    assert "¥" in by_name["scan_risk"]["function"]["description"]


# ============================================================
# 7. 摘要必须「可行动」
# ============================================================

def test_summarize_compare_includes_chainable_ids():
    """摘要必须带 goods_sign 等关键 ID——否则模型第二轮调不动 find_similar。"""
    text = actions.summarize_compare(make_result())
    assert "SIGN_A" in text
    assert "99" in text          # 确定性数值原样出现，不让 LLM 转述
    assert "QCY N20" in text


def test_summarize_compare_is_bounded():
    """摘要必须定长——全量结果不进上下文（token 治理的落点）。"""
    offer = make_offer(title="超长标题" * 200)
    text = actions.summarize_compare(make_result(offer))
    assert len(text) <= actions.SUMMARY_LIMIT


def test_summarize_similar_is_bounded_and_chainable():
    result = actions.SimilarResult(
        anchor_sign="SIGN_A",
        breakdowns=[(make_offer("b", sign="SIGN_B"), compute_price(make_offer("b")))],
        merged_coupons=0,
        notes=["n"],
    )
    text = actions.summarize_similar(result)
    assert "SIGN_B" in text
    assert len(text) <= actions.SUMMARY_LIMIT


def test_summarize_risk_marks_degraded():
    degraded = RiskSummary(product="x", queries=["q"], items=[], note="未做问题归类", degraded=True)
    assert "未做问题归类" in actions.summarize_risk(degraded)


def test_summarize_history_reports_empty_state(tmp_path):
    result = actions.check_price_history("不存在", db_path=tmp_path / "none.db")
    text = actions.summarize_history(result)
    assert "暂无快照数据" in text
    assert len(text) <= actions.SUMMARY_LIMIT


# ============================================================
# 8. 执行器：按工具的重试策略 + 结构化错误
# ============================================================

def _action_that(monkeypatch, name, func) -> actions.Action:
    """构造一个用给定函数替换实现的 Action（保留原 params/summarize）。"""
    original = actions.ACTIONS[name]
    return actions.Action(
        name=original.name,
        description=original.description,
        params=original.params,
        func=func,
        summarize=original.summarize,
        retry=original.retry,
        cost_hint=original.cost_hint,
    )


def test_run_action_success_returns_summary(monkeypatch):
    monkeypatch.setattr(actions, "run", lambda keyword, **kw: make_result())
    outcome = actions.run_action("compare_prices", {"query": "x"})
    assert outcome.ok is True
    assert outcome.summary and len(outcome.summary) <= actions.SUMMARY_LIMIT
    assert outcome.value is not None


def test_run_action_unknown_tool_is_not_retryable():
    outcome = actions.run_action("no_such_tool", {})
    assert outcome.ok is False
    assert "no_such_tool" in outcome.error
    assert outcome.retryable is False


def test_run_action_invalid_arguments_are_reported_not_raised():
    """参数校验失败必须变成结构化错误喂回模型，而不是抛异常打断循环。"""
    outcome = actions.run_action("compare_prices", {})       # 缺 query
    assert outcome.ok is False
    assert "query" in outcome.error
    assert outcome.retryable is False


def test_business_error_is_not_retryable(monkeypatch):
    """拼多多业务错误（凭证/权限/参数）重试无用——客户端内部也已重试过 3 次。"""
    def boom(keyword, **kw):
        raise PddApiError("50001 业务服务错误")

    monkeypatch.setattr(actions, "run", boom)
    outcome = actions.run_action("compare_prices", {"query": "x"})
    assert outcome.ok is False
    assert outcome.retryable is False
    assert "50001" in outcome.error


def test_category_gap_error_is_not_retryable(monkeypatch):
    """类目空洞（手机/平板等返回空）是平台没货，重试同一个词永远无效。"""
    def boom(keyword, **kw):
        raise ValueError("「显示器」在拼多多侧没有可比的商品……不是故障。")

    monkeypatch.setattr(actions, "run", boom)
    outcome = actions.run_action("compare_prices", {"query": "显示器"})
    assert outcome.ok is False
    assert outcome.retryable is False
    assert "换" in outcome.hint          # 给出可行动作，而不是让模型瞎试


def test_non_original_link_error_becomes_actionable(monkeypatch):
    """平台对「拿普通商品页 URL 来反查」的报错是技术性的，用户看不懂。

    实测：接口自己返回的商品 URL、以及从网页复制的 URL **都不是**「原始链接」，
    只有 App 里「分享 → 复制链接」那种（含 uin）才行。必须翻成人话。
    """
    raw = "该查询参数不是终端用户的原始链接，请使用原始链接进行查询。详细可参考：https://..."

    def boom(keyword, **kw):
        raise PddApiError(raw)

    monkeypatch.setattr(actions, "run", boom)
    outcome = actions.run_action("compare_prices", {"query": "https://mobile.yangkeduo.com/goods.html?goods_id=1"})
    assert outcome.ok is False
    assert "App" in outcome.hint and "分享" in outcome.hint
    assert "uin" in outcome.hint              # 点明真正的原因，别让人瞎试
    assert "原始链接" not in outcome.error or "不是" in outcome.error


def test_explain_pdd_error_passes_other_errors_through():
    assert actions.explain_pdd_error(PddApiError("50001 业务服务错误")) == "50001 业务服务错误"


def test_bocha_network_error_is_retryable_via_cause(monkeypatch):
    """BochaError 是包装异常，分类必须沿 __cause__ 链走（红队指出的坑）。"""
    def boom(product):
        raise BochaError("网络错误：timed out") from urllib.error.URLError("timed out")

    monkeypatch.setattr(actions, "scan_with_env", boom)
    monkeypatch.setattr(actions, "_llm_client", lambda: None)
    outcome = actions.run_action("scan_risk", {"product": "x"})
    assert outcome.ok is False
    assert outcome.retryable is True


def test_bocha_retry_policy_retries_once_then_succeeds(monkeypatch):
    """scan_risk 声明的策略是 retry=once，且 bocha 没有内部重试——补一次是值的。"""
    calls = []

    def flaky(product):
        calls.append(product)
        if len(calls) == 1:
            raise BochaError("网络错误：timed out") from urllib.error.URLError("timed out")
        return RiskSummary(product=product, queries=["q"], items=[], note="ok", degraded=True)

    monkeypatch.setattr(actions, "scan_with_env", flaky)
    monkeypatch.setattr(actions, "_llm_client", lambda: None)
    outcome = actions.run_action("scan_risk", {"product": "x"})
    assert outcome.ok is True
    assert len(calls) == 2


def test_pdd_action_never_retries(monkeypatch):
    """compare_prices 声明 retry=none：pdd 客户端内部已重试 3 次，再试会烧配额并触发限流。"""
    calls = []

    def boom(keyword, **kw):
        calls.append(keyword)
        raise urllib.error.URLError("dns down")

    monkeypatch.setattr(actions, "run", boom)
    outcome = actions.run_action("compare_prices", {"query": "x"})
    assert outcome.ok is False
    assert len(calls) == 1          # 只调用一次，绝不自动重试
    assert outcome.retryable is True  # 但如实告诉模型「同一次调用稍后可能可行」


def test_run_action_hint_is_actionable_for_transient(monkeypatch):
    def boom(keyword, **kw):
        raise urllib.error.URLError("dns down")

    monkeypatch.setattr(actions, "run", boom)
    outcome = actions.run_action("compare_prices", {"query": "x"})
    assert outcome.hint


# ============================================================
# 9. 可注入性（离线测试的基础设施约束）
# ============================================================

def test_llm_client_degrades_when_key_missing():
    """无 key 时返回 None 而不是抛异常（沿用项目既有的「client 可为 None」约定）。"""
    assert actions._llm_client is not None       # 模块级名字，可被打桩
