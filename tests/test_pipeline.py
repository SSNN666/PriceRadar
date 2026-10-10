"""单轮管线测试（searcher 注入，全离线）。"""

from decimal import Decimal

import pytest

from priceradar.agent.pipeline import _pick_source, render, run
from priceradar.domain.models import Offer, Platform
from priceradar.sources.jd import JdApiError

D = Decimal

PDD_N20_A = "【QCY】 N20降噪蓝牙耳机无线入耳式长续航游戏"
PDD_N20_B = "QCY N20 主动降噪蓝牙耳机 入耳式 超长续航"
PDD_N60 = "QCY N60主动降噪蓝牙耳机 入耳式 AI翻译"
UNRELATED_LAMP = "雷士 led长条灯管 一体化 高显色灯条"
JD_ACCESSORY = "适用于QCY N20耳机保护套 硅胶 防摔"


def make_offer(sku: str, title: str, price: str = "100", platform: Platform = Platform.PDD) -> Offer:
    return Offer(platform=platform, sku_id=sku, title=title, list_price=D(price))


def pdd_searcher_factory(*offers):
    def searcher(keyword: str, page_size: int):
        return list(offers)

    return searcher


def jd_searcher_factory(*offers):
    """京东侧现按关键词检索（品类路由）——签名从 (elite_id, size) 改为 (keyword)。"""

    def searcher(_keyword: str):
        return list(offers)

    return searcher


def test_pipeline_picks_representative_source_and_confirms_group():
    pool = [
        make_offer("a", PDD_N20_A, "99"),
        make_offer("b", PDD_N20_B, "105"),
        make_offer("c", PDD_N60, "181"),
        make_offer("d", UNRELATED_LAMP, "85"),
    ]
    result = run(
        "降噪耳机",
        pdd_searcher=pdd_searcher_factory(*pool),
        jd_searcher=jd_searcher_factory(make_offer("j1", JD_ACCESSORY, platform=Platform.JD)),
    )
    assert "N20" in result.source.title  # 出现次数最多的同款组被选为源
    assert len(result.confirmed) == 2  # 源 + 1 个 A 档同款
    assert result.confirmed[1].reason.startswith("型号一致")
    assert result.similar_count == 1  # N60 落入近似款
    assert result.similar_items and "型号不同" in result.similar_items[0].reason  # 近似款明细随结果透出
    assert result.excluded_summary.get("配件/周边类商品") == 1
    assert len(result.breakdowns) == 2


def test_source_selection_scans_the_whole_pool_not_just_the_head():
    """源商品的选择**不该依赖接口返回顺序**。

    实测（2026-10-10）：拼多多接口的返回顺序与到手价的秩相关约 **-0.35**
    （贵的反而略靠前，更像按推广价值排的），并不按相关度。某个关键词下真正
    的最优源商品排在第 24 位——探针只看前 5 条会直接错过。

    全量扫描是纯 Python 计算，30 条约 25 ms；一次查询总耗时以秒计，省这 20 毫秒
    不值得拿正确性去换。
    """
    head = [
        make_offer("h0", "雷士 led长条灯管 一体化 高显色灯条"),
        make_offer("h1", "PPR水管接头 4分 加厚"),
        make_offer("h2", "不锈钢保温杯 316L 便携"),
        make_offer("h3", "厨房收纳架 双层 免打孔"),
        make_offer("h4", "洗衣液 薰衣草 补充装"),
    ]
    # 真正的最优源藏在后面：同一款被三家铺货
    tail = [
        make_offer("t0", PDD_N20_A),
        make_offer("t1", PDD_N20_B),
        make_offer("t2", "QCY N20降噪蓝牙耳机 无线入耳式 长续航"),
    ]
    pool = head + tail

    assert _pick_source(pool).sku_id in {"t0", "t1", "t2"}   # 默认扫全量
    assert _pick_source(pool, probe=5).sku_id in {"h0"}      # 只扫前 5 条 → 错过


def test_pipeline_cross_platform_hit():
    pool = [make_offer("a", PDD_N20_A, "99"), make_offer("b", PDD_N60, "181")]
    jd_same = make_offer("j1", PDD_N20_B, "95", platform=Platform.JD)
    result = run("降噪耳机", pdd_searcher=pdd_searcher_factory(*pool), jd_searcher=jd_searcher_factory(jd_same))
    assert result.cross_platform_hit is True


def test_pipeline_degrades_when_jd_fails():
    def boom(_keyword: str):
        raise JdApiError("403 无访问权限")

    pool = [make_offer("a", PDD_N20_A, "99"), make_offer("b", PDD_N20_B, "105")]
    result = run("降噪耳机", pdd_searcher=pdd_searcher_factory(*pool), jd_searcher=boom)
    assert result.jd_error is not None
    assert any("京东侧检索失败" in note for note in result.notes)
    assert len(result.breakdowns) == 2  # 仍出拼多多对比卡


def test_pipeline_no_match_fallback_note():
    pool = [make_offer("a", UNRELATED_LAMP, "85"), make_offer("b", "PPR水管接头 4分 加厚", "12")]
    result = run("灯", pdd_searcher=pdd_searcher_factory(*pool), jd_searcher=jd_searcher_factory())
    assert len(result.confirmed) == 1
    assert any("未找到确切同款" in note for note in result.notes)


def test_pipeline_empty_pool_raises():
    with pytest.raises(ValueError):
        run("不存在", pdd_searcher=pdd_searcher_factory(), jd_enabled=False)


def test_empty_pool_error_explains_category_gap_and_offers_a_way_out():
    """拼多多有**类目空洞**（实测手机/平板/显示器/电视/键盘/鼠标返回空）。

    报错必须说清"这是平台没货、不是故障"，并给出可行出路——否则用户以为程序坏了。
    """
    with pytest.raises(ValueError) as excinfo:
        run("显示器", pdd_searcher=pdd_searcher_factory(), jd_enabled=False)
    message = str(excinfo.value)
    assert "显示器" in message
    assert "不是故障" in message
    assert "商品链接" in message          # 给出可操作的替代路径（链接反查已验证可用）


def test_render_contains_all_sections():
    pool = [make_offer("a", PDD_N20_A, "99"), make_offer("b", PDD_N20_B, "105")]
    result = run("降噪耳机", pdd_searcher=pdd_searcher_factory(*pool), jd_searcher=jd_searcher_factory())
    text = render(result)
    assert "真价雷达 · 单轮管线" in text
    assert "【同款组】" in text
    assert "【到手价明细】" in text
    assert "以结算页为准" in text  # 免责声明
    assert "拼多多" in text
    assert "[trace " in text  # trace 摘要行


# ---------- 同款去重折叠 ----------

def test_same_price_same_model_collapses_into_one_group():
    pool = [
        make_offer("a", PDD_N20_A, "99"),
        make_offer("b", PDD_N20_B, "99"),
        make_offer("c", "QCY N20降噪蓝牙耳机 无线入耳式 长续航 游戏专用", "99"),
    ]
    result = run("降噪耳机", pdd_searcher=pdd_searcher_factory(*pool), jd_searcher=jd_searcher_factory())
    assert len(result.confirmed) == 3
    assert len(result.groups) == 1
    assert result.groups[0].count == 3
    assert result.groups[0].representative.sku_id == "a"  # 源商品作为组代表
    text = render(result)
    assert "同款 ×3 家" in text
    assert len(result.breakdowns) == 1  # 卡片每组只出一行


def test_different_price_not_collapsed():
    pool = [make_offer("a", PDD_N20_A, "99"), make_offer("b", PDD_N20_B, "105")]
    result = run("降噪耳机", pdd_searcher=pdd_searcher_factory(*pool), jd_searcher=jd_searcher_factory())
    assert len(result.groups) == 2


def test_cross_platform_same_price_not_collapsed():
    pool = [make_offer("a", PDD_N20_A, "99")]
    jd_same = make_offer("j1", PDD_N20_B, "99", platform=Platform.JD)
    result = run("降噪耳机", pdd_searcher=pdd_searcher_factory(*pool), jd_searcher=jd_searcher_factory(jd_same))
    assert len(result.groups) == 2  # 不同平台不折叠
    assert result.cross_platform_hit is True


# ---------- trace 集成 ----------

def test_pipeline_attaches_trace_with_spans():
    pool = [make_offer("a", PDD_N20_A, "99")]
    result = run("降噪耳机", pdd_searcher=pdd_searcher_factory(*pool), jd_searcher=jd_searcher_factory())
    assert result.trace is not None
    names = [span.name for span in result.trace.spans]
    assert "pdd_search" in names and "jd_search" in names and "match" in names and "pricing" in names
    assert result.trace.api_calls["pdd"] == 1


# ---------- 风险扫描集成 ----------

def test_pipeline_with_risk_scan():
    from priceradar.risk import RiskItem, RiskSummary

    def fake_scanner(name: str) -> RiskSummary:
        return RiskSummary(
            product=name,
            queries=[f"{name} 避雷"],
            items=[RiskItem(query=f"{name} 避雷", title="翻车实录", url="https://x.com/1", site_name="小红书", snippet="s")],
            note="未做问题归类（LLM 未启用/失败）；以下为检索结果原始清单",
        )

    pool = [make_offer("a", PDD_N20_A, "99")]
    result = run(
        "降噪耳机",
        pdd_searcher=pdd_searcher_factory(*pool),
        jd_searcher=jd_searcher_factory(),
        with_risk=True,
        risk_scanner=fake_scanner,
    )
    assert result.risk is not None
    text = render(result)
    assert "【风险扫描】" in text
    assert "不代表全部用户反馈" in text  # 覆盖声明
    assert result.trace.api_calls.get("search_bocha") == 1


def test_pipeline_risk_failure_degrades_gracefully():
    def boom(name: str):
        raise RuntimeError("bocha 挂了")

    pool = [make_offer("a", PDD_N20_A, "99")]
    result = run("降噪耳机", pdd_searcher=pdd_searcher_factory(*pool), jd_searcher=jd_searcher_factory(), with_risk=True, risk_scanner=boom)
    assert result.risk is None
    assert any("风险扫描未执行" in note for note in result.notes)


# ---------- 价格观测集成 ----------

def test_pipeline_history_enrichment_render():
    pool = [make_offer("a", PDD_N20_A, "99")]

    def fake_enricher(result):
        result.history["pdd:a"] = {
            "observed_days": 21,
            "status": "ok",
            "verdict": "当前处于观测低位（第 10% 分位），可考虑入手",
            "note": "基于自观测 21 天数据（非全历史）",
            "lowest": "86（2026-10-12）",
        }

    result = run(
        "降噪耳机",
        pdd_searcher=pdd_searcher_factory(*pool),
        jd_searcher=jd_searcher_factory(),
        history_enricher=fake_enricher,
    )
    text = render(result)
    assert "【价格观测】" in text
    assert "已观测 21 天" in text
    assert "观测最低：￥86（2026-10-12）" in text
