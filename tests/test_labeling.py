"""真实标注工具链测试（候选构建 / 标签解析 / 增量续标，全离线）。"""

from decimal import Decimal

from evaluation.collect_candidates import _pair_id, build_pairs
from evaluation.label_tool import load_jsonl, parse_label, run_labeling
from priceradar.domain.models import Offer, Platform

D = Decimal


def make_offer(sku: str, title: str, platform: Platform = Platform.PDD) -> Offer:
    return Offer(platform=platform, sku_id=sku, title=title, list_price=D("99"))


def test_build_pairs_dedupes_and_balances():
    pdd = [
        make_offer("s", "源商品 QCY N20 降噪耳机"),
        make_offer("p1", "候选甲 QCY N20"),
        make_offer("p2", "候选乙 漫步者耳机"),
        make_offer("p3", "候选丙 重复标题"),
        make_offer("p4", "候选丙 重复标题"),
    ]
    jd = [
        make_offer("j1", "京东候选一", Platform.JD),
        make_offer("j2", "京东候选二", Platform.JD),
        make_offer("j3", "京东候选三", Platform.JD),
    ]
    pairs = build_pairs("QCY N20 降噪耳机", pdd, jd, per_seed=4)
    assert len(pairs) == 4
    assert all(pair.source_title == "源商品 QCY N20 降噪耳机" for pair in pairs)
    titles = [pair.candidate_title for pair in pairs]
    assert len(titles) == len(set(titles))  # 标题去重
    assert any(pair.candidate_platform == "jd" for pair in pairs)  # 平台均衡
    assert len({pair.pair_id for pair in pairs}) == 4


def test_build_pairs_empty_source():
    assert build_pairs("x", [], [make_offer("j", "y", Platform.JD)]) == []


def test_pair_id_is_stable():
    assert _pair_id("seed", "标题") == _pair_id("seed", "标题")
    assert _pair_id("seed", "标题") != _pair_id("seed", "标题2")


def test_parse_label_mapping():
    assert parse_label("Y") == "A"
    assert parse_label(" b ") == "B"
    assert parse_label("n") == "excluded"
    assert parse_label("s") is None
    assert parse_label("q") is None
    assert parse_label("") is None


def make_candidates(count: int) -> list[dict]:
    return [
        {
            "pair_id": f"p{index}",
            "seed": "s",
            "source_title": "源",
            "source_platform": "pdd",
            "candidate_title": f"候选{index}",
            "candidate_platform": "jd",
        }
        for index in range(count)
    ]


def test_run_labeling_writes_and_resumes(tmp_path):
    labeled = tmp_path / "labeled.jsonl"
    scripted = iter(["y", "n"])
    count = run_labeling(
        make_candidates(2), labeled, input_func=lambda prompt: next(scripted), output_func=lambda text: None
    )
    assert count == 2
    assert [row["expected"] for row in load_jsonl(labeled)] == ["A", "excluded"]

    # 全部已标 → 跳过；新增一条 → 只标新的（续标）
    assert run_labeling(make_candidates(2), labeled, input_func=lambda prompt: "y", output_func=lambda text: None) == 0
    assert run_labeling(make_candidates(3), labeled, input_func=lambda prompt: "b", output_func=lambda text: None) == 1


def test_run_labeling_quit_saves_partial(tmp_path):
    labeled = tmp_path / "labeled.jsonl"
    scripted = iter(["y", "q"])
    count = run_labeling(
        make_candidates(3), labeled, input_func=lambda prompt: next(scripted), output_func=lambda text: None
    )
    assert count == 1
    assert len(load_jsonl(labeled)) == 1


def test_labeled_records_flow_through_evaluator(tmp_path):
    """标注产出的记录 schema 与评测器兼容（盲标 → 指标闭环）。"""
    import evaluation.run_eval as runner

    labeled = tmp_path / "labeled.jsonl"
    run_labeling(make_candidates(1), labeled, input_func=lambda prompt: "y", output_func=lambda text: None)
    rows = load_jsonl(labeled)
    stats = runner.evaluate(rows)  # make_candidates 的源/候选都无规格锚点 → 匹配器输出 excluded，标签 A → 计为错例
    assert stats["total"] == 1
    assert "accuracy" in stats
