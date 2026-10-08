"""评测集构建与评估运行器测试。"""

import evaluation.run_eval as runner
from evaluation.build_dataset import ALLOWED_EXPECTED, build_records


def test_dataset_size_and_fields():
    records = build_records()
    assert len(records) >= 200, f"评测集应 ≥200 对，当前 {len(records)}"
    ids = [record.id for record in records]
    assert len(ids) == len(set(ids)), "id 必须唯一"
    for record in records:
        assert record.expected in ALLOWED_EXPECTED
        assert record.source_title.strip() and record.candidate_title.strip()


def test_dataset_covers_all_case_kinds():
    kinds = {record.kind for record in build_records()}
    for required in ("fluff_injection", "model_swap", "capacity_swap", "accessory_wrap", "brand_swap", "unrelated", "suspicious_price"):
        assert required in kinds


def test_evaluate_confusion_math(monkeypatch):
    mapping = {"t1": "A", "t2": "B", "t3": "A", "t4": "excluded"}
    monkeypatch.setattr(runner, "classify", lambda s, c, sp, cp: mapping[s])
    records = [
        {"id": "1", "kind": "k", "source_title": "t1", "candidate_title": "c", "expected": "A"},  # TP
        {"id": "2", "kind": "k", "source_title": "t2", "candidate_title": "c", "expected": "B"},  # 对
        {"id": "3", "kind": "k", "source_title": "t3", "candidate_title": "c", "expected": "B"},  # FP
        {"id": "4", "kind": "k", "source_title": "t4", "candidate_title": "c", "expected": "A"},  # FN
    ]
    stats = runner.evaluate(records)
    assert stats["total"] == 4
    assert stats["accuracy"] == 0.5
    assert stats["precision"] == 0.5
    assert stats["recall"] == 0.5
    assert len(stats["mismatches"]) == 2


def test_real_dataset_accuracy_floor():
    """回归护栏：规则层在构造集上的准确率下限（跌破说明改坏了）。"""
    stats = runner.evaluate([record.__dict__ for record in build_records()])
    assert stats["accuracy"] >= 0.85, f"准确率跌破护栏：{stats['accuracy']:.2%}，错例：{stats['mismatches'][:5]}"
