"""运行同款匹配评测：数据集 → 匹配器 → 指标报告。

指标：
- 总体准确率（实际输出 vs 期望，四类：A/B/C/excluded）
- 同款检出（expected==A 为正类）：Precision / Recall
- 按 kind 的分类目明细（暴露各层强弱）+ 全部错例清单
输出：终端打印 + evaluation/results_v1.md（可引用进 README / 简历）
"""

from __future__ import annotations

import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from priceradar.domain.models import Offer, Platform
from priceradar.matching import match_offers

DATASET = Path(__file__).parent / "dataset_v1.jsonl"
LABELED = Path(__file__).parent / "labeled_v2.jsonl"
RESULTS = Path(__file__).parent / "results_v1.md"
RESULTS_V2 = Path(__file__).parent / "results_v2.md"


def classify(source_title: str, candidate_title: str, source_price: str, candidate_price: str) -> str:
    """运行匹配器，把输出归一为 A / B / C / excluded。"""
    source = Offer(platform=Platform.PDD, sku_id="src", title=source_title, list_price=Decimal(source_price))
    candidate = Offer(platform=Platform.JD, sku_id="cand", title=candidate_title, list_price=Decimal(candidate_price))
    report = match_offers(source, [candidate])
    if report.results:
        return report.results[0].tier.value
    return "excluded"


def evaluate(records: list[dict]) -> dict:
    per_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    mismatches: list[dict] = []
    tp = fp = fn = 0

    for record in records:
        actual = classify(
            record["source_title"],
            record["candidate_title"],
            record.get("source_price", "100"),
            record.get("candidate_price", "100"),
        )
        expected = record["expected"]
        kind = record.get("kind", record.get("provenance", "unknown"))
        per_kind[kind][0] += 1
        per_kind[kind][1] += int(actual == expected)
        if expected == "A" and actual == "A":
            tp += 1
        elif actual == "A" and expected != "A":
            fp += 1
        elif expected == "A" and actual != "A":
            fn += 1
        if actual != expected:
            mismatches.append(
                {
                    "id": record.get("id", record.get("pair_id", "?")),
                    "kind": kind,
                    "expected": expected,
                    "actual": actual,
                    "note": record.get("note", ""),
                }
            )

    total = len(records)
    correct = sum(counts[1] for counts in per_kind.values())
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_kind": {kind: {"total": counts[0], "correct": counts[1]} for kind, counts in sorted(per_kind.items())},
        "mismatches": mismatches,
    }


def render_report(
    stats: dict,
    *,
    title: str = "同款匹配评测报告 v1（构造集）",
    dataset_note: str = "`evaluation/dataset_v1.jsonl`（构造法 + 真实标题种子，方法论见 build_dataset.py 文件头）",
) -> str:
    lines = [
        f"# {title}",
        "",
        f"数据集：{dataset_note}（{stats['total']} 对）",
        "",
        "## 总体指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 总体准确率 | {stats['accuracy']:.1%}（{stats['correct']}/{stats['total']}） |",
        f"| 同款检出 Precision | {stats['precision']:.1%}（TP={stats['tp']}, FP={stats['fp']}） |",
        f"| 同款检出 Recall | {stats['recall']:.1%}（FN={stats['fn']}） |",
        "",
        "## 分类目明细",
        "",
        "| kind | 总数 | 正确 | 准确率 |",
        "|---|---|---|---|",
    ]
    for kind, counts in stats["per_kind"].items():
        accuracy = counts["correct"] / counts["total"] if counts["total"] else 0.0
        lines.append(f"| {kind} | {counts['total']} | {counts['correct']} | {accuracy:.0%} |")
    lines += ["", f"## 错例清单（{len(stats['mismatches'])} 条）", ""]
    if stats["mismatches"]:
        lines.append("| id | kind | 期望 | 实际 | 说明 |")
        lines.append("|---|---|---|---|---|")
        for item in stats["mismatches"]:
            lines.append(f"| {item['id']} | {item['kind']} | {item['expected']} | {item['actual']} | {item['note']} |")
    else:
        lines.append("（无）")
    lines += ["", "> 说明：v1 为构造评测集，指标用于回归护栏与已知错误类型的强弱分析；", "> 真实人工标注集（v2，200-500 对）按 PRD §5.7 计划扩充。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    if hasattr(__import__("sys").stdout, "reconfigure"):
        try:
            __import__("sys").stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    if not DATASET.exists():
        print("数据集不存在，先运行：python -m evaluation.build_dataset")
        raise SystemExit(1)

    records = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    stats = evaluate(records)
    report = render_report(stats)
    RESULTS.write_text(report, encoding="utf-8")
    print(report)
    print(f"报告已写入 {RESULTS}")

    labeled = (
        [json.loads(line) for line in LABELED.read_text(encoding="utf-8").splitlines() if line.strip()]
        if LABELED.exists()
        else []
    )
    if labeled:
        real_stats = evaluate(labeled)
        real_report = render_report(
            real_stats,
            title=f"真实标注集评测（v2 · {real_stats['total']} 对人工盲标）",
            dataset_note="`evaluation/labeled_v2.jsonl`（人工盲标）",
        )
        RESULTS_V2.write_text(report + "\n\n---\n\n" + real_report, encoding="utf-8")
        print()
        print(real_report)
        print(f"合并报告已写入 {RESULTS_V2}")
    else:
        print("\n（暂无真实标注数据——运行 python -m evaluation.label_tool 标注后，此处将输出真实集指标）")


if __name__ == "__main__":
    main()
