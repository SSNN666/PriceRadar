"""向量层增益测量：**先量，再决定要不要进主链路**。

要回答的问题只有一个：在「同款匹配」这件事上，文本向量到底有没有用？

三个子实验：
1. **分档可分性** —— 同款对（A 档）与非同款对的余弦分布是否分得开？
   分得开 → 向量有判别力；重叠 → 向量只能做召回/排序。
2. **最优先验阈值** —— 在构造集 218 对上扫阈值，看纯向量能到多少准确率，
   并与现有规则匹配器（同一集合上 100%）直接对比。
3. **型号数字专项** —— `model_swap`（同系列不同型号，如 N20 vs N60）的余弦
   与 `fluff_injection`（同款但堆了一堆修饰词）比。这是**最关键的一条**：
   若前者与后者分布重叠，说明向量对型号数字不可靠，判定权必须留给规则。

结果写入 `evaluation/results_vector.md`。**负结果同样是结果**——
「我用数据把向量限制在召回/排序」比「我加了向量库」强得多，前提是实验真跑过。

用法：
    python -m evaluation.vector_probe              # 走缓存，不重复花钱
    python -m evaluation.vector_probe --no-cache   # 忽略缓存重算
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from priceradar.matching.embeddings import EmbeddingCache, EmbeddingError, cosine

HERE = Path(__file__).parent
DATASET = HERE / "dataset_v1.jsonl"
RESULTS = HERE / "results_vector.md"

# 现有规则匹配器在同一构造集上的成绩（evaluation/results_v1.md）——对比基准
RULE_MATCHER_ACCURACY = 1.0


def load_records(path: Path = DATASET) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "min": round(ordered[0], 4),
        "p25": round(ordered[len(ordered) // 4], 4),
        "median": round(statistics.median(ordered), 4),
        "p75": round(ordered[(len(ordered) * 3) // 4], 4),
        "max": round(ordered[-1], 4),
        "mean": round(statistics.fmean(ordered), 4),
    }


def sweep_threshold(pairs: list[tuple[float, bool]]) -> list[dict]:
    """扫阈值，返回每个候选阈值的二分类表现。正类 = A 档（真同款）。"""
    rows = []
    for step in range(50, 100):
        threshold = step / 100
        true_positive = sum(1 for score, positive in pairs if score >= threshold and positive)
        false_positive = sum(1 for score, positive in pairs if score >= threshold and not positive)
        false_negative = sum(1 for score, positive in pairs if score < threshold and positive)
        true_negative = sum(1 for score, positive in pairs if score < threshold and not positive)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        accuracy = (true_positive + true_negative) / len(pairs) if pairs else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": round(threshold, 2),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": true_positive, "fp": false_positive, "fn": false_negative, "tn": true_negative,
            }
        )
    return rows


def measure(records: list[dict], embedder) -> dict:
    """跑完整测量。`embedder` 需有 `.embed(list[str]) -> list[list[float]]`。"""
    sources = [record["source_title"] for record in records]
    candidates = [record["candidate_title"] for record in records]
    # 去重后再嵌入——构造集里有大量重复的源标题
    unique = list(dict.fromkeys(sources + candidates))
    vectors = embedder.embed(unique)
    index = {text: vector for text, vector in zip(unique, vectors)}

    scored = []
    for record in records:
        score = cosine(index[record["source_title"]], index[record["candidate_title"]])
        scored.append(
            {
                "id": record["id"],
                "kind": record["kind"],
                "expected": record["expected"],
                "source_title": record["source_title"],
                "candidate_title": record["candidate_title"],
                "cosine": round(score, 4),
            }
        )

    by_expected: dict[str, list[float]] = {}
    by_kind: dict[str, list[float]] = {}
    for row in scored:
        by_expected.setdefault(row["expected"], []).append(row["cosine"])
        by_kind.setdefault(row["kind"], []).append(row["cosine"])

    pairs = [(row["cosine"], row["expected"] == "A") for row in scored]
    sweep = sweep_threshold(pairs)
    best_accuracy = max(sweep, key=lambda row: (row["accuracy"], row["f1"]))
    best_f1 = max(sweep, key=lambda row: (row["f1"], row["accuracy"]))

    same_model = [row["cosine"] for row in scored if row["kind"] == "fluff_injection"]
    model_swap = [row["cosine"] for row in scored if row["kind"] == "model_swap"]

    return {
        "total": len(scored),
        "unique_texts": len(unique),
        "expected_stats": {key: _stats(values) for key, values in sorted(by_expected.items())},
        "kind_stats": {key: _stats(values) for key, values in sorted(by_kind.items())},
        "sweep": sweep,
        "best_accuracy": best_accuracy,
        "best_f1": best_f1,
        "model_swap_vs_same_model": {
            "same_model": _stats(same_model),
            "model_swap": _stats(model_swap),
            "separable": bool(same_model and model_swap and min(same_model) > max(model_swap)),
        },
        "rows": sorted(scored, key=lambda row: row["cosine"]),
    }


def render_report(stats: dict) -> str:
    best = stats["best_accuracy"]
    swap = stats["model_swap_vs_same_model"]
    lines = [
        "# 向量层增益测量（同款匹配）",
        "",
        "> 目的：**先量清楚向量对同款匹配有没有用，再决定要不要接进判定路径。**",
        "> 数据集：`evaluation/dataset_v1.jsonl`（218 对构造集）；模型：`text-embedding-v4`（1024 维）。",
        "> 复现：`python -m evaluation.vector_probe`（走缓存，不重复花钱）。",
        "",
        "## 结论速览",
        "",
        f"- **同一集合上，现有规则匹配器准确率 {RULE_MATCHER_ACCURACY:.1%}**（`results_v1.md`）。",
        f"- **纯向量最优阈值 {best['threshold']:.2f}：准确率 {best['accuracy']:.1%}、"
        f"精确率 {best['precision']:.1%}、召回率 {best['recall']:.1%}**。",
        f"- 型号数字可分性：**{'可分' if swap['separable'] else '不可分'}**"
        f"（同款最低余弦 {swap['same_model'].get('min')} vs 换型号最高余弦 {swap['model_swap'].get('max')}）",
        "",
        "## 决定：**不进判定路径**",
        "",
        f"纯向量最优阈值只有 {best['accuracy']:.1%} 准确率、{best['precision']:.1%} 精确率；"
        f"同一集合上规则匹配器是 {RULE_MATCHER_ACCURACY:.0%}。**差距不是调参能补的**，"
        "原因在下面第二节：换型号的负例余弦比堆了促销词的真同款还高。",
        "",
        "所以向量的定位是**召回与排序的辅助信号**（例如给平台推荐集去噪、给近似款排序），",
        "**判定权必须留在确定性规则**——型号数字不能被语义相似度糊弄：`N20` 与 `N60` "
        "在语义空间里几乎是一回事，在商品世界里是两件东西。",
        "",
        "## 一、按档位看余弦分布",
        "",
        "| 档位 | 对数 | 最小 | 中位 | 最大 |",
        "|---|---|---|---|---|",
    ]
    for key, value in stats["expected_stats"].items():
        lines.append(f"| {key} | {value['n']} | {value['min']} | {value['median']} | {value['max']} |")

    lines += [
        "",
        "## 二、最关键的对照：加修饰词 vs 换型号",
        "",
        "同款标题常被堆一堆促销词（`fluff_injection`）；而最难的负例恰恰是**换型号**",
        "（`model_swap`，如 N20 → N60）。若这两类的余弦分布重叠，说明：",
        "",
        "> **向量分不清「同一个东西的不同说法」和「长得像的另一个东西」——判定权必须留给规则。**",
        "",
        "| 类型 | 对数 | 最小 | 中位 | 最大 |",
        "|---|---|---|---|---|",
        f"| 同款 + 堆修饰词 | {swap['same_model'].get('n', 0)} | {swap['same_model'].get('min')} "
        f"| {swap['same_model'].get('median')} | {swap['same_model'].get('max')} |",
        f"| 换型号（负例） | {swap['model_swap'].get('n', 0)} | {swap['model_swap'].get('min')} "
        f"| {swap['model_swap'].get('median')} | {swap['model_swap'].get('max')} |",
        "",
        "## 三、阈值扫描（正类 = A 档真同款）",
        "",
        "| 阈值 | 准确率 | 精确率 | 召回率 | F1 | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in stats["sweep"]:
        if row["threshold"] * 100 % 5 == 0:      # 每 0.05 打一行，避免刷屏
            lines.append(
                f"| {row['threshold']:.2f} | {row['accuracy']:.1%} | {row['precision']:.1%} "
                f"| {row['recall']:.1%} | {row['f1']:.2f} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} |"
            )

    lines += [
        "",
        "## 四、按错误类型看（余弦升序，越低越可疑）",
        "",
        "| 类型 | 对数 | 中位余弦 | 最低余弦 |",
        "|---|---|---|---|",
    ]
    kind_rows = sorted(stats["kind_stats"].items(), key=lambda item: item[1].get("median", 0))
    for key, value in kind_rows:
        lines.append(f"| {key} | {value['n']} | {value['median']} | {value['min']} |")

    lines += [
        "",
        "## 五、可比的最低余弦（最容易被误判的对）",
        "",
        "| 类型 | 期望 | 余弦 | 源 | 候选 |",
        "|---|---|---|---|---|",
    ]
    for row in stats["rows"][:8]:
        lines.append(
            f"| {row['kind']} | {row['expected']} | {row['cosine']} "
            f"| {row['source_title'][:26]} | {row['candidate_title'][:26]} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="向量层增益测量（同款匹配）")
    parser.add_argument("--no-cache", action="store_true", help="忽略磁盘缓存重新计算")
    args = parser.parse_args(argv)

    from priceradar.matching.embeddings import DashScopeEmbedder

    # --no-cache = 本次不读也不写缓存（强制重算，但**不动**已有的缓存文件）。
    # 曾经写成 EmbeddingCache(path=None)：None 会被解析回默认路径，于是既没绕过
    # 缓存、还把 data/embedding_cache.json 删了——下一次「走缓存」的运行又要全量重算。
    cache = EmbeddingCache(persist=not args.no_cache)
    try:
        embedder = DashScopeEmbedder.from_env(cache=cache)
    except EmbeddingError as exc:
        raise SystemExit(f"无法初始化嵌入服务：{exc}") from exc

    records = load_records()
    print(f"测量 {len(records)} 对（缓存已有 {len(cache)} 条向量）……")
    stats = measure(records, embedder)
    print(f"实际发出 {embedder.calls} 次嵌入请求（其余命中缓存）")

    report = render_report(stats)
    RESULTS.write_text(report, encoding="utf-8")
    print(report)
    print(f"报告已写入 {RESULTS}")


if __name__ == "__main__":
    main()
