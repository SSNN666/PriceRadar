"""交互式盲标工具（真实标注集 v2）。

用法：
    python -m evaluation.label_tool [--limit 20]

逐对展示 源商品 × 候选商品，按键判定：
    y = 同款（同型号同规格）        b = 近似款（同系列不同规格/代际）
    n = 非同款                     s = 跳过      q = 保存并退出
标签增量写入 evaluation/labeled_v2.jsonl（可随时中断，重跑自动续标）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

CANDIDATES = Path(__file__).parent / "candidates_v2.jsonl"
LABELED = Path(__file__).parent / "labeled_v2.jsonl"

# 标签与评测期望对齐（run_eval 的 expected 取值：A / B / excluded）
_LABEL_MAP = {"y": "A", "b": "B", "n": "excluded"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_label(raw: str) -> str | None:
    """按键 → 期望标签；'s'（跳过）/ 'q'（退出）/ 非法输入 → None。"""
    return _LABEL_MAP.get(raw.strip().lower())


def run_labeling(
    candidates: list[dict],
    labeled_path: Path = LABELED,
    *,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
    limit: int | None = None,
) -> int:
    """逐对标注；已标注的 pair_id 自动跳过（可续标）。返回本次标注对数。"""
    done = {row["pair_id"] for row in load_jsonl(labeled_path)}
    todo = [pair for pair in candidates if pair["pair_id"] not in done]
    if limit is not None:
        todo = todo[:limit]

    count = 0
    with open(labeled_path, "a", encoding="utf-8") as handle:
        for index, pair in enumerate(todo, 1):
            output_func(f"\n[{index}/{len(todo)}] 种子：{pair['seed']}")
            output_func(f"  源  （{pair['source_platform']}）：{pair['source_title'][:64]}")
            output_func(f"  候选（{pair['candidate_platform']}）：{pair['candidate_title'][:64]}")
            while True:
                raw = input_func("  判定 [y=同款 / b=近似款(同系列不同规格) / n=非同款 / s=跳过 / q=退出]：")
                key = raw.strip().lower()
                if key == "q":
                    return count
                if key == "s":
                    break
                label = parse_label(key)
                if label is None:
                    output_func("  请输入 y / b / n / s / q")
                    continue
                record = {**pair, "expected": label, "provenance": "real_labeled"}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                count += 1
                break
    return count


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="交互式盲标（真实标注集 v2）")
    parser.add_argument("--limit", type=int, default=None, help="本次最多标注对数")
    args = parser.parse_args(argv)

    candidates = load_jsonl(CANDIDATES)
    if not candidates:
        print(f"未找到候选文件 {CANDIDATES}，请先运行：python -m evaluation.collect_candidates")
        sys.exit(1)

    already = len(load_jsonl(LABELED))
    print(f"候选 {len(candidates)} 对（已标注 {already} 对）；输入 q 可随时保存退出。")
    count = run_labeling(candidates, limit=args.limit)
    total = len(load_jsonl(LABELED))
    print(f"\n本次标注 {count} 对，累计 {total} 对。")
    print("查看真实集指标：python -m evaluation.run_eval")


if __name__ == "__main__":
    main()
