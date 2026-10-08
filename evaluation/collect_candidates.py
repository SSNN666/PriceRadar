"""真实候选收集（真实标注集 v2 的第一步）。

对每个种子词：拼多多检索 + 京粉频道池 → 组装「源商品 × 候选商品」真实配对（不含标签），
写 evaluation/candidates_v2.jsonl；标注用 evaluation/label_tool.py。

用法：
    python -m evaluation.collect_candidates [--per-seed 12]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from priceradar.agent.tools import search_jd_channel, search_pdd_goods
from priceradar.domain.models import Offer
from priceradar.sources.jd import JdApiError

OUTPUT = Path(__file__).parent / "candidates_v2.jsonl"

DEFAULT_SEEDS = [
    "QCY N20 降噪耳机",
    "索尼 WH-1000XM6 头戴式降噪耳机",
    "AirPods Pro 保护套",
    "罗技 GPW 无线鼠标",
    "小米充电宝 20000",
]


@dataclass
class CandidatePair:
    pair_id: str
    seed: str
    source_title: str
    source_platform: str
    candidate_title: str
    candidate_platform: str


def _pair_id(seed: str, candidate_title: str) -> str:
    raw = f"{seed}||{candidate_title}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:10]


def build_pairs(seed: str, pdd_offers: list[Offer], jd_offers: list[Offer], *, per_seed: int = 12) -> list[CandidatePair]:
    """源 = 拼多多第一条；候选 = 其余拼多多 + 京粉池（按标题去重，两平台均衡采样）。

    注意：不携带匹配器的判定结果——标注必须盲标（避免确认偏差）。
    """
    if not pdd_offers:
        return []
    source = pdd_offers[0]
    seen_titles = {source.title}
    pool: list[Offer] = []
    for offer in list(pdd_offers[1:]) + list(jd_offers):
        if offer.title in seen_titles:
            continue
        seen_titles.add(offer.title)
        pool.append(offer)

    pdd_half = [offer for offer in pool if offer.platform.value == "pdd"][: per_seed // 2]
    jd_half = [offer for offer in pool if offer.platform.value == "jd"][: per_seed - len(pdd_half)]
    chosen = pdd_half + jd_half

    return [
        CandidatePair(
            pair_id=_pair_id(seed, offer.title),
            seed=seed,
            source_title=source.title,
            source_platform=source.platform.value,
            candidate_title=offer.title,
            candidate_platform=offer.platform.value,
        )
        for offer in chosen
    ]


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="真实候选收集（标注集 v2 第一步）")
    parser.add_argument("--per-seed", type=int, default=12, help="每个种子的候选数（默认 12）")
    parser.add_argument("--seeds", nargs="*", default=None, help="临时种子词（缺省用内置清单）")
    args = parser.parse_args(argv)

    seeds = args.seeds or DEFAULT_SEEDS
    jd_pool: list[Offer] = []
    try:
        jd_pool = search_jd_channel(24, 50)
    except (JdApiError, OSError) as exc:
        print(f"△ 京东池获取失败（{exc}），本轮仅收集拼多多候选")

    pairs: list[CandidatePair] = []
    for seed in seeds:
        try:
            pdd_offers = search_pdd_goods(seed, 30)
        except Exception as exc:
            print(f"△ 「{seed}」拼多多检索失败：{exc}")
            continue
        seed_pairs = build_pairs(seed, pdd_offers, jd_pool, per_seed=args.per_seed)
        if seed_pairs:
            print(f"  - 「{seed}」：源 1 + 候选 {len(seed_pairs)}")
        else:
            print(f"  - 「{seed}」：拼多多无检索结果，跳过")
        pairs.extend(seed_pairs)

    OUTPUT.write_text(
        "\n".join(json.dumps(asdict(pair), ensure_ascii=False) for pair in pairs) + ("\n" if pairs else ""),
        encoding="utf-8",
    )
    print(f"\n共 {len(pairs)} 对候选 → {OUTPUT}")
    print("下一步：python -m evaluation.label_tool  （交互标注，支持随时中断续标）")


if __name__ == "__main__":
    main()
