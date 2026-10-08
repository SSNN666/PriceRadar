"""同款匹配器：规格硬规则 + 相似度 sanity check + 分档（PRD §5.7 的确定性层）。

v1 顺序：
0. 配件过滤（源非配件时，配件候选排除；借鉴 best-price-mcp 的配件过滤实践）
1. 品牌冲突 → 排除（平台结构化品牌优先，标题识别兜底）
2. 型号冲突 / 候选缺型号 → 近似款 B（逐条标注差异）
3. 容量 / 代际冲突 → B
4. 规格一致 → 相似度 sanity check 后分 A/B；价格异常低 → C（可疑低价）
5. 源商品无任何规格锚点 → 只能凭标题相似度 → 上限 B（不假装精确）

价格离群检测（PRD 第 3 层 IQR/组内中位数）为 v2 升级项，当前用"源价相对比例"启发式。
"""

from __future__ import annotations

from decimal import Decimal
from typing import Iterable

from pydantic import BaseModel
from priceradar.domain.models import Offer
from priceradar.formatting import fmt_amount
from priceradar.matching.analysis import similarity, specs_for
from priceradar.matching.models import (
    ExcludedCandidate,
    MatchReport,
    MatchResult,
    MatchTier,
    SpecProfile,
)

_TIER_ORDER = {MatchTier.CONFIRMED: 0, MatchTier.SIMILAR: 1, MatchTier.SUSPICIOUS: 2}


class MatchPolicy(BaseModel):
    """匹配策略阈值（可配置、随评测集校准）。"""

    sim_min: float = 0.18        # 标题相似度 sanity 下限
    price_outlier_ratio: float = 0.5  # 候选价 ≤ 源价×该比例且规格一致 → 可疑低价


DEFAULT_POLICY = MatchPolicy()


def _join(values: set[str]) -> str:
    return "/".join(sorted(values))


def _match_single(source: Offer, source_specs: SpecProfile, candidate: Offer, policy: MatchPolicy) -> MatchResult | ExcludedCandidate:
    cand_specs = specs_for(candidate)

    # 层 0：配件过滤（两边都是配件时正常比较）
    if cand_specs.is_accessory and not source_specs.is_accessory:
        return ExcludedCandidate(candidate=candidate, reason="配件/周边类商品")

    # 层 1：品牌冲突
    if source_specs.brand and cand_specs.brand and source_specs.brand != cand_specs.brand:
        return ExcludedCandidate(candidate=candidate, reason=f"品牌不同（{source_specs.brand} vs {cand_specs.brand}）")

    score = similarity(source.title, candidate.title)
    specs_intersect = (
        bool(source_specs.models & cand_specs.models)
        or bool(source_specs.capacities & cand_specs.capacities)
        or bool(source_specs.generations & cand_specs.generations)
    )

    # 相关性地板：无规格交集且标题不相似 → 直接排除（避免无关商品落入"近似款"）
    if not specs_intersect and score < policy.sim_min:
        return ExcludedCandidate(candidate=candidate, reason="相似度过低")

    # 层 2：型号（源有型号时最硬）
    if source_specs.models:
        if not cand_specs.models:
            return MatchResult(tier=MatchTier.SIMILAR, candidate=candidate, score=score, reasons=["候选缺少型号信息，无法确认同款"])
        if not (source_specs.models & cand_specs.models):
            return MatchResult(tier=MatchTier.SIMILAR, candidate=candidate, score=score, reasons=[f"型号不同（{_join(source_specs.models)} vs {_join(cand_specs.models)}）"])

    # 层 2b：容量 / 代际冲突
    if source_specs.capacities and cand_specs.capacities and not (source_specs.capacities & cand_specs.capacities):
        return MatchResult(tier=MatchTier.SIMILAR, candidate=candidate, score=score, reasons=[f"容量不同（{_join(source_specs.capacities)} vs {_join(cand_specs.capacities)}）"])
    if source_specs.generations and cand_specs.generations and not (source_specs.generations & cand_specs.generations):
        return MatchResult(tier=MatchTier.SIMILAR, candidate=candidate, score=score, reasons=[f"代际不同（{_join(source_specs.generations)} vs {_join(cand_specs.generations)}）"])

    anchors = source_specs.models | source_specs.capacities | source_specs.generations
    if not anchors:
        # 源商品无规格锚点：只能凭标题相似度，上限 B（相似度下限已在上方把关）
        return MatchResult(tier=MatchTier.SIMILAR, candidate=candidate, score=score, reasons=["无可用规格佐证，仅标题相似"])

    # 层 3：可疑低价（规格一致但价格异常低）
    outlier_threshold = source.list_price * Decimal(str(policy.price_outlier_ratio))
    if candidate.list_price <= outlier_threshold:
        return MatchResult(
            tier=MatchTier.SUSPICIOUS,
            candidate=candidate,
            score=score,
            reasons=[f"价格异常低（￥{fmt_amount(candidate.list_price)} vs 参考 ￥{fmt_amount(source.list_price)}），疑似非正品/二手/瑕疵"],
        )

    if not specs_intersect:
        return MatchResult(tier=MatchTier.SIMILAR, candidate=candidate, score=score, reasons=["无法确认规格一致（候选规格信息不足）"])
    if score < policy.sim_min:
        return MatchResult(tier=MatchTier.SIMILAR, candidate=candidate, score=score, reasons=["有规格交集但标题差异较大，建议人工核对"])
    if source_specs.models and (source_specs.models & cand_specs.models):
        reason = f"型号一致（{_join(source_specs.models & cand_specs.models)}）"
    else:
        reason = f"规格一致（{_join(anchors)}）"
    return MatchResult(tier=MatchTier.CONFIRMED, candidate=candidate, score=score, reasons=[reason])


def match_offers(
    source: Offer,
    candidates: Iterable[Offer],
    policy: MatchPolicy = DEFAULT_POLICY,
) -> MatchReport:
    """对候选池执行同款匹配，输出 A/B/C 分档结果与排除项（确定性、可复现）。"""
    source_specs = specs_for(source)
    results: list[MatchResult] = []
    excluded: list[ExcludedCandidate] = []

    for candidate in candidates:
        if candidate.platform == source.platform and candidate.sku_id == source.sku_id:
            continue  # 跳过自身
        outcome = _match_single(source, source_specs, candidate, policy)
        if isinstance(outcome, MatchResult):
            results.append(outcome)
        else:
            excluded.append(outcome)

    results.sort(key=lambda item: (_TIER_ORDER[item.tier], -item.score, item.candidate.sku_id))
    return MatchReport(source_sku=source.sku_id, results=results, excluded=excluded)
