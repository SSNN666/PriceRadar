"""同款匹配的数据模型（PRD §5.7 四层消歧的输出契约）。"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from priceradar.domain.models import Offer


class MatchTier(str, Enum):
    """候选分档（PRD §5.7 分组呈现）。"""

    CONFIRMED = "A"    # 确认同款：直接进到手价比价
    SIMILAR = "B"      # 近似款：折叠展示，逐条标注差异
    SUSPICIOUS = "C"   # 可疑低价：参与比价但强制警示标签


class SpecProfile(BaseModel):
    """标题中提取的规格画像。"""

    models: set[str] = Field(default_factory=set, description="型号/代际 token（如 N20/XM6/1000XM6）")
    capacities: set[str] = Field(default_factory=set, description="容量（如 128G/1T）")
    sizes: set[str] = Field(default_factory=set, description="尺码（如 XL/42码）")
    colors: set[str] = Field(default_factory=set, description="颜色（单字，如 黑）")
    generations: set[str] = Field(default_factory=set, description="代际（如 4代）")
    brand: str | None = Field(default=None, description="品牌（canonical 键，如 sony/qcy）")
    is_accessory: bool = Field(default=False, description="是否为配件/周边类商品")


class MatchResult(BaseModel):
    tier: MatchTier
    candidate: Offer
    score: float
    reasons: list[str] = Field(default_factory=list)


class ExcludedCandidate(BaseModel):
    candidate: Offer
    reason: str


class MatchReport(BaseModel):
    """一次匹配运行的完整结果（含排除项与原因，透明可核对）。"""

    source_sku: str
    results: list[MatchResult] = Field(default_factory=list)
    excluded: list[ExcludedCandidate] = Field(default_factory=list)
