"""图 → 结构化提取：商品图识别（PRD §5.5）与券截图识别（PRD §5.6）。

共同边界（延续"钱不由 LLM 算"）：
- VLM 只做"看"：提取结构化字段；金额合并/同款判定仍由确定性代码完成
- 防幻觉优先：不确定字段填 null 并进 uncertain_fields，禁止猜测
- 券识别产物必须经用户确认后才会写入 user_coupons（§5.6 硬性要求）
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

from priceradar.domain.models import Coupon, CouponType
from priceradar.vision.vlm import QwenVlClient

# ---------- 商品图识别 ----------


class ProductAttrs(BaseModel):
    is_product: bool = Field(description="图中是否为商品（风景/人物等为 False）")
    category: str | None = None
    brand: str | None = None
    model: str | None = None
    specs: dict[str, str] = Field(default_factory=dict, description="容量/尺寸等规格")
    color: str | None = None
    keywords: list[str] = Field(default_factory=list, description="组装好的检索关键词（中文 2-4 个）")
    uncertain_fields: list[str] = Field(default_factory=list, description="模型自报不确定的字段名")
    confidence: float = Field(default=0.5, ge=0, le=1)


PRODUCT_PROMPT = (
    "你是商品信息提取器。识别图中的商品并输出 JSON，字段："
    "is_product(布尔)、category(品类)、brand(品牌)、model(型号)、specs(对象：容量/尺寸等)、"
    "color(颜色)、keywords(检索关键词数组：中文 2-4 个)、uncertain_fields(不确定的字段名数组)、confidence(0~1)。"
    "严格规则：① 只描述图中可见信息；② 品牌/型号仅在出现明确 logo 或文字时填写，"
    "不确定填 null 并加入 uncertain_fields——禁止猜测；③ 若图中不是商品（风景/人物等），is_product=false；"
    "④ 只输出 JSON，不要任何多余文本。"
)

# ---------- 券截图识别 ----------


class CouponFields(BaseModel):
    type: Literal["shop", "platform", "category", "member"]
    face_value: Decimal = Field(ge=0, description="面额（数字，不带单位）")
    threshold: Decimal = Field(default=Decimal("0"), ge=0, description="满减门槛（无门槛填 0）")
    scope: str = Field(default="", description="适用范围描述")
    valid_until: str | None = Field(default=None, description="有效期（YYYY-MM-DD）或 null")
    confidence: float = Field(default=0.5, ge=0, le=1)


class CouponExtraction(BaseModel):
    platform: Literal["jd", "pdd", "taobao"]
    coupons: list[CouponFields]


COUPON_PROMPT = (
    "你是优惠券信息提取器。识别图中的优惠券/结算页优惠信息并输出 JSON：{"
    'platform: "jd"|"pdd"|"taobao"（依图中标识判断）, '
    "coupons: [{type: shop|platform|category|member, face_value: 面额数字, "
    "threshold: 满减门槛数字（无门槛填 0）, scope: 适用范围描述, "
    'valid_until: "YYYY-MM-DD" 或 null, confidence: 0~1}]}。'
    "严格规则：① 只提取图中清晰可见的数字与文字——模糊或遮挡的数字不要猜，宁缺毋滥；"
    "② 金额只填数字不带单位；③ 只输出 JSON，不要任何多余文本。"
)

_COUPON_TYPE_MAP = {
    "shop": CouponType.SHOP,
    "platform": CouponType.PLATFORM,
    "category": CouponType.CATEGORY,
    "member": CouponType.MEMBER,
}


def extract_product(
    image_bytes: bytes,
    media_type: str = "image/jpeg",
    *,
    client: QwenVlClient | None = None,
) -> ProductAttrs:
    """商品图 → ProductAttrs（VLM 提取；调用方负责低置信度追问与用户确认）。"""
    client = client or QwenVlClient.from_env()
    result = client.complete_json(
        PRODUCT_PROMPT, "识别这张图片中的商品。", image_bytes, json_schema=ProductAttrs, media_type=media_type
    )
    return result  # type: ignore[return-value]


def extract_coupons(
    image_bytes: bytes,
    media_type: str = "image/jpeg",
    *,
    client: QwenVlClient | None = None,
) -> CouponExtraction:
    """券截图 → CouponExtraction（VLM 提取；写入前必须经用户确认，§5.6）。"""
    client = client or QwenVlClient.from_env()
    result = client.complete_json(
        COUPON_PROMPT, "识别这张图中的优惠券信息。", image_bytes, json_schema=CouponExtraction, media_type=media_type
    )
    return result  # type: ignore[return-value]


def to_offer_coupons(extraction: CouponExtraction) -> list[Coupon]:
    """券提取结果 → 领域模型 Coupon（供到手价引擎并入计算）。"""
    coupons: list[Coupon] = []
    for index, item in enumerate(extraction.coupons):
        valid_until: date | None = None
        if item.valid_until:
            try:
                valid_until = date.fromisoformat(item.valid_until)
            except ValueError:
                valid_until = None
        coupons.append(
            Coupon(
                coupon_id=f"shot-{extraction.platform}-{index}",
                type=_COUPON_TYPE_MAP[item.type],
                face_value=item.face_value,
                threshold=item.threshold,
                scope_note=item.scope,
                valid_until=valid_until,
                source="screenshot",
                confidence=item.confidence,
            )
        )
    return coupons
