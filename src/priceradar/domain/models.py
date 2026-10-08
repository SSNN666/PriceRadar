"""领域模型（对应 PRD §5.3 存储层）。

约定：
- 金额一律 Decimal（精确到分），不用 float
- 模型只承载数据与展示标签，不含业务计算（计算全部在 engine 层，纯函数）
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field

from priceradar.formatting import fmt_amount


class Platform(str, Enum):
    JD = "jd"
    PDD = "pdd"
    TAOBAO = "taobao"


class CouponType(str, Enum):
    SHOP = "shop"          # 店铺券
    PLATFORM = "platform"  # 平台券
    CATEGORY = "category"  # 品类券
    MEMBER = "member"      # 会员券（多来自截图/手动补录）
    COLLECT = "collect"    # 店铺收藏券：需收藏店铺后领取，但无资格门槛（人人可领）


class PromotionKind(str, Enum):
    STEP = "step"        # 阶梯满减：达标只减一次（同活动多档位用 group 标注）
    EVERY_N = "every_n"  # 每满减：每满 X 减 Y，可按倍数叠加


class ShopKind(str, Enum):
    """店铺资质类型（各平台口径不同，取并集建模；缺失即 None，不猜）。

    - 京东：`owner` 字段（g=自营 / p=POP）+ 店铺名兜底识别旗舰店
    - 拼多多：`merchant_type`（1-个人 / 2-企业 / 3-旗舰店 / 4-专卖店 / 5-专营店 / 6-普通店）
    """

    SELF_RUN = "self_run"      # 自营
    FLAGSHIP = "flagship"      # 旗舰店
    EXCLUSIVE = "exclusive"    # 专卖店（品牌授权）
    FRANCHISE = "franchise"    # 专营店（品类授权）
    ENTERPRISE = "enterprise"  # 企业店
    PERSONAL = "personal"      # 个人店
    POP = "pop"                # 普通第三方店


# 展示用标签。⚠️ 刻意不含 POP：普通店是"没有信号"，给它挂个徽标只是噪声——
# 徽标只用来突出正面资质（自营/旗舰/专卖/专营）与需要留意的档位（个人店）。
SHOP_KIND_LABELS = {
    ShopKind.SELF_RUN: "自营",
    ShopKind.FLAGSHIP: "旗舰店",
    ShopKind.EXCLUSIVE: "专卖店",
    ShopKind.FRANCHISE: "专营店",
    ShopKind.ENTERPRISE: "企业店",
    ShopKind.PERSONAL: "个人店",
}

# 资质店铺：有平台认证的品牌/品类授权（用于"仅资质店铺"筛选）
CERTIFIED_SHOP_KINDS = frozenset(
    {ShopKind.SELF_RUN, ShopKind.FLAGSHIP, ShopKind.EXCLUSIVE, ShopKind.FRANCHISE}
)


_COUPON_TYPE_LABELS = {
    CouponType.SHOP: "店铺券",
    CouponType.PLATFORM: "平台券",
    CouponType.CATEGORY: "品类券",
    CouponType.MEMBER: "会员券",
    # 条件直接写进标签：用户看到低价时必须知道要做什么才能拿到（诚实口径）
    CouponType.COLLECT: "店铺收藏券（需收藏店铺）",
}


class Coupon(BaseModel):
    coupon_id: str
    type: CouponType
    face_value: Decimal = Field(ge=0, description="优惠面额")
    threshold: Decimal = Field(default=Decimal("0"), ge=0, description="使用门槛（满 X 可用）")
    scope_note: str = Field(default="", description="适用范围描述（展示用）")
    exclusion_group: str | None = Field(default=None, description="互斥组：同组券只能使用一张")
    valid_until: date | None = None
    source: str = Field(default="api", description="来源：api / manual / screenshot")
    confidence: float = Field(default=1.0, ge=0, le=1, description="来源置信度（截图补录时 < 1）")

    def label(self) -> str:
        base = _COUPON_TYPE_LABELS[self.type]
        if self.threshold > 0:
            text = f"{base} 满{fmt_amount(self.threshold)}减{fmt_amount(self.face_value)}"
        else:
            text = f"{base} 立减{fmt_amount(self.face_value)}"
        if self.source in ("manual", "screenshot"):
            return f"[补录]{text}"  # 用户补录券在明细中显式区分（PRD §5.6）
        return text


class Promotion(BaseModel):
    promotion_id: str
    kind: PromotionKind
    threshold: Decimal = Field(gt=0, description="满 / 每满 X")
    reduction: Decimal = Field(ge=0, description="减 Y")
    group: str | None = Field(default=None, description="同活动的多档位编组：同组仅取最优一档")
    scope_note: str = ""
    valid_until: date | None = None

    def label(self) -> str:
        prefix = "每满" if self.kind is PromotionKind.EVERY_N else "满"
        return f"{prefix}{fmt_amount(self.threshold)}减{fmt_amount(self.reduction)}"


class Signals(BaseModel):
    """商品质量/热度信号（PRD §5.8 L1）：缺失即为 None，不猜。"""

    good_comments_share: float | None = Field(default=None, description="好评率（%）")
    comments: int | None = Field(default=None, description="评论数")
    in_order_count_30d: int | None = Field(default=None, description="近 30 天销量")
    sales_tip: str | None = Field(default=None, description="平台销量原文（如 1.2万+，诚实展示不转写）")
    sales_estimate: int | None = Field(default=None, description="销量下界估算（模糊值解析，供筛选用）")
    shop_score_tier: str | None = Field(
        default=None,
        description="店铺评分档位（拼多多 desc/lgst/serv 三档文字；实测三维度恒同，只取一个展示）",
    )


class Offer(BaseModel):
    """某平台上的一次报价：商品 + 标价 + 可用公开优惠。"""

    platform: Platform
    sku_id: str
    title: str
    list_price: Decimal = Field(gt=0, description="标价（到手价测算基准）")
    url: str | None = None
    image_url: str | None = Field(default=None, description="商品主图 URL（展示用）")
    shop_name: str | None = None
    shop_kind: ShopKind | None = None
    brand: str | None = Field(default=None, description="平台提供的结构化品牌（供匹配，缺失为 None）")
    tags: list[str] = Field(default_factory=list, description="平台保障标签（正品险/假一赔十等）")
    signals: Signals | None = Field(default=None, description="质量/热度信号（缺失为 None，不猜）")
    coupons: list[Coupon] = Field(default_factory=list)
    promotions: list[Promotion] = Field(default_factory=list)
    source_extra: dict[str, str] = Field(
        default_factory=dict,
        description="平台特定引用（如拼多多 goods_sign），供转链/详情等后续调用",
    )
