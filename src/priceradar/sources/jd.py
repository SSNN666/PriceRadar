"""京东联盟数据源客户端（PRD §5.1）。

职责：签名、调用联盟开放接口、把商品响应映射为领域模型 Offer（含 L1 信号）。
凭证：JD_UNION_APP_KEY / JD_UNION_APP_SECRET（必填）、JD_UNION_AUTH_KEY（可选 access_token）。

实测要点（2026-10-05，详见 docs/accounts-onboarding.md）：
- 签名与拼多多同算法（sources/signing.py）；`timestamp` 必须字符串格式。
- `jd.union.open.goods.query`（关键词搜索）需单独申请权限，未开通返回 403；
  当前账号可用 `jd.union.open.goods.jingfen.query`（京粉精选频道池，**无需 token**），
  响应自带好评率 / 评论数 / 30 天销量等 L1 信号字段。
- access_token 有过期问题（过期报 code 19），jingfen 路径不受影响。
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable, Sequence

from priceradar.config import load_env
from priceradar.domain.models import (
    Coupon,
    CouponType,
    Offer,
    Platform,
    ShopKind,
    Signals,
)
from priceradar.sources.retry import call_with_retry
from priceradar.sources.signing import md5_sign

GATEWAY = "https://api.jd.com/routerjson"

_OWNER_TO_SHOP_KIND = {"g": ShopKind.SELF_RUN, "p": ShopKind.POP}


def _jd_shop_kind(owner: object, shop_name: str | None) -> ShopKind | None:
    """店铺类型：自营最硬（owner=g），其次按店名识别旗舰店，其余按 owner 兜底。

    实测 owner 只有 g/p 两种，官方旗舰店在联盟数据里落到 p（和普通 POP 混在
    一起）——只靠 owner 分不出旗舰。店名由平台给出（"XX旗舰店"），不是我们猜的。
    """
    kind = _OWNER_TO_SHOP_KIND.get(str(owner or ""))
    if kind is ShopKind.SELF_RUN:
        return kind
    if shop_name and "旗舰店" in shop_name:
        return ShopKind.FLAGSHIP
    return kind


class JdApiError(RuntimeError):
    """京东 API 返回错误时抛出（含外层 error_response 与内层业务码）。"""

    def __init__(self, message: str, *, code: int | str | None = None) -> None:
        super().__init__(message)
        self.code = code


def _fen(value) -> Decimal:
    """金额/门槛字段（float 或 str）-> Decimal 到分。"""
    if value in (None, ""):
        return Decimal("0")
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _ms_to_date(value) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000).date()
    except (ValueError, OSError, OverflowError):
        return None


def jingfen_item_to_offer(item: dict) -> Offer:
    """把京粉精选商品（goods.jingfen.query 响应项）映射为 Offer。"""
    item_id = str(item.get("itemId") or "")
    coupons: list[Coupon] = []
    for index, raw in enumerate((item.get("couponInfo") or {}).get("couponList") or []):
        if not raw.get("discount"):
            continue
        coupons.append(
            Coupon(
                coupon_id=f"jd-{item_id}-{index}",
                # platformType: 1=平台券，其余按店铺券处理（近似映射，实测校准）
                type=CouponType.PLATFORM if raw.get("platformType") == 1 else CouponType.SHOP,
                face_value=_fen(raw.get("discount")),
                threshold=_fen(raw.get("quota")),
                scope_note="最优券" if raw.get("isBest") == 1 else "",
                valid_until=_ms_to_date(raw.get("useEndTime")),
                source="api",
            )
        )
    signals = Signals(
        good_comments_share=item.get("goodCommentsShare"),
        comments=item.get("comments"),
        in_order_count_30d=item.get("inOrderCount30Days"),
    )
    raw_brand = str(item.get("brandName") or "").strip()
    brand = raw_brand if raw_brand and raw_brand != "无品牌" and not raw_brand.isdigit() else None
    shop_name = (item.get("shopInfo") or {}).get("shopName") or None
    return Offer(
        platform=Platform.JD,
        sku_id=item_id,
        title=str(item.get("skuName") or ""),
        list_price=_fen((item.get("priceInfo") or {}).get("price")),
        url=_normalize_url(item.get("materialUrl")),
        image_url=_first_image_url(item),
        shop_name=shop_name,
        brand=brand,
        shop_kind=_jd_shop_kind(item.get("owner"), shop_name),
        coupons=coupons,
        signals=signals,
        source_extra={
            "item_id": item_id,
            "material_url": str(item.get("materialUrl") or ""),
            "spuid": str(item.get("spuid") or ""),
            "lowest_coupon_price": str((item.get("priceInfo") or {}).get("lowestCouponPrice") or ""),
        },
    )


def _normalize_url(raw) -> str | None:
    """materialUrl 常为无协议形式（jingfen.jd.com/...），统一补 https。"""
    text = str(raw or "").strip()
    if not text:
        return None
    if text.startswith(("http://", "https://")):
        return text
    return f"https://{text}"


def _first_image_url(item: dict) -> str | None:
    image_list = (item.get("imageInfo") or {}).get("imageList") or []
    for image in image_list:
        candidate = str((image or {}).get("url") or "")
        if candidate:
            return candidate
    white = str((item.get("imageInfo") or {}).get("whiteImage") or "")
    return white or None


def _to_int(value) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _promo_goods_to_offer(item: dict) -> Offer:
    """jd.union.open.goods.promotiongoodsinfo.query 响应项 -> Offer。

    注意：该接口（按官方文档）不返回优惠券信息——仅京东价/佣金/自营标识/30 天引单；
    到手价所需的券需另行获取（如 jingfen 通道或转链接口）。
    """
    is_self = str(item.get("isJdSale") or "") == "1"
    return Offer(
        platform=Platform.JD,
        sku_id=str(item.get("skuId") or ""),
        title=str(item.get("goodsName") or ""),
        list_price=_fen(item.get("unitPrice")),
        shop_kind=ShopKind.SELF_RUN if is_self else ShopKind.POP,
        signals=Signals(in_order_count_30d=_to_int(item.get("inOrderCount"))),
        source_extra={
            "material_url": str(item.get("materialUrl") or ""),
            "img_url": str(item.get("imgUrl") or ""),
            "shop_id": str(item.get("shopId") or ""),
            "cid_name": str(item.get("cidName") or ""),
            "commission_wl": str(item.get("commisionRatioWl") or ""),
        },
    )


def _post_gateway(payload: dict[str, str]) -> dict:
    request = urllib.request.Request(
        GATEWAY,
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


class JdClient:
    """京东联盟客户端。transport 可注入（测试离线化），默认走真实网关。"""

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        access_token: str | None = None,
        transport: Callable[[dict[str, str]], dict] | None = None,
    ) -> None:
        self.app_key = app_key
        self.app_secret = app_secret
        self.access_token = access_token
        self._transport = transport or _post_gateway

    @classmethod
    def from_env(cls) -> "JdClient":
        env = load_env("JD_UNION_APP_KEY", "JD_UNION_APP_SECRET", "JD_UNION_AUTH_KEY")
        if not env.get("JD_UNION_APP_KEY") or not env.get("JD_UNION_APP_SECRET"):
            raise JdApiError("未找到 JD_UNION_APP_KEY / JD_UNION_APP_SECRET（环境变量或 .env）")
        return cls(env["JD_UNION_APP_KEY"], env["JD_UNION_APP_SECRET"], access_token=env.get("JD_UNION_AUTH_KEY") or None)

    def call(self, method: str, business: dict) -> dict:
        params: dict[str, str] = {
            "app_key": self.app_key,
            "method": method,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "format": "json",
            "v": "1.0",
            "sign_method": "md5",
            "360buy_param_json": json.dumps(business, ensure_ascii=False),
        }
        if self.access_token:
            params["access_token"] = self.access_token
        params["sign"] = md5_sign(params, self.app_secret)

        data = call_with_retry(lambda: self._transport(params))
        if "error_response" in data:
            error = data["error_response"]
            raise JdApiError(
                error.get("zh_desc") or error.get("en_desc") or "京东 API 错误",
                code=error.get("code"),
            )
        for key, value in data.items():
            if key.endswith("_responce") or key.endswith("_response"):
                raw = value.get("result") or value.get("queryResult")
                inner = json.loads(raw) if isinstance(raw, str) else (raw or {})
                code = inner.get("code")
                if code is not None and code != 200:
                    raise JdApiError(inner.get("message") or "京东 API 错误", code=code)
                return inner
        raise JdApiError(f"无法解析京东响应：{json.dumps(data, ensure_ascii=False)[:200]}")

    def jingfen_goods(self, elite_id: int = 1, page_index: int = 1, page_size: int = 10) -> list[Offer]:
        """京粉精选频道商品（无需 access_token）。

        elite_id 为频道 ID（如 1=好券商品），频道清单待整理（TODO）。
        """
        inner = self.call(
            "jd.union.open.goods.jingfen.query",
            {"goodsReq": {"eliteId": elite_id, "pageIndex": page_index, "pageSize": page_size}},
        )
        return [jingfen_item_to_offer(item) for item in (inner.get("data") or [])]

    def goods_by_sku(self, sku_ids: Sequence[str | int]) -> list[Offer]:
        """通过 skuId 查询推广商品信息（官方接口：promotiongoodsinfo.query）。

        参数：skuIds 为逗号分隔字符串（1-100 个），由本方法负责拼接；
        不在推广中（无佣金）的 SKU 会被服务端静默省略。
        """
        ids = [str(s).strip() for s in sku_ids if str(s).strip()]
        if not ids:
            raise ValueError("sku_ids 不能为空")
        if len(ids) > 100:
            raise ValueError("skuIds 最多 100 个")
        inner = self.call(
            "jd.union.open.goods.promotiongoodsinfo.query",
            {"skuIds": ",".join(ids)},
        )
        raw = (inner.get("data") or {}).get("promotionGoodsResp") or []
        if isinstance(raw, dict):
            raw = [raw]
        return [_promo_goods_to_offer(item) for item in raw]
