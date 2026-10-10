"""拼多多（多多进宝）数据源客户端（PRD §5.1）。

职责：签名、调用 DDK（pdd.ddk.*）接口、把商品响应映射为领域模型 Offer。
凭证：PDD_CLIENT_ID / PDD_CLIENT_SECRET / PDD_PID（环境变量 > 项目根 .env）。

实测要点（2026-10-05，详见 docs/accounts-onboarding.md）：
- 搜索接口要求 pid 已完成授权备案，未备案报 50001 / sub_code 60001；
  备案状态用 pdd.ddk.member.authority.query 查询（bind=1 为已备案）。
- 推荐接口 pdd.ddk.goods.recommend.get 无需备案。
- 金额字段单位为"分"（int），映射时 ÷100 转 Decimal。
- 签名：MD5(secret + 按 key 排序的 k1v1k2v2... + secret)，大写。
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import date
from decimal import Decimal
from typing import Callable

from priceradar.config import load_env
from priceradar.domain.models import Coupon, CouponType, Offer, Platform, ShopKind, Signals
from priceradar.sources.retry import call_with_retry
from priceradar.sources.signing import md5_sign

GATEWAY = "https://gw-api.pinduoduo.com/api/router"
CENTS = Decimal("100")


class PddApiError(RuntimeError):
    """拼多多 API 返回 error_response 时抛出。"""

    def __init__(self, message: str, *, code: int | str | None = None, sub_code: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.sub_code = sub_code


def sign_params(params: dict[str, str], secret: str) -> str:
    """拼多多签名（与京东联盟同算法，见 sources/signing.py）。"""
    return md5_sign(params, secret)


def _post_gateway(payload: dict[str, str]) -> dict:
    request = urllib.request.Request(
        GATEWAY,
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


_SALES_WAN = re.compile(r"^([0-9.]+)\s*万")


def parse_sales_tip(text: str | None) -> int | None:
    """解析拼多多模糊销量：'6861' → 6861；'1.2万+' → 12000（万为下界估算）。

    仅用于展示/筛选排序；页面展示仍保留原文（诚实口径）。
    """
    if not text:
        return None
    cleaned = str(text).strip().replace("+", "").replace(",", "")
    if "万" in cleaned:
        match = _SALES_WAN.match(cleaned)
        if not match:
            return None
        try:
            return int(float(match.group(1)) * 10000)
        except ValueError:  # 平台文本异常（如 "..万"）→ 不猜，返回 None
            return None
    return int(cleaned) if cleaned.isdigit() else None


_MERCHANT_TYPES = {
    1: ShopKind.PERSONAL,
    2: ShopKind.ENTERPRISE,
    3: ShopKind.FLAGSHIP,
    4: ShopKind.EXCLUSIVE,
    5: ShopKind.FRANCHISE,
    6: ShopKind.POP,
}


def _merchant_type_to_shop_kind(value: object) -> ShopKind | None:
    """merchant_type → ShopKind。

    ⚠️ 类型不统一：`goods.search` 返回 **int**，`goods.recommend.get` 返回 **string**，
    故统一走 int() 转换；转不动就返回 None（不猜）。
    """
    try:
        return _MERCHANT_TYPES.get(int(str(value)))
    except (TypeError, ValueError):
        return None


def _seconds_to_date(value: object) -> date | None:
    """拼多多时间戳为**秒级**（京东为毫秒，不可混用）；0 / None / 非法值 → None。"""
    try:
        stamp = int(value or 0)
    except (TypeError, ValueError):
        return None
    return date.fromtimestamp(stamp) if stamp > 0 else None


def _category_ids(item: dict) -> list[str]:
    """品类链（粗→细）。实测 `cat_ids` 100% 覆盖（83/83 样本，如 [5752, 5826, 19146]）。

    只认列表——字符串会被逐字符迭代成 ['5','7',...]，宁可不要也不给脏数据
    （与 `unified_tags` 同一个坑）。
    """
    raw = item.get("cat_ids")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(value) for value in raw if value not in (None, "")]


def goods_to_offer(item: dict) -> Offer:
    """把拼多多商品响应（搜索/推荐共用的商品结构）映射为 Offer。"""
    goods_id = str(item.get("goods_id") or "")
    coupon_amount = int(item.get("coupon_discount") or 0)
    coupons: list[Coupon] = []
    if coupon_amount > 0:
        coupons.append(
            Coupon(
                coupon_id=f"pdd-{goods_id}-coupon",
                # 多多进宝返回的多为店铺/商品券；店铺折扣券在 mall_coupon_*（比率券，暂未映射）
                # ⚠️ 不要映射 extra_coupon_amount：实测恒等于 coupon_discount（同一张券的重复字段），
                #    相加会双倍计优惠、到手价虚低。详见 docs/pdd-api-findings.md §8
                type=CouponType.SHOP,
                face_value=Decimal(coupon_amount) / CENTS,
                threshold=Decimal(int(item.get("coupon_min_order_amount") or 0)) / CENTS,
                scope_note="拼多多商品券",
                source="api",
            )
        )
    # 店铺收藏券：独立的券种字段（实测约 5% 商品有，与商品券互斥出现，故无需担心叠加）。
    # 需收藏店铺才能领，但**无资格门槛**（人人可领、有剩余量），因此计入到手价——
    # 不计入会让展示价高于用户实际能拿到的价；代价是必须在标签里写明条件。
    collect_discount = int(item.get("clt_cpn_discount") or 0)
    if collect_discount > 0:
        coupons.append(
            Coupon(
                coupon_id=f"pdd-{goods_id}-collect",
                type=CouponType.COLLECT,
                face_value=Decimal(collect_discount) / CENTS,
                threshold=Decimal(int(item.get("clt_cpn_min_amt") or 0)) / CENTS,
                scope_note="需收藏店铺后领取（无资格门槛）",
                valid_until=_seconds_to_date(item.get("clt_cpn_end_time")),
                source="api",
            )
        )
    sales_tip = str(item.get("sales_tip") or "") or None
    raw_brand = str(item.get("brand_name") or "").strip()
    raw_tags = item.get("unified_tags") or []
    # 只认列表：字符串会被逐字符迭代成 ['正','品','险']，宁可不要也不给脏数据
    tags = [str(tag) for tag in raw_tags if tag] if isinstance(raw_tags, (list, tuple)) else []
    return Offer(
        platform=Platform.PDD,
        sku_id=goods_id,
        title=str(item.get("goods_name") or ""),
        list_price=Decimal(int(item.get("min_group_price") or 0)) / CENTS,
        url=f"https://mobile.yangkeduo.com/goods.html?goods_id={goods_id}" if goods_id else None,
        image_url=str(item.get("goods_thumbnail_url") or item.get("goods_image_url") or "") or None,
        shop_name=item.get("mall_name") or None,
        # ⚠️ 拼多多**没有店铺标识符**：mall_name 实测恒为空（快照库 874 条填充率 13%），
        # 而 mall_id 连键都不存在（2026-10-10 实测 83 条搜索样本，0 命中）——
        # docs/pdd-api-findings.md 曾写"mall_id 可用"，那是错的（已订正）。
        # 结论：拼多多的店铺券**无法绑定到店铺**，只能诚实降级为不自动套用。
        shop_id=None,
        category_ids=_category_ids(item),
        shop_kind=_merchant_type_to_shop_kind(item.get("merchant_type")),
        brand=raw_brand or None,
        tags=tags,
        coupons=coupons,
        signals=Signals(
            sales_tip=sales_tip,
            sales_estimate=parse_sales_tip(sales_tip),
            # 实测：描述分/物流分/服务分三维度恒同（242 样本 0 分歧），取描述分即可
            shop_score_tier=str(item.get("desc_txt") or "").strip() or None,
        ),
        source_extra={"goods_sign": str(item.get("goods_sign") or "")},
    )


class PddClient:
    """多多进宝客户端。transport 可注入（测试离线化），默认走真实网关。"""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        pid: str | None = None,
        transport: Callable[[dict[str, str]], dict] | None = None,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.pid = pid
        self._transport = transport or _post_gateway

    @classmethod
    def from_env(cls) -> "PddClient":
        env = load_env("PDD_CLIENT_ID", "PDD_CLIENT_SECRET", "PDD_PID")
        if not env.get("PDD_CLIENT_ID") or not env.get("PDD_CLIENT_SECRET"):
            raise PddApiError("未找到 PDD_CLIENT_ID / PDD_CLIENT_SECRET（环境变量或 .env）")
        return cls(env["PDD_CLIENT_ID"], env["PDD_CLIENT_SECRET"], pid=env.get("PDD_PID") or None)

    def call(self, api_type: str, **params: str) -> dict:
        payload: dict[str, str] = {
            "type": api_type,
            "client_id": self.client_id,
            "timestamp": str(int(time.time())),
            "data_type": "JSON",
        }
        payload.update({key: str(value) for key, value in params.items()})
        payload["sign"] = sign_params(payload, self.client_secret)
        data = call_with_retry(lambda: self._transport(payload))
        if "error_response" in data:
            error = data["error_response"]
            raise PddApiError(
                error.get("sub_msg") or error.get("error_msg") or "拼多多 API 错误",
                code=error.get("error_code"),
                sub_code=error.get("sub_code"),
            )
        return data

    def recommend_similar(self, goods_sign: str, limit: int = 20) -> list[Offer]:
        """平台原生「相似商品推荐」（`channel_type=3`）：给一个参考商品，返回相似商品。

        这是**唯一**能找到"和这个商品像的其它商品"的官方能力——不需要图搜 API、
        不需要爬虫，把"同款发现"从关键词搜索提升为以商品为锚点的扩展。

        ⚠️ 两个坑：
        1. `channel_type` **默认值是 5（实时热销榜）**——那是**全局榜单**，与参考商品
           毫无关系（实测相关度 0%）。要"找相似"必须显式传 3。
        2. `goods_sign_list` **仅取数组第一位**（平台约定）。

        实测精度随品类差异极大：保温杯 100% / 降噪耳机 ~47% / 连衣裙 ~33%
        ——功能明确的标品精度高，款式驱动的服饰低。**它只能当候选来源，
        不能替代同款判定**（返回里混有无关商品，须再过匹配管线）。
        """
        data = self.call(
            "pdd.ddk.goods.recommend.get",
            channel_type=3,
            goods_sign_list=json.dumps([goods_sign]),
            limit=limit,
            offset=0,
        )
        items = (data.get("goods_basic_detail_response") or {}).get("list") or []
        return [goods_to_offer(item) for item in items]

    def search_goods(
        self,
        keyword: str,
        page: int = 1,
        page_size: int = 10,
        with_coupon: bool | None = None,
    ) -> list[Offer]:
        """关键词搜索商品。要求 pid 已授权备案（未备案报 PddApiError 50001/60001）。

        with_coupon=True 时只返回带优惠券的商品（默认 None = 全部，含无券商品）。
        """
        params = {"keyword": keyword, "page": str(page), "page_size": str(page_size)}
        if with_coupon is not None:
            params["with_coupon"] = "true" if with_coupon else "false"
        if self.pid:
            params["pid"] = self.pid
        data = self.call("pdd.ddk.goods.search", **params)
        goods = (data.get("goods_search_response") or {}).get("goods_list") or []
        return [goods_to_offer(item) for item in goods]

    def recommend_goods(self, limit: int = 5) -> list[Offer]:
        """推荐商品（无需备案；用于连通性验证与冷启动候选补充）。"""
        data = self.call("pdd.ddk.goods.recommend.get", offset="0", limit=str(limit))
        goods = (data.get("goods_basic_detail_response") or {}).get("list") or []
        return [goods_to_offer(item) for item in goods]
