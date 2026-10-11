"""价格历史洞察（PRD §3 P1「价格与时机关」的确定性实现）。

数据源：自采快照库（data/snapshots.db）。全部结论只基于**实际观测天数**：
- 样本 < MIN_OBSERVED_DAYS 时明确"数据积累中"，不出时机结论（诚实原则，PRD §4）
- 假优惠识别（先涨后降）为确定性规则，阈值可配置
- 输出固定附"基于自观测数据（非平台全历史）"说明
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from priceradar.formatting import fmt_amount

MIN_OBSERVED_DAYS = 14      # 时机结论的最少观测天数（PRD 样本门槛）
FAKE_RISE_RATIO = 1.15      # 先涨：近期峰值 ≥ 前期基线 × 该比例
FAKE_DROP_RATIO = 0.90      # 后降：现价 ≤ 峰值 × 该比例
_RECENT_WINDOW = 7          # 近 7 天为"近期"，其前为"基线期"

@dataclass(frozen=True)
class PricePoint:
    """一天的观测点。

    **标价与到手价必须分开记**——只看到手价的话，「商家提价再打折」与
    「平台/店铺把优惠收回去了」长得一模一样，而这两件事对买家的含义相反：
    前者是套路该警惕，后者往往只是促销窗口结束、等下一轮就行。
    两者都让到手价曲线往上走，光看一条曲线分不出来。
    """

    day: str
    list_price: Decimal
    final_price: Decimal

    @property
    def markdown(self) -> Decimal:
        """优惠额（标价 − 到手价）。它缩水 = 券到期/收回，不等于商家提价。"""
        return self.list_price - self.final_price

    @property
    def pair(self) -> tuple[str, Decimal]:
        """兼容旧格式 `(日期, 到手价)`。"""
        return (self.day, self.final_price)


@dataclass
class TimingAdvice:
    observed_days: int
    status: str                      # "insufficient" | "ok"
    verdict: str
    percentile: float | None = None
    lowest: PricePoint | None = None
    note: str = ""


@dataclass
class FakeDiscountReport:
    status: str                      # "insufficient" | "clean" | "suspected" | "coupon_withdrawn"
    detail: str
    driver: str = ""                 # "list_price"（商家提价）| "coupon"（优惠缩水）


def load_history(connection: sqlite3.Connection, platform: str, sku_id: str) -> list[PricePoint]:
    """读取某商品的观测历史（按日期升序）。

    取标价也取到手价——`list_price` / `total_discount` 两列一直都写在库里，
    只是以前从没被读过，于是「谁在动」这件事无从判断。
    """
    rows = connection.execute(
        """
        SELECT captured_date, list_price, final_price FROM price_snapshots
        WHERE platform = ? AND sku_id = ? ORDER BY captured_date
        """,
        (platform, sku_id),
    ).fetchall()
    return [PricePoint(row[0], Decimal(row[1]), Decimal(row[2])) for row in rows]


def advise_timing(points: list[PricePoint], current: Decimal | None = None, min_days: int = MIN_OBSERVED_DAYS) -> TimingAdvice:
    """时机建议：基于观测期内的分位数（低位可入手 / 中位按需 / 高位观望）。"""
    days = len(points)
    if days == 0:
        return TimingAdvice(0, "insufficient", "无观测数据", note="该商品尚未进入快照采集")
    if days < min_days:
        return TimingAdvice(days, "insufficient", "数据积累中", note=f"已观测 {days} 天（{days}/{min_days}），暂不出时机结论")

    prices = [point.final_price for point in points]
    current = current if current is not None else prices[-1]
    percentile = sum(1 for price in prices if price <= current) / days
    lowest = min(points, key=lambda point: point.final_price)
    if percentile <= 0.25:
        verdict = f"当前处于观测低位（第 {percentile:.0%} 分位），可考虑入手"
    elif percentile >= 0.75:
        verdict = f"当前处于观测高位（第 {percentile:.0%} 分位），可观望或等大促"
    else:
        verdict = f"价格处于观测中位（第 {percentile:.0%} 分位），按需购买"
    return TimingAdvice(days, "ok", verdict, percentile, lowest, note=f"基于自观测 {days} 天数据（非全历史）")


def detect_fake_discount(
    points: list[PricePoint],
    min_days: int = MIN_OBSERVED_DAYS,
    rise_ratio: float = FAKE_RISE_RATIO,
    drop_ratio: float = FAKE_DROP_RATIO,
) -> FakeDiscountReport:
    """先涨后降检测——**并区分是谁在动**。

    形态成立（近期峰值明显高于基线、现价又回落）之后，看标价有没有真的涨：

    - **标价涨了** → 商家提价再打折，是套路，值得警惕（`suspected`）
    - **标价没涨，只是优惠缩了** → 券到期/收回，是促销窗口结束，
      **不是套路**（`coupon_withdrawn`）；对买家的动作也不同——等下一轮券，
      而不是"警惕这家店"

    以前只看到手价一条曲线，这两种情况都会被写成「先涨后降」，措辞是错的。
    """
    if len(points) < min_days:
        return FakeDiscountReport("insufficient", f"已观测 {len(points)} 天（不足 {min_days} 天），暂无法检测")
    earlier, recent = points[:-_RECENT_WINDOW], points[-_RECENT_WINDOW:]
    baseline = min(earlier, key=lambda point: point.final_price)
    peak = max(recent, key=lambda point: point.final_price)
    current = points[-1].final_price
    rise_threshold = baseline.final_price * Decimal(str(rise_ratio))
    drop_threshold = peak.final_price * Decimal(str(drop_ratio))
    if peak.final_price < rise_threshold or current > drop_threshold:
        return FakeDiscountReport("clean", "观测期内未检测到「先涨后降」特征")

    list_rose = peak.list_price > baseline.list_price
    if not list_rose:
        lost = peak.markdown - baseline.markdown
        return FakeDiscountReport(
            "coupon_withdrawn",
            f"到手价回升来自**优惠缩水**（不是商家提价）：标价一直是 ￥{fmt_amount(baseline.list_price)} 没动，"
            f"优惠从 ￥{fmt_amount(baseline.markdown)} 缩到 ￥{fmt_amount(peak.markdown)}"
            f"（少了 ￥{fmt_amount(lost)}），到手价 {baseline.day} ￥{fmt_amount(baseline.final_price)} → "
            f"{peak.day} ￥{fmt_amount(peak.final_price)}。更可能是促销券到期或收回，可以等下一轮。",
            driver="coupon",
        )
    return FakeDiscountReport(
        "suspected",
        f"疑似先涨后降：标价 {baseline.day} ￥{fmt_amount(baseline.list_price)} → "
        f"{peak.day} 涨至 ￥{fmt_amount(peak.list_price)}（到手 ￥{fmt_amount(baseline.final_price)} → "
        f"￥{fmt_amount(peak.final_price)}）→ 现价 ￥{fmt_amount(current)}",
        driver="list_price",
    )


def sku_history_view(
    connection: sqlite3.Connection,
    platform: str,
    sku_id: str,
    current: Decimal | None = None,
    min_days: int = MIN_OBSERVED_DAYS,
) -> dict:
    """单商品的观测视图（供管线/Web 序列化）。"""
    points = load_history(connection, platform, sku_id)
    advice = advise_timing(points, current, min_days)
    view: dict = {
        "observed_days": advice.observed_days,
        "status": advice.status,
        "verdict": advice.verdict,
        "note": advice.note,
    }
    if points:
        # 序列里同时带标价与优惠额：前端可以据此把"商家调价"和"优惠变动"分开看
        view["series"] = [
            {"date": point.day, "price": str(point.final_price), "list_price": str(point.list_price)}
            for point in points[-30:]
        ]
    if advice.lowest is not None:
        view["lowest"] = f"{fmt_amount(advice.lowest.final_price)}（{advice.lowest.day}）"
    fake = detect_fake_discount(points, min_days)
    if fake.status == "suspected":
        view["fake_discount"] = fake.detail
    elif fake.status == "coupon_withdrawn":
        # 单独一个键：它不是"可疑信号"，是"促销窗口结束了"，前端该用不同的语气展示
        view["coupon_withdrawn"] = fake.detail
    return view


@dataclass
class PriceDrop:
    platform: str
    sku_id: str
    title: str
    previous_price: Decimal
    latest_price: Decimal
    drop_pct: float
    previous_date: str
    latest_date: str
    new_low: bool


def price_drops(connection: sqlite3.Connection, threshold_pct: float = 5.0) -> list[PriceDrop]:
    """站内降价提醒：最新观测价较**上次观测**下降 ≥ 阈值，或创观测新低。"""
    rows = connection.execute(
        "SELECT platform, sku_id, captured_date, final_price, title FROM price_snapshots ORDER BY platform, sku_id, captured_date"
    ).fetchall()
    grouped: dict[tuple[str, str], list[tuple[str, Decimal, str]]] = {}
    for platform, sku_id, captured_date, final_price, title in rows:
        grouped.setdefault((platform, sku_id), []).append((captured_date, Decimal(final_price), title))

    drops: list[PriceDrop] = []
    for (platform, sku_id), entries in grouped.items():
        if len(entries) < 2:
            continue
        previous_date, previous_price, _ = entries[-2]
        latest_date, latest_price, title = entries[-1]
        earlier_min = min(price for _, price, _ in entries[:-1])
        drop_pct = float((previous_price - latest_price) / previous_price * 100) if previous_price > 0 else 0.0
        new_low = latest_price < earlier_min
        if drop_pct >= threshold_pct or new_low:
            drops.append(
                PriceDrop(
                    platform=platform,
                    sku_id=sku_id,
                    title=title,
                    previous_price=previous_price,
                    latest_price=latest_price,
                    drop_pct=round(drop_pct, 1),
                    previous_date=previous_date,
                    latest_date=latest_date,
                    new_low=new_low,
                )
            )
    drops.sort(key=lambda drop: (-drop.drop_pct, drop.platform, drop.sku_id))
    return drops


def attach_history_insights(result, db_path: Path | None = None) -> None:
    """从快照库为同款组代表补充价格观测（库不存在/打不开时静默跳过，不阻塞主流程）。"""
    from priceradar.jobs.snapshot import DEFAULT_DB, connect

    path = db_path or DEFAULT_DB
    if not Path(path).exists():
        return
    try:
        connection = connect(path)
    except sqlite3.Error:
        return
    try:
        for offer, _ in result.breakdowns:
            view = sku_history_view(connection, offer.platform.value, offer.sku_id, current=offer.list_price)
            if view["observed_days"] > 0:
                result.history[f"{offer.platform.value}:{offer.sku_id}"] = view
    finally:
        connection.close()
