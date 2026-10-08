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

PricePoint = tuple[str, Decimal]


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
    status: str                      # "insufficient" | "clean" | "suspected"
    detail: str


def load_history(connection: sqlite3.Connection, platform: str, sku_id: str) -> list[PricePoint]:
    """读取某商品的观测历史（按日期升序）。"""
    rows = connection.execute(
        "SELECT captured_date, final_price FROM price_snapshots WHERE platform = ? AND sku_id = ? ORDER BY captured_date",
        (platform, sku_id),
    ).fetchall()
    return [(row[0], Decimal(row[1])) for row in rows]


def advise_timing(points: list[PricePoint], current: Decimal | None = None, min_days: int = MIN_OBSERVED_DAYS) -> TimingAdvice:
    """时机建议：基于观测期内的分位数（低位可入手 / 中位按需 / 高位观望）。"""
    days = len(points)
    if days == 0:
        return TimingAdvice(0, "insufficient", "无观测数据", note="该商品尚未进入快照采集")
    if days < min_days:
        return TimingAdvice(days, "insufficient", "数据积累中", note=f"已观测 {days} 天（{days}/{min_days}），暂不出时机结论")

    prices = [price for _, price in points]
    current = current if current is not None else prices[-1]
    percentile = sum(1 for price in prices if price <= current) / days
    lowest_day, lowest_price = min(points, key=lambda pair: pair[1])
    if percentile <= 0.25:
        verdict = f"当前处于观测低位（第 {percentile:.0%} 分位），可考虑入手"
    elif percentile >= 0.75:
        verdict = f"当前处于观测高位（第 {percentile:.0%} 分位），可观望或等大促"
    else:
        verdict = f"价格处于观测中位（第 {percentile:.0%} 分位），按需购买"
    return TimingAdvice(days, "ok", verdict, percentile, (lowest_day, lowest_price), note=f"基于自观测 {days} 天数据（非全历史）")


def detect_fake_discount(
    points: list[PricePoint],
    min_days: int = MIN_OBSERVED_DAYS,
    rise_ratio: float = FAKE_RISE_RATIO,
    drop_ratio: float = FAKE_DROP_RATIO,
) -> FakeDiscountReport:
    """先涨后降检测：近 7 天峰值 ≥ 基线期最低价 × 升幅 且 现价 ≤ 峰值 × 降幅 → 疑似。"""
    if len(points) < min_days:
        return FakeDiscountReport("insufficient", f"已观测 {len(points)} 天（不足 {min_days} 天），暂无法检测")
    earlier, recent = points[:-_RECENT_WINDOW], points[-_RECENT_WINDOW:]
    baseline_day, baseline = min(earlier, key=lambda pair: pair[1])
    peak_day, peak = max(recent, key=lambda pair: pair[1])
    current = points[-1][1]
    rise_threshold = baseline * Decimal(str(rise_ratio))
    drop_threshold = peak * Decimal(str(drop_ratio))
    if peak >= rise_threshold and current <= drop_threshold:
        return FakeDiscountReport(
            "suspected",
            f"疑似先涨后降：{baseline_day} ￥{fmt_amount(baseline)} → {peak_day} 涨至 ￥{fmt_amount(peak)} → 现价 ￥{fmt_amount(current)}",
        )
    return FakeDiscountReport("clean", "观测期内未检测到「先涨后降」特征")


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
        view["series"] = [{"date": day, "price": str(price)} for day, price in points[-30:]]
    if advice.lowest is not None:
        view["lowest"] = f"{fmt_amount(advice.lowest[1])}（{advice.lowest[0]}）"
    fake = detect_fake_discount(points, min_days)
    if fake.status == "suspected":
        view["fake_discount"] = fake.detail
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
