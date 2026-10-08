"""大促日历（PRD §3 P1「价格与时机关」的规则层）。

背景（实测，2026-10-07）：京东 `goods.activity.query` 只返回**历史存档**活动
（最新截至 2025-03；type/actStatus 参数均无效，详见 docs/jd-api-findings.md §7）
→ 不依赖接口，改用公开规律的年度大促日历（确定性规则，标注"参考往年规律"）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class Campaign:
    name: str
    start: date
    end: date
    note: str = ""


def campaigns_for_year(year: int) -> list[Campaign]:
    """年度大促窗口（典型区间，参考往年规律）。"""
    return [
        Campaign("年货节", date(year, 1, 6), date(year, 1, 25), "春节前"),
        Campaign("618", date(year, 5, 24), date(year, 6, 18), "5 月下旬预售，6 月 18 高潮"),
        Campaign("双11", date(year, 10, 20), date(year, 11, 11), "10 月中下旬预售，11 月 11 高潮"),
        Campaign("双12", date(year, 12, 8), date(year, 12, 12), ""),
    ]


def upcoming_campaigns(today: date, horizon_days: int = 60) -> list[Campaign]:
    """未来 horizon_days 内（含进行中）的大促。"""
    limit = today + timedelta(days=horizon_days)
    campaigns = campaigns_for_year(today.year) + campaigns_for_year(today.year + 1)
    return [campaign for campaign in campaigns if campaign.end >= today and campaign.start <= limit]


def campaign_context(today: date | None = None, horizon_days: int = 60) -> str:
    """一行大促情报（确定性）；无大促窗口时返回空串。"""
    today = today or date.today()
    upcoming = upcoming_campaigns(today, horizon_days)
    if not upcoming:
        return ""
    parts: list[str] = []
    for campaign in upcoming[:2]:
        if campaign.start <= today <= campaign.end:
            parts.append(f"{campaign.name}大促进行中（至 {campaign.end.month} 月 {campaign.end.day} 日）")
        else:
            days = (campaign.start - today).days
            suffix = f"，{campaign.note}" if campaign.note else ""
            parts.append(
                f"距{campaign.name}大促开始还有 {days} 天（{campaign.start.month} 月 {campaign.start.day} 日开跑{suffix}）"
            )
    return "；".join(parts)
