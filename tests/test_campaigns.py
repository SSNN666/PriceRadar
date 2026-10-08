"""大促日历测试（确定性规则层）。"""

from datetime import date

from priceradar.insights import campaign_context, campaigns_for_year, upcoming_campaigns


def test_campaigns_for_year_has_four_windows():
    names = [campaign.name for campaign in campaigns_for_year(2026)]
    assert names == ["年货节", "618", "双11", "双12"]


def test_upcoming_contains_double11_in_october():
    upcoming = upcoming_campaigns(date(2026, 10, 7))
    assert any(campaign.name == "双11" for campaign in upcoming)
    assert all(campaign.name != "618" for campaign in upcoming)


def test_upcoming_empty_in_quiet_window():
    # 3-20 + 60 天 = 5-19，早于 618 的 5-24 → 无大促窗口
    assert upcoming_campaigns(date(2026, 3, 20)) == []


def test_context_before_double11_counts_days():
    text = campaign_context(date(2026, 10, 7))
    assert "双11" in text
    assert "13 天" in text


def test_context_during_campaign():
    text = campaign_context(date(2026, 10, 25))
    assert "双11大促进行中" in text


def test_context_empty_when_no_campaign():
    assert campaign_context(date(2026, 3, 20)) == ""
