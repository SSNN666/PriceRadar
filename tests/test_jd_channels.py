"""京粉频道路由测试（京东无关键词搜索 → 靠品类路由到达正确频道池）。"""

import pytest

from priceradar.sources.jd_channels import (
    CATEGORY_CHANNELS,
    CHANNEL_LABELS,
    FALLBACK_CHANNELS,
    detect_categories,
    relax_tokens,
    route_channels,
)


def test_routes_digital_query_to_digital_channel():
    """搜耳机却拉数码频道是对的；搜保温杯拉数码频道就是本次要修的 bug。"""
    assert 24 in route_channels("降噪耳机")
    assert 24 in route_channels("机械键盘")


def test_routes_home_query_to_kitchen_channel():
    """实测「保温杯」在 27 号（家居厨具）能出 16 条，在 24 号（数码）只有 1 条噪声。"""
    channels = route_channels("保温杯")
    assert channels[0] == 27
    assert 24 not in channels


def test_detect_categories_multiple_and_stable_order():
    cats = detect_categories("保温杯 键盘")
    assert "家居厨具" in cats and "数码" in cats
    assert cats == detect_categories("保温杯 键盘")  # 结果稳定可复现


def test_unknown_query_falls_back():
    assert route_channels("qwerty zzz") == list(FALLBACK_CHANNELS)
    assert detect_categories("qwerty zzz") == []


def test_max_channels_caps_result():
    assert len(route_channels("保温杯 键盘 面膜 猫粮", max_channels=2)) == 2


def test_routed_channels_are_all_live():
    """路由表里不得出现实测已失效的频道（6/7/11-14/16-21/35/36 报"资源位失效"）。

    否则会出现"路由到了但一个商品都拉不到"的静默失败。
    """
    routed = {channel for channels in CATEGORY_CHANNELS.values() for channel in channels}
    assert routed <= set(CHANNEL_LABELS)
    assert set(FALLBACK_CHANNELS) <= set(CHANNEL_LABELS)


def test_relax_tokens_keeps_tail_noun():
    """实测：「降噪耳机」精确 0 条，放宽到「耳机」得 43 条——中文名词在词尾。"""
    tokens = relax_tokens("降噪耳机")
    assert "降噪耳机" in tokens
    assert "耳机" in tokens


def test_relax_tokens_splits_multi_word_query():
    tokens = relax_tokens("不锈钢 保温杯")
    assert "不锈钢" in tokens and "保温杯" in tokens


def test_relax_tokens_ignores_single_char_noise():
    """单字片段（如"的"）会匹配到几乎一切，必须排除。"""
    assert all(len(token) >= 2 for token in relax_tokens("好的 杯子"))


@pytest.mark.parametrize("query", ["降噪耳机", "保温杯", "面膜", "猫粮", "羽绒服"])
def test_common_queries_route_somewhere(query):
    assert route_channels(query)
