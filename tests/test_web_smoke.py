"""无头浏览器冒烟的**判定逻辑**测试（纯离线）。

脚本本身会真的起服务、开浏览器、调外部 API（不进 CI），但它的判定逻辑是纯函数，
必须可离线测——否则「护栏本身是不是摆设」就没人管了。
已经吃过一次亏：`#error` 的检查靠匹配固定属性串，属性顺序一变就静默失效。
"""

from __future__ import annotations

import pytest

from scripts.web_smoke import check, error_text

HEALTHY = (
    '<div id="error" class="error hidden"></div>'
    "Agent 正在工作 agent-tool ok agent-text dtier offer 缓存命中 我的优惠券"
)

BROKEN_TOOL = (
    '<div id="error" class="error hidden"></div>'
    "Agent 正在工作 agent-tool ok agent-tool bad agent-text dtier offer 缓存命中"
)

WITH_ERROR = (
    '<div id="error" class="error">出错了：拼多多检索失败</div>'
    "Agent 正在工作 agent-tool ok agent-text dtier offer 缓存命中 我的优惠券"
)


def failures(dom: str) -> list[str]:
    return [name for name, passed, _ in check(dom) if not passed]


def test_healthy_page_passes_everything():
    assert failures(HEALTHY) == []
    assert len(check(HEALTHY)) == 9


def test_failed_tool_is_detected():
    """这条正是发现 tool_defaults 污染的那条线索——工具标红必须让检查失败。"""
    assert "没有工具调用失败" in failures(BROKEN_TOOL)


def test_error_div_content_is_detected():
    assert "没有报错（#error 为空）" in failures(WITH_ERROR)


def test_empty_dom_fails_almost_everything():
    """页面根本没渲染出来时必须大面积失败——不能因为「没找到报错容器」而假通过。"""
    assert len(failures("")) >= 8


def test_missing_error_div_is_a_failure_not_a_pass():
    """**护栏自身的关键性质**：没有 #error 容器 ⇒ 失败，而不是「没报错所以通过」。

    第一版把「找不到容器」也返回空串，于是空 DOM 会让这条假通过——
    一个永远绿的检查比没有检查更糟。
    """
    assert error_text("<p>什么都没有</p>") is None
    assert "没有报错（#error 为空）" in failures("<p>什么都没有</p>")


def test_error_text_survives_attribute_reordering():
    """靠属性顺序匹配会静默失效；按元素取内容就不会。"""
    assert error_text('<div class="error hidden" id="error">出错了</div>') == "出错了"
    assert error_text('<div id="error">出错了</div>') == "出错了"
    assert error_text('<div id="error"></div>') == ""


def test_error_text_handles_multiline_content():
    assert "第一行" in (error_text('<div id="error">\n 第一行\n 第二行\n</div>') or "")


@pytest.mark.parametrize(
    "marker",
    ["agent-tool ok", "agent-text", "dtier", "offer", "缓存命中", "我的优惠券"],
)
def test_each_required_marker_is_actually_required(marker):
    """逐个拿掉必需元素，检查必须跟着失败——防止某个断言写成了恒真。"""
    broken = HEALTHY.replace(marker, "xxx")
    assert failures(broken), f"拿掉 {marker!r} 之后居然还全通过"
