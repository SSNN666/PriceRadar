"""全局测试夹具。

## 为什么要有「禁止联网」

这套测试号称全离线，但只靠人自觉打桩是不可靠的——实测抓到过 **6 个用例**
在真打外部接口：它们表面上都「打桩了」，只是桩错了地方（桩了 `run` 却没桩
LLM 客户端），于是每次 `pytest` 都在真花 DeepSeek 的钱、真烧拼多多的配额，
而拼多多账号正是项目文档里点名限流敏感的那个（判据是「调用量 vs 出单量」，
本项目账号天然不出单）。

发现方式是**把 `socket.getaddrinfo` 拦掉再跑全套**：联网的用例立刻现形，
而且整套从 13.2 秒掉到 2.9 秒——那 10 秒就是真实网络往返。

现在把这件事变成结构性的：任何真实网络调用都会当场报错，而不是悄悄发生。
确实需要联网的用例（目前没有）可以标 `@pytest.mark.network` 显式豁免。
"""

from __future__ import annotations

import socket

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "network: 允许该用例真实联网（默认禁止）")


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """默认拦掉一切真实网络。

    只拦 DNS 解析与建连，不动 `socket.socket` 本身——`ssl` 模块在导入时
    继承它，替换掉会让整个解释器起不来。
    """
    if request.node.get_closest_marker("network"):
        return

    def blocked(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "测试期间禁止真实联网。请打桩外部依赖（transport 可注入 / monkeypatch），"
            "或确需联网时给用例加 @pytest.mark.network。"
        )

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
