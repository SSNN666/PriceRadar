"""外部接口调用的瞬时故障重试（PRD §10 可用性）。

背景：2026-10-07 实测遇到路由器 DNS 转发临时失效（网关可达、公网 IP 可达，
唯独域名解析全部超时），一次查询直接失败。快照是"过期不补"的资产——
历史价无法回采，一次几秒的网络抖动不该让它丢一天。

原则：只重试"再试一次可能就好"的故障（DNS 解析失败 / 超时 / 连接重置 / 5xx / 429）；
业务错误（签名错、无权限、参数非法、备案未通过）一律立刻抛出——重试既无效又浪费额度，
还会把真问题拖成慢失败。
"""

from __future__ import annotations

import json
import time
import urllib.error
from typing import Callable, TypeVar

T = TypeVar("T")

DEFAULT_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 0.5  # 秒；退避 0.5 → 1.0，最坏情况总共多等 1.5 秒


def is_transient(error: BaseException) -> bool:
    """该异常是否属于"再试一次可能就好"的瞬时故障。"""
    if isinstance(error, urllib.error.HTTPError):
        # 5xx 是服务端自己的问题、429 是限流 → 值得重试；
        # 其余 4xx 是请求本身有问题（凭证/参数/权限），重试无用
        return error.code >= 500 or error.code == 429
    if isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return True  # DNS 解析失败 / 超时 / 连接被重置
    return isinstance(error, json.JSONDecodeError)  # 响应被截断（网关抖动）


def call_with_retry(
    call: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    sleeper: Callable[[float], None] | None = None,
) -> T:
    """执行 call()；瞬时故障按指数退避重试，其余异常立刻上抛。

    sleeper 可注入（测试用来验证退避节奏且不真等待）；注意默认值在调用时
    才解析 time.sleep——写成参数默认值会在导入时绑定，打桩就失效了。
    """
    sleep = sleeper or time.sleep
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as error:
            if attempt >= attempts or not is_transient(error):
                raise
            sleep(base_delay * (2 ** (attempt - 1)))
    raise AssertionError("循环要么返回要么抛出，不应走到这里")
