"""博查搜索客户端（L2 风险检索通道，PRD §5.8）。

实测（2026-10-05）：POST https://api.bocha.cn/v1/web-search，Bearer 认证；
响应体 data.webPages.value 为结果列表；新用户免费额度 1000 次。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from priceradar.config import load_env

API_URL = "https://api.bocha.cn/v1/web-search"


class BochaError(RuntimeError):
    pass


@dataclass
class SearchResult:
    title: str
    url: str
    site_name: str
    snippet: str


class BochaClient:
    """transport 可注入（测试离线化）：callable(body: dict) -> 响应 dict。"""

    def __init__(self, api_key: str, transport: Callable[[dict], dict] | None = None) -> None:
        self.api_key = api_key
        self._transport = transport or self._default_transport

    @classmethod
    def from_env(cls) -> "BochaClient":
        api_key = load_env("BOCHA_API_KEY").get("BOCHA_API_KEY", "")
        if not api_key:
            raise BochaError("未配置 BOCHA_API_KEY（环境变量或 .env）")
        return cls(api_key)

    def _default_transport(self, body: dict) -> dict:
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise BochaError(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:200]}") from error
        except urllib.error.URLError as error:
            raise BochaError(f"网络错误：{error.reason}") from error

    def web_search(self, query: str, count: int = 8) -> list[SearchResult]:
        data = self._transport({"query": query, "count": count, "summary": False})
        pages = ((data.get("data") or {}).get("webPages") or {}).get("value") or []
        code = data.get("code")
        if not pages and code not in (200, "200", None):
            raise BochaError(str(data.get("message") or data)[:200])
        return [
            SearchResult(
                title=str(page.get("name") or ""),
                url=str(page.get("url") or ""),
                site_name=str(page.get("siteName") or ""),
                snippet=str(page.get("snippet") or ""),
            )
            for page in pages
            if page.get("url")
        ]
