"""博查搜索 API 连通性检查。

用法：
    python scripts/check_bocha.py "索尼 WH-1000XM6 避雷"

读取凭证顺序：环境变量 BOCHA_API_KEY > 项目根目录 .env
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "https://api.bocha.cn/v1/web-search"
ROOT = Path(__file__).resolve().parent.parent


def load_api_key() -> str | None:
    key = os.environ.get("BOCHA_API_KEY")
    if key:
        return key.strip()
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("BOCHA_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    api_key = load_api_key()
    if not api_key:
        print("未找到 BOCHA_API_KEY。")
        print("请先复制 .env.example 为 .env 并填入密钥（见 docs/accounts-onboarding.md 第 3 节）。")
        sys.exit(1)

    query = sys.argv[1] if len(sys.argv) > 1 else "索尼 WH-1000XM6 避雷"
    payload = json.dumps({"query": query, "count": 5, "summary": True}).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:500]}")
        sys.exit(1)
    except urllib.error.URLError as error:
        print(f"网络错误：{error.reason}")
        sys.exit(1)

    pages = ((data.get("data") or {}).get("webPages") or {}).get("value") or []
    print(f"✅ 连通成功。查询：{query}，返回 {len(pages)} 条结果\n")
    if not pages:
        print("（未解析到 webPages.value，原始响应片段：）")
        print(json.dumps(data, ensure_ascii=False)[:500])
        return
    for page in pages[:5]:
        title = page.get("name", "")
        site = page.get("siteName", "")
        url = page.get("url", "")
        snippet = (page.get("snippet") or "").replace("\n", " ")[:100]
        print(f"- {title} | {site}")
        print(f"  {url}")
        if snippet:
            print(f"  {snippet}")
        print()


if __name__ == "__main__":
    main()
