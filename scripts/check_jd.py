"""京东联盟 API 连通性检查：关键词搜索商品（jd.union.open.goods.query）。

用法：
    python scripts/check_jd.py "耳机"              # 不带 access_token
    python scripts/check_jd.py "耳机" --with-token  # 带上授权key（作为 access_token）

脚本会自动尝试三种 timestamp 格式（13 位毫秒 / 10 位秒 / 字符串），直到不再报时间戳类错误。
读取凭证：JD_UNION_APP_KEY / JD_UNION_APP_SECRET / JD_UNION_AUTH_KEY（环境变量 > .env）
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

GATEWAY = "https://api.jd.com/routerjson"
METHOD = "jd.union.open.goods.query"
ROOT = Path(__file__).resolve().parent.parent


def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw = line.partition("=")
            values[key.strip()] = raw.split(" #", 1)[0].strip()
    for key in ("JD_UNION_APP_KEY", "JD_UNION_APP_SECRET", "JD_UNION_AUTH_KEY"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def sign_params(params: dict[str, str], secret: str) -> str:
    """京东签名（MD5）：secret + 排序后的 k1v1k2v2... + secret → MD5 → 大写。"""
    joined = secret + "".join(f"{k}{params[k]}" for k in sorted(params)) + secret
    return hashlib.md5(joined.encode("utf-8")).hexdigest().upper()


def timestamps() -> list[tuple[str, str]]:
    # 实测（2026-10-05）：京东要求字符串格式，毫秒/秒时间戳均报「时间戳参数不正确」
    now = time.time()
    return [
        ("str", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("ms", str(int(now * 1000))),
        ("s", str(int(now))),
    ]


def call(api_params: dict[str, str], app_key: str, secret: str) -> dict:
    request = urllib.request.Request(
        GATEWAY,
        data=urllib.parse.urlencode(api_params).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def summarize(data: dict) -> None:
    for key, value in data.items():
        if key.endswith("_responce") or key.endswith("_response"):
            inner_raw = value.get("queryResult") or value.get("result")
            print(f"外层 code={value.get('code')} key={key}")
            if not inner_raw:
                print(json.dumps(value, ensure_ascii=False)[:800])
                return
            parsed = json.loads(inner_raw)
            items = parsed.get("data") or []
            print(f"内层 code={parsed.get('code')} message={parsed.get('message')} 返回 {len(items)} 条\n")
            for item in items[:5]:
                name = str(item.get("skuName", ""))[:42]
                price = (item.get("priceInfo") or {}).get("price")
                coupons = (item.get("couponInfo") or {}).get("couponList") or []
                comments = item.get("comments")
                good_rate = item.get("goodCommentsShare")
                shop = (item.get("shopInfo") or {}).get("shopName", "")
                print(f"- {name}")
                print(f"  价 {price}｜券 {len(coupons)} 张｜好评率 {good_rate}｜评论 {comments}｜{shop}")
            return
    print(json.dumps(data, ensure_ascii=False, indent=2)[:1500])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    env = load_env()
    app_key = env.get("JD_UNION_APP_KEY")
    secret = env.get("JD_UNION_APP_SECRET")
    if not app_key or not secret:
        print("未找到 JD_UNION_APP_KEY / JD_UNION_APP_SECRET，请检查 .env。")
        sys.exit(1)

    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    keyword = argv[0] if argv else "耳机"
    use_token = "--with-token" in sys.argv
    business = json.dumps(
        {"goodsReqDTO": {"keyword": keyword, "pageIndex": 1, "pageSize": 10}},
        ensure_ascii=False,
    )

    last_data: dict | None = None
    for label, ts in timestamps():
        params: dict[str, str] = {
            "app_key": app_key,
            "method": METHOD,
            "timestamp": ts,
            "format": "json",
            "v": "1.0",
            "sign_method": "md5",
            "360buy_param_json": business,
        }
        if use_token and env.get("JD_UNION_AUTH_KEY"):
            params["access_token"] = env["JD_UNION_AUTH_KEY"]
        params["sign"] = sign_params(params, secret)
        try:
            data = call(params, app_key, secret)
        except urllib.error.HTTPError as error:
            print(f"[{label}] HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:300]}")
            continue
        last_data = data
        text = json.dumps(data, ensure_ascii=False)
        if "error_response" in data:
            error = data["error_response"]
            desc = error.get("zh_desc") or error.get("en_desc") or str(error)
            print(f"[{label}] 错误：{desc}")
            # 时间戳类错误时继续尝试其他格式
            if "timestamp" in text.lower() or "time" in text.lower() or "签名" in str(desc):
                continue
            break
        print(f"[{label}] ✅ 请求成功\n")
        summarize(data)
        return

    if last_data is not None:
        print("\n全部尝试的响应（最后一次）：")
        print(json.dumps(last_data, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
