"""拼多多（多多进宝）API 连通性检查：搜索关键词并打印前几条商品。

用法：
    python scripts/check_pdd.py "降噪耳机"

读取凭证：环境变量 PDD_CLIENT_ID / PDD_CLIENT_SECRET > 项目根目录 .env
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
from pathlib import Path

ENDPOINT = "https://gw-api.pinduoduo.com/api/router"
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
    for key in ("PDD_CLIENT_ID", "PDD_CLIENT_SECRET"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def sign_params(params: dict[str, str], secret: str) -> str:
    """拼多多签名：secret + 排序后的 k1v1k2v2... + secret 做 MD5，大写。"""
    joined = secret + "".join(f"{k}{params[k]}" for k in sorted(params)) + secret
    return hashlib.md5(joined.encode("utf-8")).hexdigest().upper()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    env = load_env()
    client_id = env.get("PDD_CLIENT_ID")
    secret = env.get("PDD_CLIENT_SECRET")
    if not client_id or not secret:
        print("未找到 PDD_CLIENT_ID / PDD_CLIENT_SECRET，请检查 .env（见 docs/accounts-onboarding.md 第 2 节）。")
        sys.exit(1)

    keyword = sys.argv[1] if len(sys.argv) > 1 else "降噪耳机"
    params = {
        "type": "pdd.ddk.goods.search",
        "client_id": client_id,
        "timestamp": str(int(time.time())),
        "data_type": "JSON",
        "keyword": keyword,
        "page": "1",
        "page_size": "10",
    }
    if env.get("PDD_PID"):
        params["pid"] = env["PDD_PID"]
    params["sign"] = sign_params(params, secret)

    request = urllib.request.Request(
        ENDPOINT,
        data=urllib.parse.urlencode(params).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
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

    if "error_response" in data:
        error = data["error_response"]
        print(f"❌ API 错误 {error.get('error_code')}: {error.get('error_msg')}")
        print(f"   完整错误信息：{json.dumps(error, ensure_ascii=False)}")
        sys.exit(1)

    result = data.get("goods_search_response") or {}
    goods = result.get("goods_list") or []
    print(f"✅ 连通成功。关键词：{keyword}，本次返回 {len(goods)} 条（显示前 5 条）\n")
    for item in goods[:5]:
        price = int(item.get("min_group_price", 0)) / 100
        coupon = int(item.get("coupon_discount", 0)) / 100
        print(f"- {item.get('goods_name', '')[:42]}")
        print(
            f"  拼团价 ￥{price:.2f}｜优惠券 ￥{coupon:.2f}｜销量 {item.get('sales_tip', '')}"
            f"｜{item.get('mall_name', '')}"
        )
    if result.get("total_count") is not None:
        print(f"\n（总计约 {result.get('total_count')} 条）")


if __name__ == "__main__":
    main()
