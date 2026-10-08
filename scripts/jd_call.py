"""京东联盟 API 通用调用探针（开发调试用，直接打印原始响应）。

用法：
    python scripts/jd_call.py jd.union.open.goods.query '{"goodsReqDTO":{"keyword":"耳机","pageIndex":1,"pageSize":10}}'
    python scripts/jd_call.py jd.union.open.goods.jingfen.query '{"goodsReq":{"eliteId":1,"pageIndex":1,"pageSize":10}}' --with-token

说明：
- 第二个参数是 360buy_param_json（业务参数 JSON 字符串）
- --with-token 时携带 JD_UNION_AUTH_KEY 作为 access_token
- 凭证读取：JD_UNION_APP_KEY / JD_UNION_APP_SECRET / JD_UNION_AUTH_KEY（环境变量 > .env）
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

GATEWAY = "https://api.jd.com/routerjson"
ROOT = Path(__file__).resolve().parent.parent

USAGE = """用法：python scripts/jd_call.py <method> '<业务参数JSON>' [--with-token]

示例：
  python scripts/jd_call.py jd.union.open.goods.query '{"goodsReqDTO":{"keyword":"耳机","pageIndex":1,"pageSize":10}}'
  python scripts/jd_call.py jd.union.open.goods.jingfen.query '{"goodsReq":{"eliteId":1,"pageIndex":1,"pageSize":10}}'
"""


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
    joined = secret + "".join(f"{k}{params[k]}" for k in sorted(params)) + secret
    return hashlib.md5(joined.encode("utf-8")).hexdigest().upper()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        print(USAGE)
        sys.exit(1)
    method, business_json = args[0], args[1]

    env = load_env()
    app_key = env.get("JD_UNION_APP_KEY")
    secret = env.get("JD_UNION_APP_SECRET")
    if not app_key or not secret:
        print("未找到 JD_UNION_APP_KEY / JD_UNION_APP_SECRET，请检查 .env。")
        sys.exit(1)

    params: dict[str, str] = {
        "app_key": app_key,
        "method": method,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "format": "json",
        "v": "1.0",
        "sign_method": "md5",
        "360buy_param_json": business_json,
    }
    if "--with-token" in sys.argv and env.get("JD_UNION_AUTH_KEY"):
        params["access_token"] = env["JD_UNION_AUTH_KEY"]
    params["sign"] = sign_params(params, secret)

    request = urllib.request.Request(
        GATEWAY,
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

    for key, value in data.items():
        if key.endswith("_responce") or key.endswith("_response"):
            inner = value.get("queryResult") or value.get("result")
            if inner:
                try:
                    inner = json.loads(inner)
                except (TypeError, ValueError):
                    pass
                print(json.dumps({"outer_code": value.get("code"), "inner": inner}, ensure_ascii=False, indent=2)[:4500])
            else:
                print(json.dumps(value, ensure_ascii=False, indent=2)[:2500])
            return
    print(json.dumps(data, ensure_ascii=False, indent=2)[:2500])


if __name__ == "__main__":
    main()
