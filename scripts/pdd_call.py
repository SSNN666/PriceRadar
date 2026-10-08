"""多多进宝 API 通用调用探针（开发调试用，直接打印原始响应）。

用法：
    python scripts/pdd_call.py pdd.ddk.member.authority.query pid=44845383_318359410
    python scripts/pdd_call.py pdd.ddk.rp.prom.url.generate p_id_list='["44845383_318359410"]' generate_authority_url=true

参数以 key=value 形式传入；值若为 JSON（以 [ 或 { 开头）原样传递。
凭证读取：PDD_CLIENT_ID / PDD_CLIENT_SECRET（环境变量 > .env）。
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

USAGE = """用法：python scripts/pdd_call.py <api_type> [key=value ...]

示例：
  python scripts/pdd_call.py pdd.ddk.member.authority.query pid=44845383_318359410
  python scripts/pdd_call.py pdd.ddk.rp.prom.url.generate p_id_list='["44845383_318359410"]' generate_authority_url=true
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
    for key in ("PDD_CLIENT_ID", "PDD_CLIENT_SECRET"):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def sign_params(params: dict[str, str], secret: str) -> str:
    joined = secret + "".join(f"{k}{params[k]}" for k in sorted(params)) + secret
    return hashlib.md5(joined.encode("utf-8")).hexdigest().upper()


def call(api_type: str, client_id: str, secret: str, **params: str) -> dict:
    payload: dict[str, str] = {
        "type": api_type,
        "client_id": client_id,
        "timestamp": str(int(time.time())),
        "data_type": "JSON",
    }
    payload.update(params)
    payload["sign"] = sign_params(payload, secret)
    request = urllib.request.Request(
        ENDPOINT,
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    env = load_env()
    client_id = env.get("PDD_CLIENT_ID")
    secret = env.get("PDD_CLIENT_SECRET")
    if not client_id or not secret:
        print("未找到 PDD_CLIENT_ID / PDD_CLIENT_SECRET，请检查 .env。")
        sys.exit(1)

    api_type = sys.argv[1]
    params: dict[str, str] = {}
    for argument in sys.argv[2:]:
        key, _, value = argument.partition("=")
        params[key] = value

    try:
        data = call(api_type, client_id, secret, **params)
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:500]}")
        sys.exit(1)
    except urllib.error.URLError as error:
        print(f"网络错误：{error.reason}")
        sys.exit(1)

    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
