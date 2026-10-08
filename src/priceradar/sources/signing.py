"""开放平台 MD5 签名（京东联盟 / 拼多多 通用算法）。

算法：MD5(secret + 按 key 升序拼接的 k1v1k2v2... + secret)，结果大写。
"""

from __future__ import annotations

import hashlib


def md5_sign(params: dict[str, str], secret: str) -> str:
    joined = secret + "".join(f"{key}{params[key]}" for key in sorted(params)) + secret
    return hashlib.md5(joined.encode("utf-8")).hexdigest().upper()
