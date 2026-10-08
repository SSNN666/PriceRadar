"""商品链接识别（PRD §5.3 输入层：链接 / 口令 / 图片 / 自然语言）。

**为什么需要**：拼多多 `goods.search` 的 `keyword` 参数可以直接吃一条**用户分享的
商品链接**并精确反查到那件商品——但要满足平台对"原始链接"的校验。

⚠️ **绝不清洗 URL 参数**（本模块的核心约束）：
实测同一条链接，**带 `uin` 参数 → 成功；去掉 `uin` → 被拒**
（*"该查询不是来自终端用户的原始链接"*）。`uin` 是分享者的用户标识，正是平台
用来证明"这是终端用户真实分享"的凭据。任何"规范化 URL / 剥掉 tracking 参数"的
习惯做法都会正好把它剥掉，然后被平台拒绝——所以这里**原样返回，只做识别**。
"""

from __future__ import annotations

import re

# 拼多多商品页与短链域名。刻意宽松匹配整条 URL，再由调用方原样使用。
_PDD_URL = re.compile(
    r"https?://[^\s<>\"']*(?:yangkeduo\.com|pinduoduo\.com|pdd\.cn)[^\s<>\"']*",
    re.IGNORECASE,
)
# 收尾常见的中文标点/括号（粘贴时常被带进来），但不触碰 URL 内部参数
_TRAILING = "）)】]》，,。；;！!？?、"


def extract_pdd_link(text: str | None) -> str | None:
    """从任意文本中识别拼多多链接，**原样返回**（含全部查询参数）。

    识别不出返回 None——调用方应退回关键词检索，不要猜测。
    """
    if not text:
        return None
    match = _PDD_URL.search(text)
    if not match:
        return None
    return match.group(0).rstrip(_TRAILING)
