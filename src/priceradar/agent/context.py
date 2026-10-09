"""上下文组装（Context Builder）：把四层状态拼成一次模型调用能吃的消息列表。

面试里「多轮对话」必答的一点是**分清四种状态**，不能都当成一个消息数组：

| 层 | 在本项目里是什么 | 生命周期 |
|---|---|---|
| Chat History | `session_store` 里的完整消息表 | 永久（落库、可审计） |
| Chat Memory | **本模块的产物**——本轮真正喂给模型的那一段 | 一次调用 |
| Session State | `SessionState` 结构化槽位（关键词/商品/goods_sign） | 跨轮存活 |
| Long-Term Memory | 暂不做（本项目的"长期偏好"是 watchlist，另有其表） |

两条硬约束：
1. **整轮丢弃**。裁剪只按「轮」为单位——留下半轮会让 `tool_calls` 与 `tool`
   消息配不上对，服务端会以 `tool_call_id` 不成对直接拒绝请求。
2. **固定前缀在前**。system prompt 是常量、不含任何变量插值，这样工具定义 +
   系统提示能吃到 DeepSeek 的 prompt 缓存（命中与未命中单价差 30-50 倍）。
   易变内容（槽位、历史、本轮提问）一律排在它后面。

**不做滚动摘要**：64K 上下文配这类短会话，要很久才会撞到预算，而摘要本身
既花钱又可能丢事实。先用「最近若干整轮 + 槽位 + 确定性裁剪」，等有真实会话
长度数据再谈。

裁剪策略：从**最老的一轮**开始整轮丢弃，直到装得下；连一轮都装不下就全丢光，
只留固定前缀与槽位——**绝不把超预算的历史原样送出去**（那会在下一层被循环的
上下文预算拦下，用户的问题根本没机会被模型看到）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

@dataclass
class SessionState:
    """结构化槽位：历史被裁掉之后仍然活下来的那部分。

    刻意用**字段**而不是自然语言摘要——字段是确定性的、可测的，
    也不会随模型措辞漂移。
    """

    keyword: str = ""       # 当前讨论的检索关键词
    product: str = ""       # 当前讨论的商品名（供 scan_risk 用）
    goods_sign: str = ""    # 拼多多商品引用（供 find_similar 用）
    platform: str = ""

    def is_empty(self) -> bool:
        return not any((self.keyword, self.product, self.goods_sign, self.platform))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "SessionState":
        raw = raw or {}
        return cls(
            keyword=str(raw.get("keyword") or ""),
            product=str(raw.get("product") or ""),
            goods_sign=str(raw.get("goods_sign") or ""),
            platform=str(raw.get("platform") or ""),
        )

    def to_prompt(self) -> str:
        """槽位 → 一句确定性的中文提示。空槽位返回空串（不发无意义的消息）。"""
        if self.is_empty():
            return ""
        bits = []
        if self.product:
            bits.append(f"商品「{self.product}」")
        if self.keyword:
            bits.append(f"检索关键词「{self.keyword}」")
        if self.goods_sign:
            bits.append(f"goods_sign={self.goods_sign}")
        if self.platform:
            bits.append(f"平台「{self.platform}」")
        return (
            "【本会话已知的当前对象】" + "；".join(bits) + "。\n"
            "用户接下来的追问（如「那有没有踩坑的」「再找找更便宜的」）如未指明对象，"
            "默认就是在说上面这个。需要 goods_sign 时直接用上面这个，不要自己编。"
        )


def update_state(state: SessionState, run) -> SessionState:
    """从本轮 Agent 运行结果里抽取槽位。

    只取**确定性**字段（工具真的返回过什么），不做 LLM 归纳——
    槽位是后续几轮的锚点，让它漂移比没有更糟。
    """
    from priceradar.agent.pipeline import PipelineResult

    updated = SessionState(**state.to_dict())
    for step in run.steps:
        for execution in step.tool_calls:
            if not execution.ok:
                continue
            value = execution.value
            if isinstance(value, PipelineResult):
                updated.keyword = value.keyword or updated.keyword
                updated.product = value.keyword or updated.product
                source = value.source
                updated.platform = source.platform.value if source.platform else updated.platform
                sign = (source.source_extra or {}).get("goods_sign", "")
                if sign:
                    updated.goods_sign = sign
                elif value.breakdowns and not updated.goods_sign:
                    # 源商品没有 goods_sign 时退而求其次，取任意一个有签的报价
                    for offer, _ in value.breakdowns:
                        candidate = (offer.source_extra or {}).get("goods_sign", "")
                        if candidate:
                            updated.goods_sign = candidate
                            break
    return updated


def _size(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, default=str))


def build_messages(
    system_prompt: str,
    state: SessionState,
    turns: list[list[dict]],
    query: str,
    *,
    char_budget: int,
) -> tuple[list[dict], str]:
    """组装本轮要喂给模型的消息。

    返回 `(messages, note)`；`note` 非空表示发生了裁剪（要如实告知，不静默丢）。
    顺序：固定 system → 槽位 → 历史各轮 → 本轮提问。
    """
    fixed = [{"role": "system", "content": system_prompt}]
    if state.to_prompt():
        fixed.append({"role": "system", "content": state.to_prompt()})
    tail = [{"role": "user", "content": query}]

    kept = list(turns)
    dropped = 0
    while kept:
        candidate = fixed + [message for turn in kept for message in turn] + tail
        if _size(candidate) <= char_budget:
            return candidate, _trim_note(dropped, len(kept))
        kept.pop(0)          # 从最老的一轮**整轮**丢弃
        dropped += 1

    # 连一轮都放不下（单轮历史本身就超预算）→ 丢光历史，只留固定前缀 + 槽位兜底。
    # 之前这里写作 `if not kept and turns`，但循环里的 `len(kept) <= MIN_TURNS`
    # 让 kept 永远不为空，这段兜底根本走不到，超预算的历史被原样送了出去。
    return fixed + tail, _trim_note(dropped, 0, all_dropped=True)


def _trim_note(dropped: int, kept_turns: int, *, all_dropped: bool = False) -> str:
    """裁剪要如实告知，不能静默丢。"""
    if not dropped:
        return ""
    if all_dropped:
        return f"为控制上下文长度，本会话较早的 {dropped} 轮对话已全部省略；若仍相关请重新询问。"
    return (
        f"为控制上下文长度，已省略本会话较早的 {dropped} 轮对话"
        f"（保留最近 {kept_turns} 轮）；更早的结论若仍相关，请重新询问。"
    )


# 供调用方复用的字段名清单（避免到处写字符串字面量）
STATE_FIELDS = tuple(item.name for item in fields(SessionState))
