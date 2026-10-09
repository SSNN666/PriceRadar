"""DeepSeek LLM 客户端（OpenAI 兼容协议，零第三方依赖）。

两个入口，分工明确：
- `complete(system, user, ...)`：单轮「系统 + 用户」→ 文本/结构化对象。
  既有节点（llm_nodes）用它，**契约保持不变**。
- `chat(messages, tools=...)`：多轮消息列表 + 工具调用（function calling）。
  Agent 循环用它。DeepSeek 兼容 OpenAI 的 `tools` / `tool_calls` / `role:"tool"`，
  支持一轮返回多个调用（并行），且**对 `tool_call_id` 强校验**——助手消息里的
  每个 `tool_calls` 条目，下一轮都必须有对应的 `tool` 消息，否则请求直接失败。

设计（PRD §5.2）：
- 结构化输出走 JSON Schema 校验；校验失败自动重试一次；仍失败抛 LlmError
- 调用方（llm_nodes / loop）必须提供确定性兜底路径，LLM 失败不阻塞主流程
- `usage` 在这一层读出并向上传递——DeepSeek 把 prompt 分成**缓存命中/未命中**
  两档，单价差 30-50 倍，多轮 Agent 的成本大头就在这里
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from priceradar.config import load_env

API_URL = "https://api.deepseek.com/chat/completions"


class LlmError(RuntimeError):
    pass


@dataclass(frozen=True)
class Usage:
    """一次调用的 token 计量。缺失字段一律记 0 —— **不猜**。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_hit_tokens: int = 0    # DeepSeek: prompt_cache_hit_tokens
    cache_miss_tokens: int = 0   # DeepSeek: prompt_cache_miss_tokens

    @property
    def is_empty(self) -> bool:
        return not (self.prompt_tokens or self.completion_tokens or self.cache_hit_tokens or self.cache_miss_tokens)

    @property
    def cache_hit_rate(self) -> float:
        """缓存命中率。分母用「命中 + 未命中」而非 `prompt_tokens`。

        两者通常相等，但分开记时更稳：某些实现的 `prompt_tokens` 与两项之和对不上。
        """
        total = self.cache_hit_tokens + self.cache_miss_tokens
        return self.cache_hit_tokens / total if total else 0.0


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict           # 已解析（协议里是 JSON 字符串）
    raw_arguments: str = ""   # 原文，解析失败时供排查


@dataclass
class ChatReply:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


def _parse_usage(raw: Any) -> Usage:
    if not isinstance(raw, dict):
        return Usage()

    def number(key: str) -> int:
        value = raw.get(key)
        # bool 是 int 的子类，得排除；字符串数字也接受（部分网关会序列化成字符串）
        if isinstance(value, bool) or value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return Usage(
        prompt_tokens=number("prompt_tokens"),
        completion_tokens=number("completion_tokens"),
        total_tokens=number("total_tokens"),
        cache_hit_tokens=number("prompt_cache_hit_tokens"),
        cache_miss_tokens=number("prompt_cache_miss_tokens"),
    )


def _parse_tool_call(item: dict) -> ToolCall:
    """解析单个 tool_call。

    `arguments` 是 JSON **字符串**（OpenAI 协议），但实践中会坏：
    模型偶尔给出截断的 JSON，部分兼容实现直接给对象。三种都要能接住，
    坏 JSON 解析成 `{}` 交给工具的参数校验去报可读错误——**不能在这里崩掉循环**。
    """
    function = item.get("function") or {}
    raw = function.get("arguments")
    if isinstance(raw, dict):
        arguments, raw_text = raw, json.dumps(raw, ensure_ascii=False)
    else:
        raw_text = str(raw or "")
        try:
            parsed = json.loads(raw_text) if raw_text.strip() else {}
        except json.JSONDecodeError:
            parsed = {}
        arguments = parsed if isinstance(parsed, dict) else {}
    return ToolCall(
        id=str(item.get("id") or ""),
        name=str(function.get("name") or ""),
        arguments=arguments,
        raw_arguments=raw_text,
    )


class DeepSeekClient:
    """transport 可注入（测试离线化）：callable(body: dict) -> 响应 dict。"""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        transport: Callable[[dict], dict] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._transport = transport or self._default_transport

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        api_key = load_env("DEEPSEEK_API_KEY").get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise LlmError("未配置 DEEPSEEK_API_KEY（环境变量或 .env）")
        return cls(api_key)

    def _default_transport(self, body: dict) -> dict:
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise LlmError(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:200]}") from error
        except urllib.error.URLError as error:
            raise LlmError(f"网络错误：{error.reason}") from error

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_object: bool = False,
    ) -> ChatReply:
        """多轮对话 / 工具调用。返回结构化 `ChatReply`（含 usage）。

        `messages` 用 OpenAI 的线上格式（裸 dict）——它本身就是协议，
        再包一层模型只会多一次翻译和一类 bug，且 `tool_call_id` 的严格配对
        在包装层里最容易出错。
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        if json_object:
            body["response_format"] = {"type": "json_object"}

        data = self._transport(body)
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise LlmError(f"响应结构异常：{str(data)[:200]}") from error
        if not isinstance(message, dict):
            raise LlmError(f"响应结构异常：{str(data)[:200]}")

        return ChatReply(
            content=message.get("content"),
            tool_calls=[_parse_tool_call(item) for item in (message.get("tool_calls") or [])],
            usage=_parse_usage(data.get("usage")),
            finish_reason=str(choice.get("finish_reason") or ""),
            raw=data,
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        usage_sink: Callable[[Usage], None] | None = None,
    ) -> str | BaseModel:
        """调用模型。传 json_schema 时返回校验过的模型实例（失败重试一次）。

        `usage_sink` 可选：传入后每次真实调用（**只有拿到 usage 时**）回调一次，
        供上层累计 token 计量。默认 None，既有调用点零改动。
        """
        prompt = user
        last_error: Exception | None = None
        for _ in range(2):  # 首次 + 校验失败重试一次
            reply = self.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                json_object=json_schema is not None,
            )
            if usage_sink is not None and not reply.usage.is_empty:
                usage_sink(reply.usage)
            content = reply.content or ""
            if json_schema is None:
                return content
            try:
                return json_schema.model_validate_json(content)
            except ValidationError as error:
                last_error = error
                prompt = f"{user}\n\n注意：上次输出未通过 schema 校验（{error}）。请严格按要求的 JSON 格式输出，不要包含多余文本。"
        raise LlmError(f"结构化输出反复校验失败：{last_error}")
