"""DeepSeek LLM 客户端（OpenAI 兼容协议，零第三方依赖）。

设计（PRD §5.2）：
- 结构化输出走 JSON Schema 校验；校验失败自动重试一次；仍失败抛 LlmError
- 调用方（llm_nodes）必须提供确定性兜底路径，LLM 失败不阻塞主流程
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from priceradar.config import load_env

API_URL = "https://api.deepseek.com/chat/completions"


class LlmError(RuntimeError):
    pass


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

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: type[BaseModel] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str | BaseModel:
        """调用模型。传 json_schema 时返回校验过的模型实例（失败重试一次）。"""
        prompt = user
        last_error: Exception | None = None
        for _ in range(2):  # 首次 + 校验失败重试一次
            body: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if json_schema is not None:
                body["response_format"] = {"type": "json_object"}
            data = self._transport(body)
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise LlmError(f"响应结构异常：{str(data)[:200]}") from error
            if json_schema is None:
                return content
            try:
                return json_schema.model_validate_json(content)
            except ValidationError as error:
                last_error = error
                prompt = f"{user}\n\n注意：上次输出未通过 schema 校验（{error}）。请严格按要求的 JSON 格式输出，不要包含多余文本。"
        raise LlmError(f"结构化输出反复校验失败：{last_error}")
