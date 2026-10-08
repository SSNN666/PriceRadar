"""Qwen-VL 客户端（DashScope OpenAI 兼容模式，零第三方依赖）。

用途（PRD §5.5 / §5.6）：商品图识别、券截图识别。
凭证：DASHSCOPE_API_KEY（环境变量 > .env）。
隐私（PIPL 最小化）：图片以 data URL 内联直接送模型，服务端不落盘原图。

边界：VLM 只做"看"（结构化提取），产物必须经用户确认后才进入检索/计算。
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from priceradar.config import load_env

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


class VisionError(RuntimeError):
    pass


def to_data_url(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


class QwenVlClient:
    """transport 可注入（测试离线化）：callable(body: dict) -> 响应 dict。"""

    def __init__(self, api_key: str, model: str = "qwen-vl-max", transport: Callable[[dict], dict] | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self._transport = transport or self._default_transport

    @classmethod
    def from_env(cls) -> "QwenVlClient":
        api_key = load_env("DASHSCOPE_API_KEY").get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise VisionError("未配置 DASHSCOPE_API_KEY（环境变量或 .env；阿里云百炼控制台申请）")
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
            raise VisionError(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:200]}") from error
        except urllib.error.URLError as error:
            raise VisionError(f"网络错误：{error.reason}") from error

    def complete_json(
        self,
        system: str,
        user_text: str,
        image_bytes: bytes,
        *,
        json_schema: type[BaseModel],
        media_type: str = "image/jpeg",
        max_retries: int = 1,
    ) -> BaseModel:
        """图 + 提示词 → 经 Pydantic 校验的结构化输出（失败重试一次）。"""
        data_url = to_data_url(image_bytes, media_type)
        prompt = user_text
        last_error: Exception | None = None
        for _ in range(max_retries + 1):
            body: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                "temperature": 0.1,
            }
            data = self._transport(body)
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise VisionError(f"响应结构异常：{str(data)[:200]}") from error
            try:
                return json_schema.model_validate_json(content)
            except ValidationError as error:
                last_error = error
                prompt = f"{user_text}\n\n注意：上次输出未通过 schema 校验（{error}）。请严格按要求的 JSON 格式重新输出，不要包含多余文本。"
        raise VisionError(f"结构化输出反复校验失败：{last_error}")
