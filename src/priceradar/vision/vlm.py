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

# DashScope **会下架旧模型名**：2026-10-10 实测 `qwen-vl-max` / `qwen-vl-plus` 已从
# 兼容端点的模型列表消失，调用直接 404 `model_not_found`（识图与券识别一起失效）。
# 故模型名走 `DASHSCOPE_VL_MODEL` 环境变量，换代时改配置即可，不必改代码。
# 可用列表：GET /compatible-mode/v1/models（需要 key）。
DEFAULT_MODEL = "qwen3-vl-plus"

_MODEL_NOT_FOUND_HINT = (
    "模型不存在——DashScope 会下架旧模型名。"
    "可用 DASHSCOPE_VL_MODEL 环境变量（或 .env）指定当前模型；"
    "当前已知可用：qwen3-vl-plus（通用识图）、qwen-vl-ocr（纯 OCR）；"
    "完整列表：GET https://dashscope.aliyuncs.com/compatible-mode/v1/models"
)


class VisionError(RuntimeError):
    pass


def _format_http_error(code: int, body: str) -> str:
    """把 HTTP 错误转成人话；模型下架是最常见的一种，单独给出处方。"""
    text = f"HTTP {code}: {body[:200]}"
    if code == 404 and "model_not_found" in body:
        return f"{text}\n{_MODEL_NOT_FOUND_HINT}"
    return text


def to_data_url(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


class QwenVlClient:
    """transport 可注入（测试离线化）：callable(body: dict) -> 响应 dict。"""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, transport: Callable[[dict], dict] | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self._transport = transport or self._default_transport

    @classmethod
    def from_env(cls) -> "QwenVlClient":
        env = load_env("DASHSCOPE_API_KEY", "DASHSCOPE_VL_MODEL")
        api_key = env.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise VisionError("未配置 DASHSCOPE_API_KEY（环境变量或 .env；阿里云百炼控制台申请）")
        return cls(api_key, model=env.get("DASHSCOPE_VL_MODEL") or DEFAULT_MODEL)

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
            raise VisionError(_format_http_error(error.code, error.read().decode("utf-8", "replace"))) from error
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
