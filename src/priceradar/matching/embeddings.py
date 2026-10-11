"""文本向量（DashScope 兼容 OpenAI 的 embeddings 接口）+ 磁盘缓存。

**定位：只服务于"测量"，尚未进主链路。** 计划里写的是「先测量再决定」——
先量清楚向量对同款匹配到底有没有增益，再决定要不要接进判定路径。

⚠️ 与 PRD / `docs/resume-strategy.md` 的叙事冲突：那两份文档承诺的是
**bge-small-zh 本地**（延迟/成本/隐私）。这里用 DashScope 是为了**做实验**，
不是最终选型；真要进主链路得先统一说法（见 docs/agent-upgrade-plan.md 批 ⑦）。

费用控制：
- 结果按文本 sha1 落盘缓存（`data/embedding_cache.json`，gitignored）
- 每批最多 10 条（接口硬限制），超长文本先截断（单条上限 8192 token）
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from priceradar.config import load_env, project_root

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
DEFAULT_MODEL = "text-embedding-v4"
BATCH_SIZE = 10          # 接口硬限制：每批最多 10 条
MAX_CHARS = 2000         # 商品标题远用不到；截断只为防意外超长输入
CACHE_PATH = project_root() / "data" / "embedding_cache.json"


class EmbeddingError(RuntimeError):
    pass


def _digest(text: str, model: str) -> str:
    return hashlib.sha1(f"{model}\u0000{text}".encode("utf-8")).hexdigest()


class EmbeddingCache:
    """极简磁盘缓存：{sha1: [float, ...]}。坏了就重来，不阻塞实验。

    `persist=False` 时只做内存缓存、完全不碰磁盘（`--no-cache` 用）。
    ⚠️ 别用「把 path 传成 None」来表达这个意思——`None` 会被解析回默认路径，
    结果既没绕过缓存、又会把已有缓存文件删掉。
    """

    def __init__(self, path: Path | None = None, *, persist: bool = True) -> None:
        self.path = Path(path or CACHE_PATH)
        self.persist = persist
        self._data: dict[str, list[float]] = {}
        self._dirty = 0
        self.load()

    def load(self) -> None:
        if not self.persist or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._data = raw if isinstance(raw, dict) else {}
        except (json.JSONDecodeError, OSError):
            self._data = {}          # 缓存坏了不是错误，重算就好

    def get(self, text: str, model: str) -> list[float] | None:
        return self._data.get(_digest(text, model))

    def put(self, text: str, model: str, vector: list[float]) -> None:
        self._data[_digest(text, model)] = vector
        self._dirty += 1

    def flush(self, force: bool = False) -> None:
        if not self.persist or (not self._dirty and not force):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")
        self._dirty = 0

    def __len__(self) -> int:
        return len(self._data)


class DashScopeEmbedder:
    """transport 可注入（测试离线化）：callable(body: dict) -> 响应 dict。"""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        transport: Callable[[dict], dict] | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._transport = transport or self._default_transport
        self.cache = cache if cache is not None else EmbeddingCache()
        self.calls = 0               # 真实发出的请求数（缓存命中不计）——便于核算花费

    @classmethod
    def from_env(cls, **kwargs) -> "DashScopeEmbedder":
        api_key = load_env("DASHSCOPE_API_KEY").get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise EmbeddingError("未配置 DASHSCOPE_API_KEY（环境变量或 .env）")
        return cls(api_key, **kwargs)

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
            raise EmbeddingError(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:200]}") from error
        except urllib.error.URLError as error:
            raise EmbeddingError(f"网络错误：{error.reason}") from error

    def embed(self, texts: list[str]) -> list[list[float]]:
        """按顺序返回向量（与输入一一对应）。缓存命中不发请求。"""
        # 先统一截断，再算缓存键——否则「缓存键用全文、写入用截断文」会永远不命中
        prepared = [text[:MAX_CHARS] for text in texts]
        results: list[list[float] | None] = [self.cache.get(text, self.model) for text in prepared]
        pending = [(index, prepared[index]) for index, value in enumerate(results) if value is None]

        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start : start + BATCH_SIZE]
            data = self._transport(
                {"model": self.model, "input": [text for _, text in batch], "encoding_format": "float"}
            )
            self.calls += 1
            try:
                items = data["data"]
            except (KeyError, TypeError) as error:
                raise EmbeddingError(f"响应结构异常：{str(data)[:200]}") from error
            if len(items) != len(batch):
                raise EmbeddingError(f"返回条数不符：请求 {len(batch)} 条，返回 {len(items)} 条")
            for (index, text), item in zip(batch, items):
                vector = item["embedding"]
                results[index] = vector
                self.cache.put(text, self.model, vector)

        self.cache.flush()
        if any(vector is None for vector in results):
            raise EmbeddingError("有文本未取得向量（缓存与批次逻辑异常）")
        return [vector for vector in results if vector is not None]


def cosine(left: list[float], right: list[float]) -> float:
    """余弦相似度（零向量返回 0，不做除零）。"""
    if len(left) != len(right):
        raise ValueError(f"维度不一致：{len(left)} vs {len(right)}")
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)
