"""向量客户端与缓存测试（transport 注入，全离线，不花一分钱）。"""

import json

import pytest

from priceradar.matching.embeddings import (
    BATCH_SIZE,
    EmbeddingCache,
    EmbeddingError,
    cosine,
    DashScopeEmbedder,
)


class FakeTransport:
    """按输入文本返回确定性向量；记录每次请求。"""

    def __init__(self, dim=4):
        self.dim = dim
        self.requests: list[dict] = []

    def __call__(self, body: dict) -> dict:
        self.requests.append(body)
        return {
            "data": [
                {"embedding": [float(len(text) % 7)] * self.dim}
                for text in body["input"]
            ]
        }


def make_embedder(tmp_path, **kwargs) -> tuple[DashScopeEmbedder, FakeTransport]:
    transport = FakeTransport()
    cache = EmbeddingCache(tmp_path / "cache.json")
    return DashScopeEmbedder("key", transport=transport, cache=cache, **kwargs), transport


# ---------- 基本行为 ----------

def test_embed_returns_one_vector_per_input_in_order(tmp_path):
    embedder, _ = make_embedder(tmp_path)
    texts = ["a", "bb", "ccc"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 3
    assert vectors[0][0] == float(len("a") % 7)
    assert vectors[1][0] == float(len("bb") % 7)


def test_embed_batches_by_ten(tmp_path):
    """接口硬限制每批 10 条——25 条文本必须拆成 3 次请求。"""
    embedder, transport = make_embedder(tmp_path)
    embedder.embed([f"文本{i}" for i in range(25)])
    assert len(transport.requests) == 3
    assert [len(request["input"]) for request in transport.requests] == [10, 10, 5]


def test_requests_carry_model_and_encoding_format(tmp_path):
    embedder, transport = make_embedder(tmp_path, model="text-embedding-v9")
    embedder.embed(["x"])
    assert transport.requests[0]["model"] == "text-embedding-v9"
    assert transport.requests[0]["encoding_format"] == "float"


def test_empty_input_makes_no_request(tmp_path):
    embedder, transport = make_embedder(tmp_path)
    assert embedder.embed([]) == []
    assert transport.requests == []


# ---------- 缓存 ----------

def test_second_call_hits_cache_and_costs_nothing(tmp_path):
    embedder, transport = make_embedder(tmp_path)
    first = embedder.embed(["同一个标题", "另一个"])
    calls_after_first = embedder.calls

    second = embedder.embed(["同一个标题", "另一个"])
    assert second == first
    assert embedder.calls == calls_after_first      # 没有再发请求
    assert len(transport.requests) == 1


def test_partial_cache_only_requests_the_missing(tmp_path):
    embedder, transport = make_embedder(tmp_path)
    embedder.embed(["已有的"])
    embedder.embed(["已有的", "新的"])
    assert len(transport.requests) == 2
    assert transport.requests[1]["input"] == ["新的"]


def test_cache_survives_a_new_client(tmp_path):
    path = tmp_path / "cache.json"
    first, _ = make_embedder(tmp_path)
    first.embed(["标题"])
    first.cache.flush(force=True)

    reopened = EmbeddingCache(path)
    assert len(reopened) == 1
    assert reopened.get("标题", first.model) is not None


def test_cache_key_uses_the_truncated_text(tmp_path):
    """缓存键必须与实际送出去的文本一致。

    曾经「读取用全文、写入用截断文」，结果超长文本永远不命中——每次都白花钱。
    """
    from priceradar.matching.embeddings import MAX_CHARS

    embedder, transport = make_embedder(tmp_path)
    long_text = "长" * (MAX_CHARS + 500)
    embedder.embed([long_text])
    embedder.embed([long_text])
    assert len(transport.requests) == 1
    assert len(transport.requests[0]["input"][0]) == MAX_CHARS


def test_corrupt_cache_file_is_ignored(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{不是合法 JSON", encoding="utf-8")
    assert len(EmbeddingCache(path)) == 0


def test_cache_flush_writes_readable_json(tmp_path):
    embedder, _ = make_embedder(tmp_path)
    embedder.embed(["x"])
    payload = json.loads(embedder.cache.path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict) and len(payload) == 1


# ---------- 失败路径 ----------

def test_missing_api_key_raises(monkeypatch):
    import priceradar.matching.embeddings as module

    monkeypatch.setattr(module, "load_env", lambda *names: {})
    with pytest.raises(EmbeddingError, match="DASHSCOPE_API_KEY"):
        DashScopeEmbedder.from_env()


def test_malformed_response_raises(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.json")
    embedder = DashScopeEmbedder("k", transport=lambda body: {"oops": True}, cache=cache)
    with pytest.raises(EmbeddingError, match="响应结构异常"):
        embedder.embed(["x"])


def test_response_count_mismatch_raises(tmp_path):
    """返回条数对不上会静默错位——必须当场炸，不能悄悄算错。"""
    cache = EmbeddingCache(tmp_path / "c.json")
    embedder = DashScopeEmbedder("k", transport=lambda body: {"data": [{"embedding": [0.0]}]}, cache=cache)
    with pytest.raises(EmbeddingError, match="返回条数不符"):
        embedder.embed(["a", "b"])


# ---------- 余弦 ----------

def test_cosine_identical_is_one():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_is_minus_one():
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_does_not_divide_by_zero():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="维度不一致"):
        cosine([1.0], [1.0, 2.0])


def test_batch_size_matches_api_limit():
    assert BATCH_SIZE == 10


# ---------- 护栏：向量层不得进入判定路径 ----------

def test_matcher_does_not_depend_on_embeddings():
    """`evaluation/results_vector.md` 用数据得出的结论：向量不得参与同款判定。

    实测：纯向量最优阈值下准确率 70.2%（规则匹配器同一集合上 100%），
    且换型号的负例余弦（最高 0.96）**高于**堆了促销词的真同款（最低 0.91）
    ——两者分布重叠，向量分不清「同一个东西的不同说法」与「长得像的另一个东西」。

    这条测试把这个结论钉住：日后若有人想把向量接进 matcher，会先在这里失败，
    必须回到 results_vector.md 重新论证。
    """
    from pathlib import Path

    matcher_source = (
        Path(__file__).resolve().parent.parent / "src" / "priceradar" / "matching" / "matcher.py"
    ).read_text(encoding="utf-8")
    assert "embedding" not in matcher_source.lower()
    assert "cosine" not in matcher_source.lower()


def test_vector_probe_pure_functions_are_offline_testable():
    """测量脚本的核心计算（阈值扫描、分位数）不依赖网络，可离线复算。"""
    from evaluation.vector_probe import _stats, sweep_threshold

    stats = _stats([0.1, 0.2, 0.3, 0.4])
    assert stats["n"] == 4 and stats["min"] == 0.1 and stats["max"] == 0.4
    assert _stats([]) == {"n": 0}

    # 三对：两正一负，正对分数高 → 高阈值下应完美分开
    rows = sweep_threshold([(0.9, True), (0.95, True), (0.2, False)])
    perfect = [row for row in rows if row["accuracy"] == 1.0]
    assert perfect, "完全可分的数据必须存在一个 100% 准确的阈值"
    assert all(row["precision"] == 1.0 and row["recall"] == 1.0 for row in perfect)


def test_sweep_threshold_counts_add_up():
    from evaluation.vector_probe import sweep_threshold

    pairs = [(0.9, True), (0.8, True), (0.3, False), (0.4, False), (0.85, True)]
    for row in sweep_threshold(pairs):
        assert row["tp"] + row["fn"] == 3        # 正类共 3 个
        assert row["fp"] + row["tn"] == 2        # 负类共 2 个
        assert row["tp"] + row["fp"] + row["fn"] + row["tn"] == len(pairs)
