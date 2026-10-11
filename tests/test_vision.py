"""识图模块测试（transport 注入，全离线；VLM 行为用脚本化响应模拟）。"""

from decimal import Decimal

import pytest

from priceradar.domain.models import CouponType
from priceradar.vision import (
    CouponExtraction,
    ProductAttrs,
    QwenVlClient,
    VisionError,
    extract_coupons,
    extract_product,
    to_data_url,
    to_offer_coupons,
)
from priceradar.vision.vlm import _format_http_error

FAKE_IMAGE = b"\xff\xd8\xff\xe0fake-jpeg-bytes"


def fake_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class ScriptedTransport:
    def __init__(self, *responses: dict) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, body: dict) -> dict:
        self.calls.append(body)
        return self.responses.pop(0)


# ---------- 基础 ----------

def test_to_data_url_embeds_base64():
    url = to_data_url(FAKE_IMAGE, "image/png")
    assert url.startswith("data:image/png;base64,")


def test_from_env_raises_without_key(monkeypatch):
    monkeypatch.setattr("priceradar.vision.vlm.load_env", lambda *names: {"DASHSCOPE_API_KEY": ""})
    with pytest.raises(VisionError):
        QwenVlClient.from_env()


# ---------- 模型名（会被平台下架，故走配置而非硬编码）----------

def test_default_model_is_the_current_dashscope_vision_model():
    """2026-10-10：`qwen-vl-max`/`qwen-vl-plus` 已被 DashScope 下架，调用 404 直接让识图全废。

    这条测试钉住默认值——换代时要连着改这里，不能只改代码不更新认识。
    """
    assert QwenVlClient("k").model == "qwen3-vl-plus"


def test_vl_model_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setattr(
        "priceradar.vision.vlm.load_env",
        lambda *names: {"DASHSCOPE_API_KEY": "k", "DASHSCOPE_VL_MODEL": "qwen-next-vl"},
    )
    assert QwenVlClient.from_env().model == "qwen-next-vl"


def test_blank_model_env_falls_back_to_default(monkeypatch):
    """`.env` 里留空的 `DASHSCOPE_VL_MODEL=` 不该把模型名变成空串（会 404 得更莫名其妙）。"""
    monkeypatch.setattr(
        "priceradar.vision.vlm.load_env",
        lambda *names: {"DASHSCOPE_API_KEY": "k", "DASHSCOPE_VL_MODEL": ""},
    )
    assert QwenVlClient.from_env().model == "qwen3-vl-plus"


def test_model_not_found_404_explains_the_fix():
    """模型下架时的报错必须给出处方——否则用户只看到一坨 HTTP 404 的 JSON。"""
    body = '{"error":{"message":"Model not exist.","code":"model_not_found"}}'
    message = _format_http_error(404, body)
    assert "DASHSCOPE_VL_MODEL" in message
    assert "qwen3-vl-plus" in message


def test_other_http_errors_do_not_get_the_model_hint():
    """别的错（限流、欠费）不该被误诊成"模型下架"，那会把人带到错误的方向。"""
    assert "DASHSCOPE_VL_MODEL" not in _format_http_error(429, '{"error":{"code":"throttled"}}')
    assert "DASHSCOPE_VL_MODEL" not in _format_http_error(404, '{"error":{"code":"not_found"}}')


def test_request_body_carries_image_and_schema_prompt():
    transport = ScriptedTransport(fake_response('{"is_product": true, "keywords": ["QCY N20"]}'))
    client = QwenVlClient("k", transport=transport)
    extract_product(FAKE_IMAGE, client=client)
    body = transport.calls[0]
    user_content = body["messages"][1]["content"]
    assert user_content[0]["type"] == "image_url"
    assert user_content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "只输出 JSON" in body["messages"][0]["content"]


# ---------- 商品图识别 ----------

def test_extract_product_parses_fields():
    transport = ScriptedTransport(
        fake_response(
            '{"is_product": true, "category": "耳机", "brand": "QCY", "model": "N20", '
            '"specs": {"续航": "28小时"}, "color": "白色", "keywords": ["QCY N20", "降噪耳机"], '
            '"uncertain_fields": ["color"], "confidence": 0.82}'
        )
    )
    result = extract_product(FAKE_IMAGE, client=QwenVlClient("k", transport=transport))
    assert isinstance(result, ProductAttrs)
    assert result.model == "N20"
    assert result.keywords == ["QCY N20", "降噪耳机"]
    assert result.confidence == pytest.approx(0.82)


def test_extract_product_retries_after_invalid_json():
    transport = ScriptedTransport(
        fake_response("这不是JSON"), fake_response('{"is_product": true, "keywords": []}')
    )
    client = QwenVlClient("k", transport=transport)
    result = extract_product(FAKE_IMAGE, client=client)
    assert result.is_product is True
    assert len(transport.calls) == 2
    assert "校验" in transport.calls[1]["messages"][1]["content"][1]["text"]


def test_extract_product_raises_after_repeated_invalid():
    transport = ScriptedTransport(fake_response("x"), fake_response("y"))
    with pytest.raises(VisionError):
        extract_product(FAKE_IMAGE, client=QwenVlClient("k", transport=transport))


def test_non_product_image_passes_through():
    transport = ScriptedTransport(fake_response('{"is_product": false, "keywords": []}'))
    result = extract_product(FAKE_IMAGE, client=QwenVlClient("k", transport=transport))
    assert result.is_product is False  # 由调用方负责追问"换一张图"


# ---------- 券截图识别 ----------

def test_extract_coupons_and_map_to_domain():
    transport = ScriptedTransport(
        fake_response(
            '{"platform": "jd", "coupons": ['
            '{"type": "platform", "face_value": 50, "threshold": 300, "scope": "全平台", '
            '"valid_until": "2026-11-11", "confidence": 0.9},'
            '{"type": "shop", "face_value": 20, "threshold": 0, "scope": "XX旗舰店", '
            '"valid_until": null, "confidence": 0.7}]}'
        )
    )
    extraction = extract_coupons(FAKE_IMAGE, client=QwenVlClient("k", transport=transport))
    assert isinstance(extraction, CouponExtraction)
    coupons = to_offer_coupons(extraction)
    assert len(coupons) == 2
    first = coupons[0]
    assert first.type is CouponType.PLATFORM
    assert first.face_value == Decimal("50") and first.threshold == Decimal("300")
    assert first.valid_until is not None and first.valid_until.isoformat() == "2026-11-11"
    assert first.source == "screenshot"
    assert first.confidence == pytest.approx(0.9)
    assert coupons[1].threshold == Decimal("0") and coupons[1].valid_until is None


def test_bad_valid_until_falls_back_to_none():
    extraction = CouponExtraction.model_validate(
        {"platform": "pdd", "coupons": [{"type": "shop", "face_value": 5, "valid_until": "下个月"}]}
    )
    coupons = to_offer_coupons(extraction)
    assert coupons[0].valid_until is None


def test_coupon_schema_rejects_unknown_type():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CouponExtraction.model_validate({"platform": "jd", "coupons": [{"type": "magic", "face_value": 5}]})
