"""同款匹配模块测试（样本取自本会话实测的真实平台标题）。"""

from decimal import Decimal

from priceradar.domain.models import Offer, Platform
from priceradar.matching import (
    MatchTier,
    canonicalize_brand,
    detect_brand,
    extract_specs,
    is_accessory,
    match_offers,
    normalize_title,
    similarity,
    specs_for,
)

D = Decimal

# —— 真实标题样本（拼多多/京东实测）——
PDD_QCY = "【QCY】 N20降噪蓝牙耳机无线入耳式2026新款长续航游戏适用小米苹果"
JD_QCY = "QCY N20 主动降噪蓝牙耳机 无线入耳式 超长续航 通话降噪 2026新款"
ACC_SONY = "适用跨境索尼WF-1000XM5保护套SONY耳机壳防摔减震收纳软盒仓tpu"
ACC_JD = "适用于SONY索尼WH-1000XM4头戴式蓝牙耳机保护套WH-1000XM3/2横梁头梁套"
WIRED_EARPHONE = "爱国者线控有线耳机 3.5mm圆孔圆头手机耳机 带线睡眠游戏"
IPHONE_128 = "Apple iPhone 16 128GB 黑色 5G手机"
IPHONE_256 = "苹果iPhone 16 256G 黑色手机"


def make_offer(sku="sku-1", title="测试", price="100", platform=Platform.PDD) -> Offer:
    return Offer(platform=platform, sku_id=sku, title=title, list_price=D(price))


# ---------- 标题归一化 ----------

def test_normalize_fullwidth_and_punct():
    assert normalize_title("【ＱＣＹ】Ｎ２０！") == "qcy n20"
    assert normalize_title("QCY   N20   降噪") == "qcy n20 降噪"


# ---------- 规格提取 ----------

def test_extract_capacity_variants():
    assert extract_specs("iPhone 16 256GB 黑色").capacities == {"256G"}
    assert extract_specs("游戏本 1T 固态").capacities == {"1T"}


def test_extract_models():
    assert "N20" in extract_specs(PDD_QCY).models
    assert "1000XM6" in extract_specs("WH-1000XM6 降噪").models
    assert "IPHONE16" in extract_specs("苹果16 手机").models


def test_year_and_units_are_not_models():
    assert extract_specs("2026新款蓝牙耳机").models == set()
    assert extract_specs("3.5mm 耳机 40W快充").models == set()


def test_extract_generation_and_color():
    specs = extract_specs("AirPods 第4代 白色")
    assert specs.generations == {"4代"}
    assert specs.colors == {"白"}


def test_blacktech_is_not_color():
    assert extract_specs("黑科技降噪耳机").colors == set()


def test_brand_aliases():
    assert detect_brand("SONY WH-1000XM6") == "sony"
    assert detect_brand("索尼降噪耳机") == "sony"
    assert detect_brand("Apple AirPods") == "apple"
    assert detect_brand("苹果手机壳") == "apple"
    assert detect_brand("某不知名品牌耳机") is None


def test_extract_size_tokens():
    specs = extract_specs("卫生巾 ML/XL码 学生")
    assert specs.sizes == {"ML", "XL"}


# ---------- 配件判定（真实样本） ----------

def test_accessory_real_titles():
    assert is_accessory(ACC_SONY) is True
    assert is_accessory(ACC_JD) is True
    assert is_accessory("手机数据线 快充") is True


def test_wired_earphone_is_not_accessory():
    assert is_accessory(WIRED_EARPHONE) is False
    assert is_accessory(PDD_QCY) is False


# ---------- 相似度 ----------

def test_similarity_high_for_variants_low_for_unrelated():
    assert similarity(PDD_QCY, JD_QCY) > 0.25
    assert similarity("QCY N20 降噪耳机", "婚纱摄影套餐") < 0.05


# ---------- 匹配分档 ----------

def test_same_model_across_platforms_is_confirmed():
    report = match_offers(make_offer(sku="a", title=PDD_QCY), [make_offer(sku="b", title=JD_QCY, platform=Platform.JD)])
    assert len(report.results) == 1
    assert report.results[0].tier is MatchTier.CONFIRMED
    assert "N20" in report.results[0].reasons[0]


def test_capacity_conflict_is_similar_with_diff_note():
    report = match_offers(make_offer(title=IPHONE_128), [make_offer(sku="b", title=IPHONE_256, platform=Platform.JD)])
    assert report.results[0].tier is MatchTier.SIMILAR
    assert "容量不同" in report.results[0].reasons[0]


def test_model_conflict_is_similar():
    report = match_offers(make_offer(title="QCY N20 降噪耳机"), [make_offer(sku="b", title="QCY N60 降噪耳机")])
    assert report.results[0].tier is MatchTier.SIMILAR
    assert "型号不同" in report.results[0].reasons[0]


def test_candidate_without_model_is_similar():
    report = match_offers(make_offer(title=PDD_QCY), [make_offer(sku="b", title="QCY 降噪蓝牙耳机 无线")])
    assert report.results[0].tier is MatchTier.SIMILAR
    assert "缺少型号" in report.results[0].reasons[0]


def test_accessory_candidate_is_excluded():
    report = match_offers(make_offer(title="索尼 WH-1000XM6 降噪耳机"), [make_offer(sku="b", title=ACC_JD)])
    assert report.results == []
    assert report.excluded[0].reason == "配件/周边类商品"


def test_brand_conflict_is_excluded():
    report = match_offers(make_offer(title="索尼 WH-1000XM6 降噪耳机"), [make_offer(sku="b", title="Bose QC45 降噪耳机")])
    assert report.excluded[0].reason.startswith("品牌不同")


def test_suspicious_low_price_is_tier_c():
    report = match_offers(make_offer(title=JD_QCY, price="300"), [make_offer(sku="b", title=JD_QCY, price="100")])
    assert report.results[0].tier is MatchTier.SUSPICIOUS
    assert "价格异常低" in report.results[0].reasons[0]


def test_no_spec_source_is_capped_at_similar():
    report = match_offers(make_offer(title="XX 蓝牙耳机 无线 入耳式"), [make_offer(sku="b", title="XX 蓝牙耳机 无线 入耳式")])
    assert report.results[0].tier is MatchTier.SIMILAR
    assert "无可用规格佐证" in report.results[0].reasons[0]


def test_low_similarity_is_excluded():
    report = match_offers(make_offer(title="蓝牙耳机 无线 入耳式"), [make_offer(sku="b", title="蓝牙音箱 便携 低音炮")])
    assert report.results == []
    assert report.excluded[0].reason == "相似度过低"


def test_both_accessories_are_compared_normally():
    source = make_offer(title="适用AirPods保护套 硅胶 防摔")
    report = match_offers(source, [make_offer(sku="b", title="AirPods 保护套 透明 硅胶")])
    assert len(report.results) == 1  # 不当配件排除


def test_self_is_skipped():
    offer = make_offer(sku="same", title=PDD_QCY)
    report = match_offers(offer, [offer])
    assert report.results == [] and report.excluded == []


def test_results_sorted_by_tier_then_score():
    source = make_offer(title=JD_QCY, price="300")
    candidates = [
        make_offer(sku="c-susp", title=JD_QCY, price="50"),
        make_offer(sku="b-similar", title="QCY N60 降噪耳机", price="200"),
        make_offer(sku="a-confirmed", title=PDD_QCY, price="280"),
    ]
    report = match_offers(source, candidates)
    assert [r.tier for r in report.results] == [MatchTier.CONFIRMED, MatchTier.SIMILAR, MatchTier.SUSPICIOUS]


def test_match_is_deterministic():
    source = make_offer(title=JD_QCY)
    candidates = [make_offer(sku="c1", title=PDD_QCY), make_offer(sku="c2", title=ACC_JD)]
    assert match_offers(source, candidates) == match_offers(source, candidates)


def test_empty_candidates():
    report = match_offers(make_offer(), [])
    assert report.results == [] and report.excluded == []


# ---------- 回归：真实数据暴露的两个问题（2026-10-05） ----------

def test_fullwidth_title_extracts_same_specs():
    """回归（评测集 v1 发现）：全角标题不应丢失型号/品牌提取。"""
    original = "【QCY】 N20降噪蓝牙耳机无线入耳式长续航"
    fullwidth = "【ＱＣＹ】 Ｎ２０降噪蓝牙耳机无线入耳式长续航"
    specs = extract_specs(fullwidth)
    assert "N20" in specs.models
    assert specs.brand == "qcy"
    assert specs.models == extract_specs(original).models

def test_compat_context_is_not_model_or_brand():
    """'适用苹果17' 是兼容性说明，不能提取为型号；'适用小米' 不能算作品牌。"""
    specs = extract_specs("QCY Buds 降噪耳机 适用苹果17")
    assert "IPHONE17" not in specs.models
    assert "BUDS" not in specs.models or "N20" not in specs.models  # BUDS 无数字本就非型号
    assert detect_brand("降噪耳机 适用小米苹果") is None
    assert detect_brand("QCY 耳机 适用小米手机") == "qcy"
    assert detect_brand("Apple iPhone 16 手机") == "apple"


def test_unrelated_product_is_excluded_not_similar():
    """无关商品（灯管）不得落入"近似款"。"""
    report = match_offers(make_offer(title="QCY N20 降噪耳机"), [make_offer(sku="b", title="雷士 led长条灯管一体化 高显色灯条")])
    assert report.results == []
    assert report.excluded[0].reason == "相似度过低"


# —— 平台结构化品牌（拼多多 brand_name / 京东 brandName）——


def test_canonicalize_brand_maps_platform_values():
    """实测值："Edifier/漫步者" 这类双语名直接命中别名表；无品牌/数字代码不猜。"""
    assert canonicalize_brand("Edifier/漫步者") == "edifier"
    assert canonicalize_brand("QCY") == "qcy"
    assert canonicalize_brand("索尼") == "sony"


def test_canonicalize_brand_rejects_placeholders():
    assert canonicalize_brand("无品牌") is None
    assert canonicalize_brand("12345") is None
    assert canonicalize_brand("") is None
    assert canonicalize_brand(None) is None


def test_no_brand_placeholder_does_not_exclude_cheaper_sellers():
    """⚠️ 生产 bug 回归：拼多多大量返回 brand_name="无品牌/无注册商标"。

    若把它当成真品牌，源商品（有品牌）与候选（无品牌占位）会判"品牌冲突"而被排除——
    恰好把**最便宜的未注册商标卖家整批误杀**，与产品目标相反。
    评测集没抓到它，因为构造集只给标题、不设 brand 字段。
    """
    source = make_offer(title="QCY N20 降噪耳机").model_copy(update={"brand": "QCY"})
    candidate = make_offer(sku="b", title="QCY N20 降噪蓝牙耳机", price="95").model_copy(
        update={"brand": "无品牌/无注册商标"}
    )
    report = match_offers(source, [candidate])
    assert report.excluded == []
    assert len(report.results) == 1


def test_brand_placeholders_are_recognised():
    for placeholder in ("无品牌", "无注册商标", "无品牌/无注册商标", "其他", "无", "unknown", "N/A"):
        assert canonicalize_brand(placeholder) is None, placeholder


def test_placeholder_filter_does_not_eat_real_brands():
    """占位值判定要精确：不能把"无印良品"（MUJI）这类含"无"的真品牌一并干掉。"""
    assert canonicalize_brand("无印良品") is not None
    assert canonicalize_brand("无品牌/MUJI") == "muji"   # 取有实义的那一段


def test_canonicalize_brand_keeps_unknown_brand():
    """未收录品牌保留首段规范化文本，不丢弃（仍可用于冲突判断）。"""
    assert canonicalize_brand("YINPAGE/音贝奇") == "yinpage"


def test_specs_for_prefers_structured_brand_over_title():
    """标题里没有品牌时，平台结构化品牌补上——这是 brand_name 的最大价值。"""
    offer = make_offer(title="N20降噪蓝牙耳机无线入耳式2026新款").model_copy(update={"brand": "QCY"})
    assert extract_specs(offer.title).brand is None  # 标题识别不出来
    assert specs_for(offer).brand == "qcy"


def test_specs_for_falls_back_to_title_when_brand_missing():
    """平台没给品牌（None / "无品牌"）时，退回标题识别。"""
    assert specs_for(make_offer(title=PDD_QCY)).brand == "qcy"
    assert specs_for(make_offer(title=PDD_QCY).model_copy(update={"brand": "无品牌"})).brand == "qcy"


def test_brand_conflict_detected_via_structured_brand():
    """标题完全看不出品牌时，靠结构化品牌仍能排除非同款（否则会误判成 A 档）。"""
    source = make_offer(title="WH-1000XM5 头戴式蓝牙耳机").model_copy(update={"brand": "SONY"})
    candidate = make_offer(sku="b", title="WH-1000XM5 头戴式蓝牙耳机").model_copy(update={"brand": "Bose"})
    report = match_offers(source, [candidate])
    assert report.results == []
    assert "品牌不同" in report.excluded[0].reason
