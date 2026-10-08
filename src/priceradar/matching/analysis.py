"""标题分析与规格提取（确定性规则；向量层在后续版本替换/补充相似度实现）。"""

from __future__ import annotations

import re

from priceradar.domain.models import Offer
from priceradar.matching.lexicon import (
    ACCESSORY_MARKERS,
    BRAND_ALIASES,
    CAPACITY_PATTERN,
    COLOR_PATTERN,
    FIT_PREFIXES,
    GENERATION_PATTERN,
    SIZE_PATTERN,
    _CN_NUMERALS,
    _MODEL_UNIT_RE,
)
from priceradar.matching.models import SpecProfile

_FW_TABLE = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
)
_PUNCT_RE = re.compile(r"[【】\[\]（）()·—\-_/｜|,，。.!！?？:：;；'\"“”]+")
_MODEL_RUN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")
_IPHONE_RE = re.compile(r"(?:iphone|苹果)\s*(\d{1,2})", re.IGNORECASE)
_FIT_CONTEXT = ("适用", "适配", "兼容", "支持")  # 兼容性语境（"适用XX手机"不是品牌/型号归属）


def _in_fit_context(title: str, index: int) -> bool:
    """判断某个匹配位置是否处于"适用/适配/兼容"语境（前面 6 个字符内出现标记词）。"""
    window = title[max(0, index - 6) : index]
    return any(marker in window for marker in _FIT_CONTEXT)


def normalize_title(title: str) -> str:
    """全角转半角、去促销标点、小写、压缩空白。"""
    text = title.translate(_FW_TABLE)
    text = _PUNCT_RE.sub(" ", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def is_accessory(title: str) -> bool:
    """配件/周边判定：以"适用/适配/兼容"开头，或含强标记词。"""
    stripped = title.translate(_FW_TABLE).strip()
    if stripped.startswith(FIT_PREFIXES):
        return True
    return any(marker in stripped for marker in ACCESSORY_MARKERS)


# 平台的"没有品牌"占位值。拼多多大量返回 "无品牌/无注册商标"（卖家未注册商标），
# 必须识别为"无品牌信息"——否则它会被当成一个真品牌，与源商品的真实品牌冲突，
# 导致**最便宜的未注册卖家被整批误排除**（与产品目标相反）。
_BRAND_PLACEHOLDERS = frozenset(
    {"无品牌", "无注册商标", "无商标", "无", "其他", "其它", "unknown", "none", "n/a", "-"}
)


def canonicalize_brand(raw: str | None) -> str | None:
    """平台提供的结构化品牌 → canonical 键（与标题识别共用别名表）。

    - "Edifier/漫步者" → edifier（命中别名）；"无品牌"/纯数字代码 → None
    - 未知品牌保留首段规范化文本（如 "YINPAGE/音贝奇" → "yinpage"）
    """
    if not raw:
        return None
    text = raw.translate(_FW_TABLE).strip().lower()
    if not text or text in _BRAND_PLACEHOLDERS:
        return None  # 整串判定要在拆分之前：否则 "N/A" 会被斜杠拆成 ["n","a"] 侥幸漏过
    # 逐段判定占位值：斜杠分隔的各段**全是**占位值才算"无品牌"
    # （"无品牌/无注册商标" → None；"Edifier/漫步者" → 有实义段，继续）
    segments = [segment.strip() for segment in text.split("/") if segment.strip()]
    if not segments or all(segment in _BRAND_PLACEHOLDERS for segment in segments):
        return None
    if text.isdigit():
        return None
    for canonical, aliases in BRAND_ALIASES.items():
        if any(alias in text for alias in aliases):
            return canonical
    meaningful = [segment for segment in segments if segment not in _BRAND_PLACEHOLDERS]
    return meaningful[0] if meaningful else None


def detect_brand(title: str) -> str | None:
    """返回标题中最早真实出现的品牌（忽略"适用/适配"等兼容性语境）。"""
    base = title.translate(_FW_TABLE)
    lowered = base.lower()
    best: str | None = None
    best_pos = len(lowered) + 1
    for canonical, aliases in BRAND_ALIASES.items():
        for alias in aliases:
            start = 0
            while True:
                index = lowered.find(alias, start)
                if index < 0:
                    break
                if not _in_fit_context(base, index):
                    if index < best_pos:
                        best, best_pos = canonical, index
                    break
                start = index + 1
    return best


def _extract_models(title: str) -> set[str]:
    models: set[str] = set()
    for run in _MODEL_RUN_RE.findall(title):
        for part in run.split("-"):
            token = part.upper()
            if len(token) < 2 or not re.search(r"\d", token):
                continue
            if re.fullmatch(r"20\d{2}", token):  # 年份（2026新款）不是型号
                continue
            if re.fullmatch(_MODEL_UNIT_RE, token):  # 128G/40W/5MM 等是单位不是型号
                continue
            if re.fullmatch(r"\d+", token):  # 纯数字（个数/年份残片）不是型号
                continue
            models.add(token)
    for match in _IPHONE_RE.finditer(title):  # "苹果16 / iPhone 16" → IPHONE16
        if _in_fit_context(title, match.start()):  # "适用苹果17" 是兼容性说明，不是型号
            continue
        models.add(f"IPHONE{match.group(1)}")
    return models


def _extract_generations(title: str) -> set[str]:
    generations: set[str] = set()
    for raw in re.findall(GENERATION_PATTERN, title):
        generations.add(f"{_CN_NUMERALS.get(raw, raw)}代")
    return generations


def extract_specs(title: str) -> SpecProfile:
    """从标题提取规格画像（缺失即为空集合，不做猜测）。

    全角字符先归一半角（评测集 v1 发现：全角 Ｎ２０ 会导致型号提取全部失效）。
    """
    title = title.translate(_FW_TABLE)
    capacities: set[str] = set()
    for num, unit in re.findall(CAPACITY_PATTERN, title, re.IGNORECASE):
        if unit.upper() == "G" and num in {"3", "4", "5"}:  # 5G/4G 为网络制式，不是容量
            continue
        capacities.add(f"{num}{'T' if unit.upper().startswith('T') else 'G'}")
    sizes = {a or f"{b}码" for a, b in re.findall(SIZE_PATTERN, title)}
    colors = set(re.findall(COLOR_PATTERN, title))
    return SpecProfile(
        models=_extract_models(title),
        capacities=capacities,
        sizes=sizes,
        colors=colors,
        generations=_extract_generations(title),
        brand=detect_brand(title),
        is_accessory=is_accessory(title),
    )


def specs_for(offer: Offer) -> SpecProfile:
    """Offer → 规格画像：平台结构化品牌优先，标题识别兜底。

    平台给的 brand_name 比从标题猜更可靠（拼多多 "Edifier/漫步者" 这种双语名
    直接命中别名表；标题里品牌被省略/写促销词时也能拿到）。拿不到时才退回标题。
    """
    profile = extract_specs(offer.title)
    structured = canonicalize_brand(offer.brand)
    if structured and structured != profile.brand:
        return profile.model_copy(update={"brand": structured})
    return profile


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def similarity(a: str, b: str) -> float:
    """标题相似度（字符 bigram Dice，0-1）。

    占位实现：v1 以规格硬规则为主证据，本分数仅作 sanity check；
    后续以评测集校准阈值，并引入 bge-small-zh 向量层（PRD §5.7 第 2 层）。
    """
    ga, gb = _bigrams(normalize_title(a)), _bigrams(normalize_title(b))
    if not ga or not gb:
        return 0.0
    return 2 * len(ga & gb) / (len(ga) + len(gb))
