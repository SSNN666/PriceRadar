"""生成评测集 v1：真实标题种子 × 确定性变换（ground truth by construction）。

方法论（诚实声明，引用时请保留）：
- 种子 = 2026-10-05 实测抓取的拼多多/京东真实商品标题
- 变换均为语义明确的确定性操作，每对的期望输出（A / B / excluded）由变换语义预先给定
- 目的：检验规则层在"已知错误类型"（型号/容量/配件/品牌/无关/价格异常）上的表现，
  并按类别暴露强弱（不做单一聚合分掩盖问题）
- 局限：不覆盖开放世界的未知错误模式；PRD §5.7 计划的 200-500 对人工标注集为 v2
- 已知缺口：颜色差异当前未拦截（v1 不构造颜色负例，避免用已知缺口污染基线）

输出：evaluation/dataset_v1.jsonl
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

OUTPUT = Path(__file__).parent / "dataset_v1.jsonl"
ALLOWED_EXPECTED = {"A", "B", "C", "excluded"}

# —— 真实标题种子（2026-10-05 实测；has_specs=标题中可提取出规格锚点）——
SEEDS: list[tuple[str, bool]] = [
    ("【QCY】 N20降噪蓝牙耳机无线入耳式2026新款长续航游戏适用小米苹果", True),
    ("【QCY】 N60主动降噪蓝牙耳机无线入耳式AI翻译2026新款适用苹果17", True),
    ("【QCY】 A30八麦降噪豆真无线蓝牙耳机入耳式耳塞TWS运动游戏2026新款", True),
    ("QCY MeloBuds Pro真无线主动降噪耳机全能舱蓝牙耳机", False),
    ("漫步者（EDIFIER）X Clip耳夹式蓝牙耳机 翻译耳机", False),
    ("iKF Nano 空间音频主动降噪头戴式耳机无线电竞游戏适用", False),
    ("索尼 WH-1000XM6 头戴式降噪耳机", True),
    ("爱国者线控有线耳机 3.5mm圆孔圆头手机耳机 带线睡眠游戏", False),
    ("华强北新款Air4代降噪蓝牙耳机无线迷你运动四代适用于苹果", True),
    ("Apple iPhone 16 128GB 黑色 5G手机", True),
    ("鄱湖鸭舍 正宗鄱阳湖特大松花蛋溏心无铅工艺皮蛋 大号50-60克*16枚", False),
    ("雷士（NVC）led长条灯管一体化高显色灯条仓库厂房", False),
    ("京鲜生 福建琯溪蜜柚净重10斤 大小果混发 源头直发", False),
    ("得力（deli）学生可自动进笔手摇卷笔刀削笔机铅笔", False),
    ("蔡司擦镜纸 眼镜擦拭湿巾相机镜头镜片清洁湿巾 一次性", False),
    ("适用于SONY索尼WH-1000XM4头戴式蓝牙耳机保护套WH-1000XM3/2横梁头梁套", True),
    ("适用跨境索尼WF-1000XM5保护套SONY耳机壳防摔减震收纳软盒仓tpu", True),
    ("手机数据线 快充 适用华为小米苹果", False),
]

SEEDS_OTHER = [title for title, _ in SEEDS[10:15]]  # 跨品类无关种子

_FULLWIDTH_TABLE = str.maketrans(
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
)


def _safe_rotation(title: str) -> str | None:
    """在字符边界处旋转标题，保证不切断 ASCII token 及其配对（如 Air4 / iPhone 16）。

    优先切在中文边界（中文不参与任何 ASCII 型号配对）；纯 ASCII 标题则向左避开字母数字串。
    """
    index = len(title) // 2
    for position in range(index, len(title)):
        if not title[position].isascii():  # 中文边界
            index = position
            break
    else:
        while index > 0 and title[index].isascii() and title[index].isalnum() and title[index - 1].isascii() and title[index - 1].isalnum():
            index -= 1
    if index <= 0 or index >= len(title):
        return None
    return title[index:] + " " + title[:index]


@dataclass
class Record:
    id: str
    kind: str
    source_title: str
    candidate_title: str
    expected: str
    note: str
    provenance: str = "constructed"
    source_price: str = "100"
    candidate_price: str = "100"


def _add(
    records: list[Record],
    kind: str,
    source: str,
    candidate: str,
    expected: str,
    note: str,
    *,
    source_price: str = "100",
    candidate_price: str = "100",
) -> None:
    records.append(
        Record(
            id=f"{kind}-{len(records):04d}",
            kind=kind,
            source_title=source,
            candidate_title=candidate,
            expected=expected,
            note=note,
            source_price=source_price,
            candidate_price=candidate_price,
        )
    )


def build_records() -> list[Record]:
    records: list[Record] = []

    # —— 正类：同商品、确定性变换（有规格 → 期望 A；无规格 → 按标品声明上限 B）——
    for title, has_specs in SEEDS:
        expected = "A" if has_specs else "B"
        reason = "" if has_specs else "（无规格锚点，按标品范围声明上限 B）"
        _add(records, "fluff_injection", title, f"{title} 2026新款 顺丰包邮 正品保障", expected, f"促销语注入不应影响同款判定{reason}")
        punctuation = title.replace("【", "(").replace("】", ") ").replace("（", " ").replace("）", " ")
        _add(records, "punctuation_variant", title, punctuation, expected, f"括号/标点变体{reason}")
        _add(records, "fullwidth_variant", title, title.translate(_FULLWIDTH_TABLE), expected, f"全角字符不应影响判定{reason}")
        _add(records, "prefix_noise", title, f"【官方旗舰店】{title}", expected, f"前缀噪声不应影响判定{reason}")
        _add(records, "suffix_fluff", title, f"{title} 官方旗舰店 速发", expected, f"后缀噪声不应影响判定{reason}")
        rotated = _safe_rotation(title)
        if rotated is not None:
            _add(records, "segment_reorder", title, rotated, expected, f"片段重排（同成分）{reason}")

    # —— 正类：年份/别名互换 ——
    for title, has_specs in SEEDS:
        expected = "A" if has_specs else "B"
        if "2026" in title:
            _add(records, "year_swap", title, title.replace("2026", "2027"), expected, "年份营销词更换不影响同款")
        for cn, en in (("索尼", "SONY"), ("苹果", "Apple")):
            if cn in title and en not in title:
                _add(records, "brand_alias_swap", title, title.replace(cn, en), expected, f"品牌别名互换（{cn}↔{en}）")
                break

    # —— 负类：型号 / 容量 / 缺型号（期望 B，逐条标注差异）——
    model_swaps = [
        (SEEDS[0][0], "N20", "N60"),
        (SEEDS[1][0], "N60", "N65"),
        (SEEDS[2][0], "A30", "A40"),
        ("索尼 WH-1000XM6 头戴式降噪耳机", "1000XM6", "1000XM5"),
        ("Apple iPhone 16 128GB 黑色 5G手机", "iPhone 16", "iPhone 15"),
    ]
    for title, old, new in model_swaps:
        _add(records, "model_swap", title, title.replace(old, new), "B", f"型号冲突（{old}→{new}）应为近似款")

    capacity_swaps = [
        ("Apple iPhone 16 128GB 黑色 5G手机", "128GB", "256GB"),
        ("Apple iPhone 16 128GB 黑色 5G手机", "128GB", "512G"),
    ]
    for title, old, new in capacity_swaps:
        _add(records, "capacity_swap", title, title.replace(old, new), "B", f"容量冲突（{old}→{new}）应为近似款")

    for title, token in ((SEEDS[0][0], "N20"), (SEEDS[2][0], "A30"), (SEEDS[1][0], "N60")):
        _add(records, "candidate_missing_model", title, title.replace(token, " "), "B", "候选缺少型号信息，应为近似款")

    # —— 负类：配件（期望 excluded）——
    for title, _ in SEEDS[:10]:
        _add(records, "accessory_wrap", title, f"适用于{title[:14]} 保护套 硅胶防摔", "excluded", "配件类候选应被排除")
        _add(records, "accessory_suffix", title, f"{title} 专用保护套 收纳盒", "excluded", "配件类候选应被排除")

    # —— 负类：品牌冲突（期望 excluded）——
    brand_swaps = [
        (SEEDS[0][0], "QCY", "漫步者"),
        (SEEDS[1][0], "QCY", "EDIFIER"),
        ("索尼 WH-1000XM6 头戴式降噪耳机", "索尼", "Bose"),
        ("iKF Nano 空间音频主动降噪头戴式耳机无线电竞游戏适用", "iKF", "QCY"),
    ]
    for title, old, new in brand_swaps:
        _add(records, "brand_swap", title, title.replace(old, new), "excluded", f"品牌冲突（{old}→{new}）应被排除")

    # —— 负类：跨品类无关（期望 excluded）——
    for index, (title, _) in enumerate(SEEDS[:10]):
        for offset in range(4):
            other = SEEDS_OTHER[(index + offset) % len(SEEDS_OTHER)]
            _add(records, "unrelated", title, other, "excluded", "跨品类无关商品应被排除")
    for index, other in enumerate(SEEDS_OTHER):
        for offset in range(4):
            title = SEEDS[(index + offset) % 10][0]
            _add(records, "unrelated", other, title, "excluded", "跨品类无关商品应被排除")

    # —— 边界：无规格锚点（期望 B）——
    no_spec = [
        ("蓝牙耳机 无线入耳式 长续航", "蓝牙耳机 无线入耳式"),
        ("保暖内衣 男款 加绒", "保暖内衣 男款 加绒加厚"),
        ("陶瓷马克杯 大容量 家用", "陶瓷马克杯 家用 大容量 带盖"),
    ]
    for source, candidate in no_spec:
        _add(records, "no_spec_cap_b", source, candidate, "B", "无规格佐证时按设计上限 B")

    # —— 边界：可疑低价（期望 C）——
    suspicious = [
        (SEEDS[0][0], "300", "30"),
        (SEEDS[1][0], "200", "80"),
        ("Apple iPhone 16 128GB 黑色 5G手机", "5000", "1800"),
    ]
    for title, source_price, candidate_price in suspicious:
        _add(
            records,
            "suspicious_price",
            title,
            title,
            "C",
            "同款但价格异常低，应标可疑低价",
            source_price=source_price,
            candidate_price=candidate_price,
        )

    # —— 真实跨平台对（人工确认的正确匹配，期望 A）——
    _add(
        records,
        "real_cross_platform",
        "【QCY】 N20降噪蓝牙耳机无线入耳式2026新款长续航游戏适用小米苹果",
        "QCY N20 主动降噪蓝牙耳机 无线入耳式 超长续航 通话降噪 2026新款",
        "A",
        "真实跨平台同款（拼多多↔京东风格的标题差异）",
    )
    _add(
        records,
        "real_accessory_pair",
        "适用于SONY索尼WH-1000XM4头戴式蓝牙耳机保护套WH-1000XM3/2横梁头梁套",
        "适用跨境索尼WF-1000XM5保护套SONY耳机壳防摔减震收纳软盒仓tpu",
        "B",
        "真实配件对：型号不同（1000XM4 vs 1000XM5）应为近似款",
    )

    return records


def main() -> None:
    records = build_records()
    OUTPUT.write_text(
        "\n".join(json.dumps(asdict(record), ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    from collections import Counter

    by_kind = Counter(record.kind for record in records)
    print(f"已生成 {len(records)} 条 → {OUTPUT}")
    for kind, count in sorted(by_kind.items()):
        print(f"  - {kind}: {count}")


if __name__ == "__main__":
    main()
