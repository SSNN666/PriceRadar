"""金额格式化（展示层唯一入口）。"""

from decimal import Decimal

FEN = Decimal("0.01")


def fmt_amount(value: Decimal) -> str:
    """人类可读金额：50.00 -> '50'，9.90 -> '9.9'，0.05 -> '0.05'。"""
    text = format(value.quantize(FEN), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"
