"""Qwen-VL 连通性检查（识图模块探针）。

用法：
    python scripts/check_qwen_vl.py 商品图.jpg            # 商品识别
    python scripts/check_qwen_vl.py 券截图.png --coupon   # 券识别

读取凭证：DASHSCOPE_API_KEY（环境变量 > 项目根 .env；未配置时会提示申请入口）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from priceradar.vision import extract_coupons, extract_product
from priceradar.vision.vlm import VisionError

_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="Qwen-VL 识图连通性检查")
    parser.add_argument("image", type=Path, help="图片路径（jpg/png/webp）")
    parser.add_argument("--coupon", action="store_true", help="按券截图模式识别（默认商品模式）")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"图片不存在：{args.image}")
        sys.exit(1)
    media_type = _MEDIA_TYPES.get(args.image.suffix.lower(), "image/jpeg")
    payload = args.image.read_bytes()

    try:
        result = extract_coupons(payload, media_type) if args.coupon else extract_product(payload, media_type)
    except VisionError as exc:
        print(f"识别失败：{exc}")
        print("（未配置 key 时：阿里云百炼控制台 → API-KEY 管理 → 创建后写入 .env 的 DASHSCOPE_API_KEY）")
        sys.exit(1)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
