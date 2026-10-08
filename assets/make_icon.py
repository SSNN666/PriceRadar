"""生成 PriceRadar 应用图标 260x260（拼多多开放平台创建应用要求）。

用法：
    python make_icon.py
输出：
    assets/icon-260.png
"""

from pathlib import Path

from PIL import Image, ImageDraw

W = 260          # 目标尺寸（平台要求 260x260）
S = 4            # 超采样倍数，缩小时得到平滑边缘
BG = "#0C447C"   # 深蓝底
SWEEP = "#185FA5"  # 扫掠扇形
LINE = "#FFFFFF"
DOT = "#E24B4A"  # 中心红点（中国习惯：醒目用红）

img = Image.new("RGB", (W * S, W * S), BG)
d = ImageDraw.Draw(img)

cx, cy = 130 * S, 128 * S
R = 96

# 扫掠扇形：从正上方顺时针扫到右上约 20°
d.pieslice(
    [cx - R * S, cy - R * S, cx + R * S, cy + R * S],
    start=-90,
    end=-20,
    fill=SWEEP,
)

# 三层同心环
for r, w in ((R, 5), (62, 4), (28, 4)):
    d.ellipse(
        [cx - r * S, cy - r * S, cx + r * S, cy + r * S],
        outline=LINE,
        width=w * S,
    )

# 扫掠指针
d.line([cx, cy, cx + int(R * 0.94 * S), cy - int(R * 0.34 * S)], fill=LINE, width=6 * S)

# 中心红点
cr = 14
d.ellipse([cx - cr * S, cy - cr * S, cx + cr * S, cy + cr * S], fill=DOT)

# 圆角裁切
mask = Image.new("L", (W * S, W * S), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, W * S - 1, W * S - 1], radius=56 * S, fill=255)
img.putalpha(mask)

img = img.resize((W, W), Image.LANCZOS)

out = Path(__file__).resolve().parent / "icon-260.png"
img.save(out, "PNG")
print(f"saved: {out}  size={img.size}  bytes={out.stat().st_size}")
