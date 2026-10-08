"""京粉频道池缓存：定期批量拉取 → 落盘 → 查询走本地。

**为什么要缓存**（2026-10-07 实测）：京东侧没有关键词搜索接口，每次查询都要现场
按频道翻页拉取（3 频道 × 4 页 = 12 次调用），查询延迟 2.4 秒。改为**定期批量刷新一次**
落盘后，查询读本地——延迟降到 ~90ms（约 25 倍），且**全池都在本地，命中不足时
可直接全池搜索，不必再额外拉取**（实测「运动鞋」在路由外的频道，现场版要多花 4 秒）。

⚠️ 关于限流的一个更正：本模块初版把 *"当前duoId因大量调取商品接口但出单较少被
限制使用"* 归因于京东，**这是错的——"duoId" 是拼多多（多多客）的术语，那是拼多多的限流**。
实测 JD 侧在完成 80 次刷新调用后仍正常。缓存对京东的价值是**延迟与覆盖**，
不是规避限流；真正需要留意调用量的是拼多多侧。
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from priceradar.config import project_root
from priceradar.domain.models import Offer
from priceradar.sources.jd import JdApiError, JdClient
from priceradar.sources.jd_channels import CHANNEL_LABELS

DEFAULT_CACHE = project_root() / "data" / "jd_pool.json"

# 刷新参数：4 页 × 20 频道 ≈ 80 次调用已实测可行（JD 侧未触发任何限制）
DEFAULT_PAGES = 4
DEFAULT_PAGE_SIZE = 50
DEFAULT_MAX_WORKERS = 4


def _fetch_channel(client: JdClient, elite_id: int, pages: int, page_size: int) -> list[Offer]:
    """拉取单个频道的多页（翻到最后一页会抛错，视为正常终止）。"""
    offers: list[Offer] = []
    for page in range(1, pages + 1):
        try:
            batch = client.jingfen_goods(elite_id=elite_id, page_index=page, page_size=page_size)
        except JdApiError:
            break
        if not batch:
            break
        offers.extend(batch)
    return offers


def refresh(
    cache_path: Path = DEFAULT_CACHE,
    *,
    channels: list[int] | None = None,
    pages: int = DEFAULT_PAGES,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict:
    """刷新频道池缓存。返回 {"channels": n, "offers": n, "errors": [...]}。"""
    client = JdClient.from_env()
    targets = channels if channels is not None else list(CHANNEL_LABELS)
    errors: list[str] = []

    def safe(elite_id: int) -> tuple[int, list[Offer]]:
        try:
            return elite_id, _fetch_channel(client, elite_id, pages, page_size)
        except (JdApiError, OSError) as exc:
            errors.append(f"频道 {elite_id}：{exc}")
            return elite_id, []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(safe, targets))

    pool: dict[str, Offer] = {}
    for _elite, offers in results:
        for offer in offers:
            pool.setdefault(offer.sku_id, offer)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "channels": [e for e, batch in results if batch],
        "offers": [offer.model_dump(mode="json") for offer in pool.values()],
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"channels": len(payload["channels"]), "offers": len(pool), "errors": errors}


def load_pool(cache_path: Path = DEFAULT_CACHE) -> list[Offer]:
    """读取缓存的频道池（不存在或损坏 → 空列表，由调用方决定降级策略）。"""
    if not cache_path.exists():
        return []
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return [Offer.model_validate(item) for item in payload.get("offers", [])]
    except (json.JSONDecodeError, ValueError, OSError):
        return []


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="刷新京东京粉频道池缓存")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES, help=f"每频道页数（默认 {DEFAULT_PAGES}）")
    parser.add_argument("--report", action="store_true", help="只看缓存现状，不刷新")
    args = parser.parse_args(argv)

    if args.report:
        offers = load_pool()
        if not offers:
            print("缓存不存在或为空——运行 `python -m priceradar.jobs.jd_pool` 刷新")
            return 0
        print(f"缓存商品数：{len(offers)}")
        with_signal = sum(1 for o in offers if o.signals and o.signals.comments is not None)
        print(f"其中带评论数（可用于质量评估）：{with_signal}")
        return 0

    try:
        summary = refresh(pages=args.pages)
    except JdApiError as exc:
        print(f"刷新失败：{exc}")
        return 1
    print(f"刷新完成：{summary['channels']} 个频道，{summary['offers']} 条商品")
    for error in summary["errors"]:
        print(f"  △ {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
