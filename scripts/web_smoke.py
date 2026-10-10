"""无头浏览器冒烟：用真实浏览器渲染真实页面，检查关键元素与报错。

## 为什么需要它

有些 bug **只在「Web 路径 + 真工具 + 真模型」三者同时在场时才复现**。已经踩过一次：

> 为修「界面上的京东频道设置不生效」而加的 `tool_defaults`，被无差别塞进了每个
> 工具调用——`scan_risk` 收到一个它不认的参数、被参数白名单拒绝，整个风险扫描报废。
> 而 CLI 不传 `tool_defaults`（一路正常）、单元测试只桩了 `run_action`（没走到
> 白名单那层）、HTTP 层测试只断言状态码——**三层测试全绿，功能是坏的**。
> 最后是无头浏览器截图上那句「工具一直报参数错误」露的马脚。

所以这个脚本把那次人工验证固化成可重复的一步。

## 它检查什么

- 页面真的能渲染（不是白屏、不是 500）
- `#error` 里没有内容（我们的前端把未捕获异常写在这里）
- Agent 面板出现了、且**每个工具调用都是 ✓**（有 ✗ 就说明某个工具在报错）
- 决策卡/报价渲染出来了
- 顺带存一张截图，人眼可复核

## 它检查不了什么

按钮点击、文件上传、剪贴板授权这些要交互的路径——那仍然需要人点一遍。

## 用法

    python scripts/web_smoke.py                 # 起服务 → 渲染 → 检查 → 关服务
    python scripts/web_smoke.py --keep          # 保留服务与截图，便于手动排查
    python scripts/web_smoke.py --query "..."   # 换一个查询

需要：真实网络 + `.env` 里配好凭证 + 系统装了 Edge 或 Chrome。
**不进 CI**——它会真的调用外部 API（有花费与拼多多配额）。
"""

from __future__ import annotations

import argparse
import http.client
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BROWSER_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)

DEFAULT_QUERY = "QCY N20 哪里买划算，顺便有没有踩坑的"


def find_browser() -> str:
    for candidate in BROWSER_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in ("msedge", "google-chrome", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("找不到 Edge/Chrome。装一个，或用 --browser 指定路径。")


def wait_for_port(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def start_server(port: int) -> subprocess.Popen:
    python = sys.executable
    process = subprocess.Popen(
        [python, "-m", "priceradar.service.api", "--no-open", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if not wait_for_port(port):
        process.terminate()
        raise SystemExit("服务没能在 30 秒内起来；用 --keep 看看它卡在哪。")
    return process


def render_dom(browser: str, url: str, timeout_ms: int) -> str:
    """让浏览器把渲染后的 DOM 打出来（--dump-dom）。"""
    result = subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--virtual-time-budget=%d" % timeout_ms,
            "--dump-dom",
            url,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_ms / 1000 + 60,
    )
    return result.stdout or ""


def screenshot(browser: str, url: str, out: Path, timeout_ms: int, height: int = 1000) -> None:
    subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            f"--window-size=1100,{height}",
            f"--virtual-time-budget={timeout_ms}",
            f"--screenshot={out}",
            url,
        ],
        capture_output=True,
        timeout=timeout_ms / 1000 + 60,
    )


# 前后都允许有别的属性：`<div id="error" class="...">` 与 `<div class="..." id="error">`
# 都要能匹配。**属性的书写顺序不该影响判定**——只认 `id="error"` 这个事实本身。
_ERROR_DIV = re.compile(r'<div[^>]*\bid="error"[^>]*>(.*?)</div>', re.S)


def error_text(dom: str) -> str | None:
    """`#error` 里的内容；**元素本身不存在时返回 None**。

    用正则取内容，而不是匹配固定的属性串——属性顺序一变，那种检查会静默失效
    （要么永远通过、要么永远失败），比没有检查更糟。

    ⚠️ 找不到容器必须返回 None 而不是空串：那说明页面根本没渲染出来，
    返回空串会让「没有报错」这条**假通过**。
    """
    match = _ERROR_DIV.search(dom)
    return None if match is None else match.group(1).strip()


def check(dom: str) -> list[tuple[str, bool, str]]:
    """从渲染后的 DOM 里挑出该有的与不该有的。返回 [(说明, 是否通过, 现场)]。"""
    error = error_text(dom)
    failed_tools = dom.count("agent-tool bad")
    succeeded_tools = dom.count("agent-tool ok")

    return [
        (
            "没有报错（#error 为空）",
            error == "",
            "页面里没有 #error 容器（可能根本没渲染出来）" if error is None else f"#error 里有内容：{error[:120]}",
        ),
        ("Agent 面板出现", "Agent 正在工作" in dom or "agent-tool" in dom, "没找到 Agent 面板"),
        # 任何工具失败都会渲染成 agent-tool bad——这正是发现 tool_defaults 污染的那条线索
        ("没有工具调用失败", failed_tools == 0, f"{failed_tools} 个工具标红"),
        ("至少有一个工具成功", succeeded_tools >= 1, "一个工具都没成功"),
        ("给出了最终答复", "agent-text" in dom or "最终答复" in dom, "没有答复文本"),
        ("决策卡渲染出来了", "dtier" in dom or "推荐" in dom, "没有决策卡"),
        ("报价列表渲染出来了", "offer" in dom, "没有报价条目"),
        ("成本已计量", "缓存命中" in dom, "元信息里没有 token / 缓存字段"),
        ("券面板渲染出来了", "我的优惠券" in dom, "没有券面板"),
    ]


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="无头浏览器冒烟（真实渲染，会调用外部 API）")
    parser.add_argument("--port", type=int, default=8750)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--browser", default=None, help="浏览器可执行文件路径（默认自动找）")
    parser.add_argument("--wait-ms", type=int, default=45000, help="给页面多少虚拟时间跑完查询")
    parser.add_argument("--keep", action="store_true", help="保留服务与截图，便于手动排查")
    parser.add_argument("--out", type=Path, default=None, help="截图输出路径")
    args = parser.parse_args(argv)

    browser = args.browser or find_browser()
    out = args.out or Path(tempfile.gettempdir()) / "priceradar_smoke.png"
    print(f"浏览器：{browser}")

    process = start_server(args.port)
    try:
        url = f"http://127.0.0.1:{args.port}/?" + urllib.parse.urlencode({"q": args.query})
        print(f"渲染：{url}")
        dom = render_dom(browser, url, args.wait_ms)
        if not dom.strip():
            raise SystemExit("浏览器没吐出 DOM——检查 --browser 指向的可执行文件对不对。")

        results = check(dom)
        for name, passed, detail in results:
            print(f"{'✓' if passed else '✗'} {name}" + (f"  → {detail}" if detail and not passed else ""))

        screenshot(browser, url, out, args.wait_ms)
        print(f"截图：{out}")

        failed = [name for name, passed, _ in results if not passed]
        print(f"\n{len(results) - len(failed)}/{len(results)} 通过")
        if failed:
            sys.exit(1)
    finally:
        if args.keep:
            print(f"服务仍在 {args.port} 上跑（Ctrl+C 结束，或手动 kill）")
        else:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
