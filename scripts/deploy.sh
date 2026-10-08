#!/usr/bin/env bash
# 真价雷达 · 服务器部署/更新脚本（在服务器上运行，非 Windows）
# 首次部署前请按 docs/deployment-runbook.md 完成环境准备。
# 用法：bash scripts/deploy.sh
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/priceradar}"
VENV="$APP_DIR/.venv"
PY="$VENV/bin/python"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "==> 拉取最新代码"
cd "$APP_DIR"
git pull --ff-only

echo "==> 安装/更新依赖（清华镜像）"
"$PY" -m pip install -q -i "$MIRROR" -e ".[dev]"

echo "==> 测试护栏（不过不上线）"
"$PY" -m pytest -q || { echo "❌ 测试未通过，中止部署"; exit 1; }

echo "==> 重启服务"
sudo systemctl restart priceradar

echo "==> 健康检查"
sleep 2
if curl -fsS "http://127.0.0.1:8000/api/watchlist-status" >/dev/null; then
    echo "✅ 部署完成，服务正常"
else
    echo "❌ 服务未响应，请检查：journalctl -u priceradar -n 50"
    exit 1
fi
