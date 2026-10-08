@echo off
setlocal
cd /d "%~dp0"
title 真价雷达 PriceRadar

echo ============================================
echo   真价雷达 PriceRadar - 一键启动
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境 .venv，请先执行：
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -e ".[dev]"
    pause
    exit /b 1
)

echo 正在启动服务，浏览器会自动打开 http://127.0.0.1:8000
echo （保持本窗口开启，关闭本窗口即停止服务）
echo.

".venv\Scripts\python.exe" -m priceradar.service.api %*

echo.
echo 服务已停止。
pause
