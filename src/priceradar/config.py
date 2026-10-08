"""环境变量加载：环境变量优先，其次项目根目录 .env（gitignored）。"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """src/priceradar/config.py -> 项目根目录。"""
    return Path(__file__).resolve().parent.parent.parent


def load_env(*names: str) -> dict[str, str]:
    """读取指定变量：环境变量 > .env 文件；未找到返回空字符串。"""
    file_values: dict[str, str] = {}
    env_file = project_root() / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw = line.partition("=")
            file_values[key.strip()] = raw.split(" #", 1)[0].strip()
    return {name: os.environ.get(name) or file_values.get(name, "") for name in names}
